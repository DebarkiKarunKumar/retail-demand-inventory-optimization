"""Fixed-origin naive baselines and validation-only evaluation.

Run:
    python -m src.forecast_baselines

Training forecasts are generated before validation actuals are loaded.
CSV parsing necessarily tokenizes records, but test-row sales fields are
never accessed, converted, retained, summarized, plotted, or evaluated.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import tempfile
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from numbers import Real
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluate import calculate_metrics, evaluate_records


ROOT = Path(__file__).resolve().parents[1]
STORE = "CA_1"
DEPARTMENT = "FOODS_3"
SELECTED_COUNT = 20
TRAIN_END = 1857
VALIDATION_START = 1858
VALIDATION_END = 1885
TEST_START = 1886
HISTORY_END = 1913
HORIZON = 28
MODELS = ("naive", "seasonal_naive_7")
LEVELS = ("pooled_product_day", "aggregate_daily")

SALES_COLUMNS = (
    "id", "item_id", "store_id", "dept_id", "d",
    "day_number", "date", "sales", "split",
)
MANIFEST_COLUMNS = (
    "selection_rank", "id", "item_id", "store_id", "dept_id",
    "train_total_units", "train_positive_days", "train_zero_sales_pct",
)
PREDICTION_COLUMNS = (
    "id", "item_id", "model", "forecast_origin_day", "horizon",
    "d", "date", "actual_sales", "predicted_sales",
)
PRODUCT_COLUMNS = (
    "id", "item_id", "model", "observations",
    "mae", "rmse", "wape_pct", "bias_units",
)
SUMMARY_COLUMNS = (
    "model", "evaluation_level", "observations",
    "mae", "rmse", "wape_pct", "bias_units",
)
FORECAST_COLUMNS = (
    "id", "item_id", "model", "forecast_origin_day",
    "horizon", "d", "predicted_sales",
)
METRIC_FIELDS = ("observations", "mae", "rmse", "wape_pct", "bias_units")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def unit_value(value: Any, context: str, allow_text: bool = False) -> int:
    """Validate exact nonnegative unit sales without rounding or imputation."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{context}: booleans are invalid unit sales.")
    if isinstance(value, str):
        require(allow_text, f"{context}: numeric strings are not accepted.")
        require(
            bool(value) and value == value.strip(),
            f"{context}: missing or whitespace-padded numeric value.",
        )
        # Decimal accepts underscores; reject them as malformed source tokens.
        import re

        require(
            re.fullmatch(
                r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?",
                value,
            ) is not None,
            f"{context}: malformed or non-finite numeric value.",
        )
    else:
        require(
            isinstance(value, (Real, Decimal)),
            f"{context}: expected a real numeric value.",
        )
    try:
        number = Decimal(value) if isinstance(value, (str, Decimal)) else Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"{context}: invalid numeric value.") from exc
    require(number.is_finite(), f"{context}: value must be finite.")
    require(number >= 0, f"{context}: value must be nonnegative.")
    require(
        number == number.to_integral_value(),
        f"{context}: fractional unit sales are invalid.",
    )
    return int(number)


def training_values(history: Any, minimum: int) -> list[int]:
    if isinstance(history, (str, bytes, Mapping, set, frozenset)):
        raise ValueError("Training history must be an ordered numeric sequence.")
    if isinstance(history, np.ndarray):
        require(history.ndim == 1, "Training history must be one-dimensional.")
    try:
        values = list(history)
    except (TypeError, ValueError) as exc:
        raise ValueError("Training history must be an iterable sequence.") from exc
    require(
        len(values) >= minimum,
        f"Insufficient training history: need at least {minimum} observations.",
    )
    return [
        unit_value(value, f"Training observation {index}")
        for index, value in enumerate(values)
    ]


def naive_forecast(history: Any) -> list[int]:
    """Repeat the final observed training sale for all 28 forecast days."""
    values = training_values(history, minimum=1)
    return [values[-1]] * HORIZON


def seasonal_naive_forecast(history: Any) -> list[int]:
    """Repeat the final seven observed training sales four times."""
    values = training_values(history, minimum=7)
    return values[-7:] * (HORIZON // 7)


def checked_header(reader: Any, required: tuple[str, ...], source: str) -> list[str]:
    header = next(reader, [])
    require(bool(header), f"{source}: empty CSV header.")
    require(all(name.strip() for name in header), f"{source}: blank column header.")
    require(len(header) == len(set(header)), f"{source}: duplicate column headers.")
    missing = [name for name in required if name not in header]
    require(not missing, f"{source}: missing columns {missing}.")
    return header


def reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def load_manifest_and_metadata(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = root / "data/processed/selected_products.csv"
    metadata_path = root / "data/processed/selection_metadata.json"

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        header = checked_header(reader, MANIFEST_COLUMNS, manifest_path.name)
        positions = {name: header.index(name) for name in MANIFEST_COLUMNS}
        records = []
        for number, row in enumerate(reader, start=1):
            require(
                len(row) == len(header),
                f"Manifest record {number}: incorrect field count.",
            )
            records.append({name: row[index] for name, index in positions.items()})

    manifest = pd.DataFrame(records, columns=MANIFEST_COLUMNS, dtype=object)
    require(
        len(manifest) == SELECTED_COUNT,
        f"Approved scope requires exactly {SELECTED_COUNT} selected series; "
        f"manifest contains {len(manifest)}.",
    )
    for column in ("id", "item_id", "store_id", "dept_id"):
        require(
            all(isinstance(value, str) and value.strip() for value in manifest[column]),
            f"Manifest has missing {column} values.",
        )
    require(not manifest["id"].duplicated().any(), "Manifest IDs are duplicated.")
    require(
        manifest["store_id"].eq(STORE).all()
        and manifest["dept_id"].eq(DEPARTMENT).all(),
        "Manifest scope must be CA_1 / FOODS_3.",
    )
    for column in ("selection_rank", "train_total_units", "train_positive_days"):
        manifest[column] = pd.Series(
            [
                unit_value(value, f"Manifest {column}", allow_text=True)
                for value in manifest[column]
            ],
            dtype=object,
        )
    require(
        manifest["selection_rank"].tolist() == list(range(1, SELECTED_COUNT + 1)),
        "Manifest selection ranks must be ordered 1 through 20.",
    )
    percentages = []
    for value in manifest["train_zero_sales_pct"]:
        try:
            percentage = float(value)
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid manifest zero-sales percentage.") from exc
        require(
            math.isfinite(percentage) and 0 <= percentage <= 100,
            "Manifest zero-sales percentages must be finite and within 0–100.",
        )
        percentages.append(percentage)
    manifest["train_zero_sales_pct"] = percentages

    with metadata_path.open("r", encoding="utf-8-sig") as handle:
        metadata = json.load(handle, parse_constant=reject_json_constant)
    require(isinstance(metadata, dict), "Selection metadata must be an object.")
    require(
        type(metadata.get("actual_selected_series_count")) is int
        and metadata["actual_selected_series_count"] == len(manifest),
        "Metadata selected-series count disagrees with the manifest.",
    )
    require(
        metadata.get("selected_series_ids_in_rank_order") == manifest["id"].tolist(),
        "Metadata selected IDs disagree with manifest rank order.",
    )
    require(
        metadata.get("store_id") == STORE and metadata.get("dept_id") == DEPARTMENT,
        "Metadata scope disagrees with CA_1 / FOODS_3.",
    )
    return manifest, metadata


def load_period(
    sales_path: Path, manifest: pd.DataFrame, period: str
) -> pd.DataFrame:
    """Stream one requested period, filtering before accessing sales.

    Two filtered passes are intentional: validation actuals are not obtained
    until forecasts have been generated from the first, training-only pass.
    """
    require(period in ("train", "validation"), "Only train/validation may be loaded.")
    first, last = (
        (1, TRAIN_END) if period == "train"
        else (VALIDATION_START, VALIDATION_END)
    )
    manifest_map = manifest.set_index("id").to_dict("index")
    records = []
    seen: set[tuple[str, int]] = set()
    last_seen: dict[str, int] = {}

    with sales_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        header = checked_header(reader, SALES_COLUMNS, sales_path.name)
        positions = {name: header.index(name) for name in SALES_COLUMNS}

        for number, row in enumerate(reader, start=1):
            require(
                len(row) == len(header),
                f"Sales record {number}: incorrect field count.",
            )
            # Only routing fields are inspected before the demand-access gate.
            split = row[positions["split"]]
            day_text = row[positions["day_number"]]
            require(
                day_text.isascii() and day_text.isdecimal(),
                f"Sales record {number}: invalid day_number.",
            )
            day = int(day_text)
            require(1 <= day <= HISTORY_END, f"Sales record {number}: day out of range.")
            expected = (
                "train" if day <= TRAIN_END
                else "validation" if day <= VALIDATION_END
                else "test"
            )
            require(
                split == expected,
                f"Sales record {number}: split disagrees with day_number.",
            )
            if split != period or not first <= day <= last:
                continue

            # Test records never reach this point.
            series_id = row[positions["id"]]
            require(series_id in manifest_map, f"Unexpected selected ID {series_id!r}.")
            key = (series_id, day)
            require(key not in seen, f"Duplicate product-day record: {key}.")
            seen.add(key)
            require(
                day > last_seen.get(series_id, 0),
                f"{series_id}: input days are not in chronological order.",
            )
            last_seen[series_id] = day

            record = {name: row[positions[name]] for name in SALES_COLUMNS if name != "sales"}
            for field in ("id", "item_id", "store_id", "dept_id", "d", "date"):
                require(bool(record[field].strip()), f"Missing {field} at record {number}.")
            for field in ("item_id", "store_id", "dept_id"):
                require(
                    record[field] == manifest_map[series_id][field],
                    f"{series_id}: {field} disagrees with the manifest.",
                )
            require(record["d"] == f"d_{day}", f"{series_id}: invalid day ID.")
            record["day_number"] = day
            record["sales"] = unit_value(
                row[positions["sales"]],
                f"{series_id}, d_{day}",
                allow_text=True,
            )
            records.append(record)

    frame = pd.DataFrame(records, columns=SALES_COLUMNS, dtype=object)
    require(
        len(frame) == len(manifest) * (last - first + 1),
        f"{period}: incorrect total observation count.",
    )
    require(
        set(frame["id"]) == set(manifest_map),
        f"{period}: selected IDs are incomplete.",
    )
    require(
        frame["date"].str.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}").all(),
        f"{period}: dates must be ISO YYYY-MM-DD.",
    )
    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
    for series_id, group in frame.groupby("id", sort=True):
        require(
            group["day_number"].tolist() == list(range(first, last + 1)),
            f"{series_id}: missing or unordered {period} days.",
        )
    require(
        frame.groupby("day_number")["date"].nunique().eq(1).all(),
        f"{period}: inconsistent date mappings.",
    )
    calendar = (
        frame[["day_number", "date"]].drop_duplicates()
        .sort_values("day_number")
    )
    differences = calendar["date"].diff().iloc[1:].to_numpy(dtype="timedelta64[ns]")
    one_day = np.timedelta64(1, "D")
    require((differences == one_day).all(), f"{period}: calendar is not consecutive.")

    return frame.sort_values(["id", "day_number"]).reset_index(drop=True)


def generate_forecasts(training: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Generate all forecasts without accepting validation actuals."""
    records = []
    manifest_map = manifest.set_index("id").to_dict("index")
    require(
        set(training["id"]) == set(manifest_map),
        "Training IDs disagree with the manifest.",
    )
    require(
        not training.duplicated(["id", "day_number"]).any(),
        "Duplicate training product-day records.",
    )
    for series_id in sorted(manifest_map):
        group = training.loc[training["id"].eq(series_id)]
        require(
            group["day_number"].tolist() == list(range(1, TRAIN_END + 1)),
            f"{series_id}: expected ordered training days 1–1857.",
        )
        values = group["sales"].tolist()
        checked = training_values(values, TRAIN_END)
        source = manifest_map[series_id]
        positive = sum(value > 0 for value in checked)
        zero_pct = 100.0 * sum(value == 0 for value in checked) / TRAIN_END
        require(
            sum(checked) == source["train_total_units"]
            and positive == source["train_positive_days"]
            and math.isclose(
                zero_pct, source["train_zero_sales_pct"], abs_tol=1e-9, rel_tol=0
            ),
            f"{series_id}: training statistics disagree with the manifest.",
        )
        require(positive >= 28, f"{series_id}: training activity threshold not met.")

        forecasts = {
            "naive": naive_forecast(checked),
            "seasonal_naive_7": seasonal_naive_forecast(checked),
        }
        for model in MODELS:
            for horizon, predicted in enumerate(forecasts[model], start=1):
                records.append(
                    {
                        "id": series_id,
                        "item_id": source["item_id"],
                        "model": model,
                        "forecast_origin_day": f"d_{TRAIN_END}",
                        "horizon": horizon,
                        "d": f"d_{TRAIN_END + horizon}",
                        "predicted_sales": predicted,
                    }
                )
    return pd.DataFrame(records, columns=FORECAST_COLUMNS, dtype=object).sort_values(
        ["model", "id", "horizon"]
    ).reset_index(drop=True)


def align_forecasts(
    forecasts: pd.DataFrame, validation: pd.DataFrame
) -> pd.DataFrame:
    require(
        not validation.duplicated(["id", "day_number"]).any(),
        "Duplicate validation product-day records.",
    )
    actuals = {
        (row.id, row.day_number): row
        for row in validation.itertuples(index=False)
    }
    records = []
    for row in forecasts.itertuples(index=False):
        day = TRAIN_END + row.horizon
        require(
            (row.id, day) in actuals,
            f"Missing validation observation for {row.id}, d_{day}.",
        )
        actual = actuals[(row.id, day)]
        require(actual.item_id == row.item_id, "Validation item identifier mismatch.")
        records.append(
            {
                "id": row.id,
                "item_id": row.item_id,
                "model": row.model,
                "forecast_origin_day": row.forecast_origin_day,
                "horizon": row.horizon,
                "d": row.d,
                "date": actual.date.date().isoformat(),
                "actual_sales": actual.sales,
                "predicted_sales": row.predicted_sales,
            }
        )
    return pd.DataFrame(records, columns=PREDICTION_COLUMNS, dtype=object)


def evaluate_predictions(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Delegate all metric calculations to the existing evaluation module."""
    product_records = []
    summary_records = []
    for model in MODELS:
        model_rows = predictions.loc[predictions["model"].eq(model)]
        records = [
            {
                "id": row.id,
                "day_number": TRAIN_END + row.horizon,
                "actual": row.actual_sales,
                "predicted": row.predicted_sales,
            }
            for row in model_rows.itertuples(index=False)
        ]
        for series_id, group in model_rows.groupby("id", sort=True):
            metrics = calculate_metrics(
                group["actual_sales"].tolist(), group["predicted_sales"].tolist()
            )
            product_records.append(
                {
                    "id": series_id,
                    "item_id": group["item_id"].iloc[0],
                    "model": model,
                    **metrics,
                }
            )
        for level in LEVELS:
            result = evaluate_records(records, level)
            summary_records.append(
                {
                    "model": model,
                    "evaluation_level": level,
                    **{field: result[field] for field in METRIC_FIELDS},
                }
            )
    return (
        pd.DataFrame(product_records, columns=PRODUCT_COLUMNS, dtype=object)
        .sort_values(["model", "id"]).reset_index(drop=True),
        pd.DataFrame(summary_records, columns=SUMMARY_COLUMNS, dtype=object)
        .sort_values(["model", "evaluation_level"]).reset_index(drop=True),
    )


def validate_outputs(
    predictions: pd.DataFrame,
    products: pd.DataFrame,
    summary: pd.DataFrame,
    training: pd.DataFrame,
    validation: pd.DataFrame,
    manifest: pd.DataFrame,
) -> dict[str, Any]:
    count = len(manifest)
    require(tuple(predictions.columns) == PREDICTION_COLUMNS, "Prediction schema mismatch.")
    require(tuple(products.columns) == PRODUCT_COLUMNS, "Product metrics schema mismatch.")
    require(tuple(summary.columns) == SUMMARY_COLUMNS, "Summary metrics schema mismatch.")
    require(
        len(predictions) == count * len(MODELS) * HORIZON,
        "Incorrect prediction count.",
    )
    require(
        not predictions.duplicated(["id", "model", "horizon"]).any(),
        "Duplicate prediction keys.",
    )
    require(
        set(predictions["id"]) == set(manifest["id"])
        and set(predictions["model"]) == set(MODELS),
        "Prediction IDs or model names disagree with scope.",
    )
    require(
        predictions["forecast_origin_day"].eq(f"d_{TRAIN_END}").all(),
        "Forecast origin is not fixed at d_1857.",
    )
    require(
        not predictions.isna().any().any(),
        "Predictions contain missing values.",
    )
    require(
        predictions.equals(
            predictions.sort_values(["model", "id", "horizon"]).reset_index(drop=True)
        ),
        "Predictions are not deterministically sorted.",
    )
    actual_map = {
        (row.id, row.day_number): row for row in validation.itertuples(index=False)
    }
    checked_cells = 0
    for (model, series_id), group in predictions.groupby(["model", "id"], sort=True):
        require(
            group["horizon"].tolist() == list(range(1, HORIZON + 1))
            and all(type(value) is int for value in group["horizon"]),
            "Each product/model must contain integer horizons 1–28.",
        )
        history = training.loc[training["id"].eq(series_id), "sales"].tolist()
        expected = (
            [history[-1]] * HORIZON
            if model == "naive"
            else [history[-7 + index % 7] for index in range(HORIZON)]
        )
        for row, forecast in zip(group.itertuples(index=False), expected):
            day = TRAIN_END + row.horizon
            actual = actual_map[(series_id, day)]
            require(
                VALIDATION_START <= day <= VALIDATION_END
                and row.d == f"d_{day}"
                and row.date == actual.date.date().isoformat(),
                "Forecast horizon/date alignment failed.",
            )
            require(
                unit_value(row.actual_sales, "Output actual") == actual.sales
                and unit_value(row.predicted_sales, "Output prediction") == forecast,
                "Source actual or fixed-origin forecast reconciliation failed.",
            )
            checked_cells += 1

    require(len(products) == count * len(MODELS), "Incorrect per-product metric count.")
    require(
        not products.duplicated(["id", "model"]).any()
        and set(zip(products["id"], products["model"]))
        == {(series_id, model) for series_id in manifest["id"] for model in MODELS},
        "Per-product metric coverage is incorrect.",
    )
    require(products["observations"].eq(HORIZON).all(), "Product metric count is not 28.")
    require(
        len(summary) == len(MODELS) * len(LEVELS)
        and not summary.duplicated(["model", "evaluation_level"]).any()
        and set(zip(summary["model"], summary["evaluation_level"]))
        == {(model, level) for model in MODELS for level in LEVELS},
        "Summary metric coverage is incorrect.",
    )
    for row in summary.itertuples(index=False):
        expected_count = count * HORIZON if row.evaluation_level == "pooled_product_day" else HORIZON
        require(row.observations == expected_count, "Incorrect summary observation count.")
    for table in (products, summary):
        for row in table.to_dict("records"):
            for field in ("mae", "rmse", "bias_units"):
                require(
                    isinstance(row[field], Real) and math.isfinite(row[field]),
                    f"Invalid {field} result.",
                )
            require(
                row["wape_pct"] is None
                or (
                    isinstance(row["wape_pct"], Real)
                    and math.isfinite(row["wape_pct"])
                    and row["wape_pct"] >= 0
                ),
                "Invalid WAPE result.",
            )

    return {
        "status": "PASS",
        "training_observations": len(training),
        "validation_observations": len(validation),
        "reconciled_prediction_rows": checked_cells,
        "duplicate_prediction_keys": 0,
        "fixed_origin_verified": True,
        "forecast_rules_verified": True,
        "validation_actuals_reconciled": True,
        "calendar_alignment_verified": True,
        "metric_observation_counts_verified": True,
        "metrics_implementation": "src.evaluate",
        "test_sales_retained": False,
        "test_sales_evaluated": False,
    }


def comparison_chart(predictions: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 5))
    try:
        actual = (
            predictions.loc[predictions["model"].eq("naive")]
            .groupby("date", sort=True)["actual_sales"].sum()
        )
        dates = pd.to_datetime(actual.index)
        axis.plot(dates, actual.astype(float), label="Actual aggregate sales", color="#202020", linewidth=2)
        for model, label, color in (
            ("naive", "Naive forecast", "#2878a5"),
            ("seasonal_naive_7", "Weekly seasonal-naive forecast", "#c56b2c"),
        ):
            totals = (
                predictions.loc[predictions["model"].eq(model)]
                .groupby("date", sort=True)["predicted_sales"].sum()
                .reindex(actual.index)
            )
            axis.plot(dates, totals.astype(float), label=label, color=color, linestyle="--")
        axis.set(
            title="Fixed-origin baseline forecasts — validation period",
            xlabel="Validation date",
            ylabel="Aggregate unit sales per day",
        )
        axis.set_ylim(bottom=0)
        locator = mdates.AutoDateLocator(minticks=5, maxticks=9)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        axis.grid(axis="y", alpha=0.2)
        axis.legend()
        figure.tight_layout()
        figure.savefig(
            path, dpi=160, format="png",
            metadata={"Software": "Retail fixed-origin baselines"},
        )
    finally:
        plt.close(figure)


def publish(
    root: Path,
    predictions: pd.DataFrame,
    products: pd.DataFrame,
    summary: pd.DataFrame,
    metadata: dict[str, Any],
) -> list[Path]:
    output_root = root / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    destinations = [
        output_root / "evaluation/baseline_predictions.csv",
        output_root / "evaluation/baseline_metrics_by_product.csv",
        output_root / "evaluation/baseline_metrics_summary.csv",
        output_root / "evaluation/baseline_run_metadata.json",
        output_root / "figures/baseline_forecast_comparison.png",
    ]
    replaced = []
    try:
        with tempfile.TemporaryDirectory(prefix=".baseline-stage-", dir=output_root) as temporary:
            stage = Path(temporary)
            staged = [stage / path.name for path in destinations]
            for frame, path in zip((predictions, products, summary), staged[:3]):
                frame.to_csv(path, index=False, lineterminator="\n", na_rep="")
                reread = pd.read_csv(path, dtype=str, keep_default_na=False)
                require(
                    tuple(reread.columns) == tuple(frame.columns)
                    and len(reread) == len(frame),
                    f"Staged CSV verification failed: {path.name}",
                )
            staged[3].write_text(
                json.dumps(metadata, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            with staged[3].open("r", encoding="utf-8") as handle:
                require(json.load(handle) == metadata, "Staged metadata verification failed.")
            comparison_chart(predictions, staged[4])
            image = plt.imread(staged[4])
            require(image.size > 0, "Staged chart cannot be decoded.")

            hashes = []
            for path in staged:
                require(path.is_file() and path.stat().st_size > 0, f"Empty output: {path}")
                hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
            # Each replacement is atomic individually, not as a five-file set.
            for source, destination in zip(staged, destinations):
                destination.parent.mkdir(parents=True, exist_ok=True)
                source.replace(destination)
                replaced.append(destination.name)
            for destination, expected in zip(destinations, hashes):
                require(
                    hashlib.sha256(destination.read_bytes()).hexdigest() == expected,
                    f"Published output verification failed: {destination}",
                )
    except Exception as exc:
        raise RuntimeError(
            "Output publication failed. Files replaced this attempt: "
            f"{', '.join(replaced) or 'none'}. Outputs may be from different runs; "
            f"rerun successfully before using them. Cause: {exc}"
        ) from exc
    return destinations


def run_pipeline(root: Path = ROOT) -> dict[str, Any]:
    root = Path(root)
    sales_path = root / "data/processed/selected_sales.csv"
    for path in (
        sales_path,
        root / "data/processed/selected_products.csv",
        root / "data/processed/selection_metadata.json",
    ):
        require(path.is_file(), f"Required input does not exist: {path}")

    manifest, _ = load_manifest_and_metadata(root)
    training = load_period(sales_path, manifest, "train")
    forecasts = generate_forecasts(training, manifest)

    # This is deliberately the first point at which validation demand is obtained.
    validation = load_period(sales_path, manifest, "validation")
    require(
    (
        validation["date"].min().to_datetime64()
        - training["date"].max().to_datetime64()
    ) == np.timedelta64(1, "D"),
    "Training and validation calendars are not adjacent.",
    )
    predictions = align_forecasts(forecasts, validation)
    products, summary = evaluate_predictions(predictions)
    checks = validate_outputs(
        predictions, products, summary, training, validation, manifest
    )
    metadata = {
        "dataset": "Walmart M5 Forecasting Accuracy",
        "source_filenames": [
            "selected_sales.csv", "selected_products.csv", "selection_metadata.json"
        ],
        "scope": {"store_id": STORE, "dept_id": DEPARTMENT},
        "selected_series_count": len(manifest),
        "selected_series_ids": sorted(manifest["id"].tolist()),
        "model_names": list(MODELS),
        "fixed_forecast_origin": {
            "day": f"d_{TRAIN_END}",
            "date": training["date"].max().date().isoformat(),
        },
        "validation_period": {
            "start_day": f"d_{VALIDATION_START}",
            "end_day": f"d_{VALIDATION_END}",
            "start_date": validation["date"].min().date().isoformat(),
            "end_date": validation["date"].max().date().isoformat(),
        },
        "forecast_horizon": HORIZON,
        "prediction_counts": {
            "total": len(predictions),
            "per_product_per_model": HORIZON,
            "by_model": {
                model: int(predictions["model"].eq(model).sum()) for model in MODELS
            },
        },
        "metric_definitions": {
            "mae": "mean(abs(predicted - actual))",
            "rmse": "sqrt(mean((predicted - actual) ** 2))",
            "wape_pct": "sum(abs(predicted - actual)) / sum(actual) * 100",
            "undefined_wape": "null in JSON-compatible metrics; empty CSV field",
            "bias_units": "mean(predicted - actual)",
            "bias_convention": "Positive: overforecasting; negative: underforecasting",
            "implementation": "src.evaluate",
        },
        "evaluation_levels": ["per_product", *LEVELS],
        "metric_row_counts": {"per_product": len(products), "summary": len(summary)},
        "integrity_checks": checks,
        "no_test_demand_access": {
            "statement": (
                "Test rows were discarded by split/day routing before their sales "
                "field was accessed. Test sales were not converted, retained, "
                "summarized, forecast against, plotted, or evaluated."
            ),
            "test_demand_retained": False,
            "test_demand_evaluated": False,
            "reader_behavior": (
                "The CSV parser transiently tokenizes full records; excluded-row "
                "demand fields are never extracted into pipeline data structures."
            ),
        },
        "limitations_and_assumptions": [
            "The 20 products are a high-volume selected subset, not representative.",
            "Recorded sales may be constrained by unobserved inventory availability.",
            "All 28 predictions are generated at d_1857 without validation updates.",
            "Validation results do not change product membership or baseline rules.",
            "No final-test accuracy is calculated.",
            "Undefined WAPE is preserved rather than replaced with zero.",
            "Individual output replacement is atomic; the five-file set is not.",
            "Source-to-output checks cover prepared inputs, not raw M5 files.",
        ],
    }
    paths = publish(root, predictions, products, summary, metadata)
    return {"metadata": metadata, "output_paths": paths}


def main() -> int:
    try:
        result = run_pipeline()
    except Exception as exc:
        print(f"Baseline forecasting FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "Existing baseline outputs must not be interpreted as a successful current run.",
            file=sys.stderr,
        )
        return 1

    metadata = result["metadata"]
    print("Fixed-origin baseline evaluation: PASS")
    print(f"Selected series: {metadata['selected_series_count']}")
    print(f"Prediction rows: {metadata['prediction_counts']['total']}")
    print(f"Per-product metric rows: {metadata['metric_row_counts']['per_product']}")
    print(f"Summary metric rows: {metadata['metric_row_counts']['summary']}")
    print("All integrity checks passed. Test sales were not retained or evaluated.")
    for path in result["output_paths"]:
        print(f"Output: {path}")
    print("Exit code: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())