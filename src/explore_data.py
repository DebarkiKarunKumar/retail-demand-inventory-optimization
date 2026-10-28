"""Training-only exploratory analysis of the prepared M5 subset.

Run from the repository root:
    python -m src.explore_data

No validation/test sales are parsed, summarized, grouped, or plotted.
Input files and earlier-stage artifacts are never modified.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
import sys
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SALES_PATH = ROOT / "data/processed/selected_sales.csv"
MANIFEST_PATH = ROOT / "data/processed/selected_products.csv"
METADATA_PATH = ROOT / "data/processed/selection_metadata.json"

FIGURE_DIR = ROOT / "outputs/figures"
SUMMARY_PATH = ROOT / "outputs/evaluation/eda_summary.json"

STORE = "CA_1"
DEPARTMENT = "FOODS_3"
TRAIN_DAYS = 1857
VALIDATION_END = 1885
HISTORY_DAYS = 1913
MIN_POSITIVE_DAYS = 28
MAX_SERIES = 20
ROLLING_DAYS = 28

SALES_COLUMNS = (
    "id", "item_id", "store_id", "dept_id", "d",
    "day_number", "date", "sales", "split",
)
MANIFEST_COLUMNS = (
    "selection_rank", "id", "item_id", "store_id", "dept_id",
    "train_total_units", "train_positive_days", "train_zero_sales_pct",
)
WEEKDAYS = (
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
)
NUMBER_PATTERN = re.compile(
    r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$"
)
FIGURE_NAMES = (
    "aggregate_daily_demand.png",
    "product_demand_profile.png",
    "weekday_demand.png",
)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def reject_json_constant(token: str) -> None:
    raise ValueError(f"Invalid JSON numeric constant: {token}")


def read_csv_strings(path: Path, required: tuple[str, ...]) -> pd.DataFrame:
    """Preserve raw tokens and reject malformed headers before Pandas parsing."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle, strict=True), [])

    require(bool(header), f"{path.name}: empty CSV header.")
    require(
        all(name.strip() for name in header),
        f"{path.name}: blank column header.",
    )
    require(
        len(header) == len(set(header)),
        f"{path.name}: duplicate column headers.",
    )
    missing = [name for name in required if name not in header]
    require(not missing, f"{path.name}: missing required columns: {missing}.")

    # Treat parser warnings as failures: they can indicate discarded fields.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.ParserWarning)
        frame = pd.read_csv(
            path,
            encoding="utf-8-sig",
            dtype=str,
            na_filter=False,
            keep_default_na=False,
            index_col=False,
            skip_blank_lines=False,
            on_bad_lines="error",
        )
    require(
        list(frame.columns) == header,
        f"{path.name}: parsed columns differ from the inspected header.",
    )
    return frame


def require_strings(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        require(
            all(
                isinstance(value, str) and bool(value.strip())
                for value in frame[column]
            ),
            f"Missing or invalid string values in {column}.",
        )


def parse_integer(raw: Any, context: str) -> int:
    """Accept exact, finite, nonnegative integers without rounding."""
    require(
        isinstance(raw, str) and bool(raw.strip()),
        f"{context}: missing value.",
    )
    require(
        NUMBER_PATTERN.fullmatch(raw) is not None,
        f"{context}: malformed numeric token {raw[:80]!r}.",
    )
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise ValueError(f"{context}: invalid numeric value.") from exc
    require(value.is_finite(), f"{context}: non-finite value.")
    require(value >= 0, f"{context}: negative value.")
    require(
        value == value.to_integral_value(),
        f"{context}: fractional value.",
    )
    return int(value)


def parse_percentage(raw: Any, context: str) -> float:
    require(
        isinstance(raw, str) and NUMBER_PATTERN.fullmatch(raw) is not None,
        f"{context}: malformed percentage.",
    )
    value = float(raw)
    require(
        math.isfinite(value) and 0 <= value <= 100,
        f"{context}: percentage must be finite and between 0 and 100.",
    )
    return value


def expected_split(day: int) -> str:
    if day <= TRAIN_DAYS:
        return "train"
    if day <= VALIDATION_END:
        return "validation"
    return "test"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    for path in (SALES_PATH, MANIFEST_PATH, METADATA_PATH):
        require(path.is_file(), f"Required input file does not exist: {path}")

    raw = read_csv_strings(SALES_PATH, SALES_COLUMNS)

    # Mandatory leakage boundary: isolate train immediately after reading.
    # The remaining frame retains structural fields only; held-out demand
    # tokens are discarded without parsing or calculating any statistics.
    train = raw.loc[raw["split"].eq("train"), list(SALES_COLUMNS)].copy()
    structure = raw.drop(columns=["sales"]).copy()
    del raw

    manifest = read_csv_strings(MANIFEST_PATH, MANIFEST_COLUMNS)
    with METADATA_PATH.open("r", encoding="utf-8-sig") as handle:
        metadata = json.load(handle, parse_constant=reject_json_constant)
    require(isinstance(metadata, dict), "Selection metadata must be a JSON object.")
    return train, structure, manifest, metadata


def validate_inputs(
    train: pd.DataFrame,
    structure: pd.DataFrame,
    manifest: pd.DataFrame,
    metadata: dict[str, Any],
) -> list[str]:
    """Validate structure globally, but validate sales values only in train."""
    require(
        1 <= len(manifest) <= MAX_SERIES,
        "Manifest must contain between 1 and 20 selected series.",
    )
    require_strings(manifest, ("id", "item_id", "store_id", "dept_id"))
    require(not manifest["id"].duplicated().any(), "Duplicate manifest series IDs.")
    require(
        manifest["store_id"].eq(STORE).all()
        and manifest["dept_id"].eq(DEPARTMENT).all(),
        "Manifest scope must be CA_1 / FOODS_3.",
    )

    for column in (
        "selection_rank", "train_total_units", "train_positive_days"
    ):
        manifest[column] = pd.Series(
            [
                parse_integer(raw, f"Manifest {column}, row {index + 1}")
                for index, raw in enumerate(manifest[column])
            ],
            index=manifest.index,
            dtype=object,
        )
    manifest["train_zero_sales_pct"] = [
        parse_percentage(raw, f"Manifest zero-sales percentage, row {index + 1}")
        for index, raw in enumerate(manifest["train_zero_sales_pct"])
    ]
    require(
        manifest["selection_rank"].tolist() == list(range(1, len(manifest) + 1)),
        "Manifest must be ordered by consecutive selection ranks beginning at 1.",
    )

    ids = manifest["id"].tolist()
    metadata_ids = metadata.get("selected_series_ids_in_rank_order")
    require(
        isinstance(metadata_ids, list)
        and all(isinstance(value, str) for value in metadata_ids)
        and metadata_ids == ids,
        "Metadata selected IDs must exactly match manifest rank order.",
    )
    require(
        type(metadata.get("actual_selected_series_count")) is int
        and metadata["actual_selected_series_count"] == len(ids),
        "Metadata selected-series count disagrees with the manifest.",
    )
    require(
        metadata.get("store_id") == STORE
        and metadata.get("dept_id") == DEPARTMENT,
        "Metadata scope must be CA_1 / FOODS_3.",
    )

    structural_strings = (
        "id", "item_id", "store_id", "dept_id", "d", "date", "split"
    )
    require_strings(structure, structural_strings)
    require(
        set(structure["id"]) == set(ids),
        "Prepared-sales IDs disagree with the selected manifest.",
    )
    require(
        structure["store_id"].eq(STORE).all()
        and structure["dept_id"].eq(DEPARTMENT).all(),
        "Prepared-sales scope disagrees with the requested scope.",
    )

    structure["day_number"] = [
        parse_integer(raw, f"Prepared day_number, row {index + 1}")
        for index, raw in enumerate(structure["day_number"])
    ]
    require(
        structure["day_number"].between(1, HISTORY_DAYS).all(),
        "Prepared day_number must be between 1 and 1913.",
    )
    require(
        not structure.duplicated(["id", "day_number"]).any(),
        "Duplicate prepared (id, day_number) keys.",
    )
    require(
        structure["d"].tolist()
        == [f"d_{day}" for day in structure["day_number"]],
        "Prepared day IDs disagree with day_number.",
    )
    require(
        structure["split"].tolist()
        == [expected_split(day) for day in structure["day_number"]],
        "Split labels disagree with the approved day boundaries.",
    )

    require(
        structure["date"].str.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}").all(),
        "Dates must be populated ISO YYYY-MM-DD strings.",
    )
    structure["date"] = pd.to_datetime(
        structure["date"], format="%Y-%m-%d", errors="raise"
    )
    date_counts = structure.groupby("day_number", sort=True)["date"].nunique()
    require(
        date_counts.eq(1).all(),
        "A day_number maps to more than one calendar date.",
    )
    calendar = (
        structure[["day_number", "date"]]
        .drop_duplicates()
        .sort_values("day_number")
    )
    require(
        calendar["day_number"].tolist() == list(range(1, HISTORY_DAYS + 1)),
        "Prepared calendar does not cover all historical day numbers.",
    )
    import numpy as np
    require(
        (
            calendar["date"].diff().iloc[1:].to_numpy(dtype="timedelta64[ns]")
            == np.timedelta64(1, "D")
        ).all(),
        "Prepared calendar dates must increase by exactly one day.",
    )

    manifest_by_id = manifest.set_index("id")
    for series_id, group in structure.groupby("id", sort=True):
        require(
            sorted(group["day_number"]) == list(range(1, HISTORY_DAYS + 1)),
            f"{series_id}: incomplete historical structural coverage.",
        )
        for column in ("item_id", "store_id", "dept_id"):
            require(
                group[column].eq(manifest_by_id.at[series_id, column]).all(),
                f"{series_id}: {column} disagrees with the manifest.",
            )

    require(
        set(train["id"]) == set(ids),
        "Training IDs disagree with selected IDs.",
    )
    require(
        len(train) == len(ids) * TRAIN_DAYS,
        "Training row count must equal selected series count times 1857.",
    )

    # Reuse validated structural types without touching held-out sales.
    train["day_number"] = structure.loc[train.index, "day_number"]
    train["date"] = structure.loc[train.index, "date"]
    require(
        not train.duplicated(["id", "day_number"]).any(),
        "Duplicate training product-day records.",
    )
    for series_id, group in train.groupby("id", sort=True):
        require(
            sorted(group["day_number"]) == list(range(1, TRAIN_DAYS + 1)),
            f"{series_id}: training must contain days 1 through 1857 exactly once.",
        )

    # Object dtype preserves Python integers, including valid integer zeros.
    train["sales"] = pd.Series(
        [
            parse_integer(raw, f"Training sales at source row {index + 2}")
            for index, raw in zip(train.index, train["sales"])
        ],
        index=train.index,
        dtype=object,
    )
    train.sort_values(["id", "day_number"], inplace=True)
    train.reset_index(drop=True, inplace=True)
    return ids


def calculate_analysis(
    train: pd.DataFrame,
    manifest: pd.DataFrame,
    ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    aggregate = (
        train.groupby("date", sort=True)["sales"]
        .agg(lambda values: sum(values.tolist()))
        .rename("total_units")
        .reset_index()
    )
    require(
        len(aggregate) == TRAIN_DAYS,
        "Aggregate demand must contain exactly 1857 training dates.",
    )

    # Full trailing windows only: the first 27 rolling values are unavailable.
    # center=False ensures no future observations enter a rolling average.
    aggregate["trailing_28_day_average"] = aggregate["total_units"].rolling(
        window=ROLLING_DAYS,
        min_periods=ROLLING_DAYS,
        center=False,
    ).mean()

    product_records = []
    manifest_by_id = manifest.set_index("id")
    for series_id in ids:
        group = train.loc[train["id"].eq(series_id)]
        values = group["sales"].tolist()
        zero_days = sum(value == 0 for value in values)
        positive_days = sum(value > 0 for value in values)
        total = sum(values)
        zero_pct = 100.0 * zero_days / TRAIN_DAYS
        require(
            len(values) == TRAIN_DAYS and zero_days + positive_days == TRAIN_DAYS,
            f"{series_id}: invalid training denominator.",
        )
        require(
            total == manifest_by_id.at[series_id, "train_total_units"],
            f"{series_id}: training total disagrees with the manifest.",
        )
        require(
            positive_days == manifest_by_id.at[series_id, "train_positive_days"],
            f"{series_id}: positive-day count disagrees with the manifest.",
        )
        require(
            positive_days >= MIN_POSITIVE_DAYS,
            f"{series_id}: selected series fails the training activity threshold.",
        )
        require(
            math.isclose(
                zero_pct,
                manifest_by_id.at[series_id, "train_zero_sales_pct"],
                rel_tol=0,
                abs_tol=1e-9,
            ),
            f"{series_id}: zero-sales percentage disagrees with the manifest.",
        )
        product_records.append(
            {
                "selection_rank": int(manifest_by_id.at[series_id, "selection_rank"]),
                "id": series_id,
                "item_id": manifest_by_id.at[series_id, "item_id"],
                "training_observations": len(values),
                "total_training_units": total,
                "mean_daily_units": statistics.mean(values),
                "std_daily_units": statistics.stdev(values),
                "zero_sales_days": zero_days,
                "zero_sales_percentage": zero_pct,
                "positive_sales_days": positive_days,
            }
        )

    products = pd.DataFrame(product_records)
    aggregate_values = aggregate["total_units"].tolist()
    aggregate_total = sum(aggregate_values)
    require(
        sum(record["total_training_units"] for record in product_records)
        == aggregate_total,
        "Product totals do not reconcile with aggregate totals.",
    )
    require(
        aggregate_total == sum(train["sales"].tolist()),
        "Aggregate total does not reconcile with training observations.",
    )

    # Calendar-derived weekday numbers guarantee Monday-to-Sunday ordering.
    aggregate["weekday_number"] = aggregate["date"].dt.dayofweek
    weekday_records = []
    for number, label in enumerate(WEEKDAYS):
        values = aggregate.loc[
            aggregate["weekday_number"].eq(number), "total_units"
        ].tolist()
        require(bool(values), f"No training dates found for {label}.")
        weekday_records.append(
            {
                "weekday": label,
                "weekday_number": number,
                "training_date_count": len(values),
                "total_units": sum(values),
                "mean_aggregate_daily_units": statistics.mean(values),
            }
        )
    require(
        sum(record["training_date_count"] for record in weekday_records)
        == TRAIN_DAYS,
        "Weekday groups do not account for every training date.",
    )
    require(
        sum(record["total_units"] for record in weekday_records)
        == aggregate_total,
        "Weekday totals do not reconcile with aggregate totals.",
    )

    zero_count = sum(record["zero_sales_days"] for record in product_records)
    require(
        zero_count == sum(value == 0 for value in train["sales"]),
        "Product zero counts do not reconcile with training observations.",
    )
    denominator = len(ids) * TRAIN_DAYS
    require(
        denominator == len(train),
        "Overall zero-sales denominator is incorrect.",
    )

    summary = {
        "status": "PASS",
        "scope": {"store_id": STORE, "dept_id": DEPARTMENT},
        "data_source": {
            "selected_sales": SALES_PATH.relative_to(ROOT).as_posix(),
            "selected_products": MANIFEST_PATH.relative_to(ROOT).as_posix(),
            "selection_metadata": METADATA_PATH.relative_to(ROOT).as_posix(),
        },
        "analysis_split": "train",
        "training_date_range": {
            "start": aggregate["date"].min().date().isoformat(),
            "end": aggregate["date"].max().date().isoformat(),
            "first_day": "d_1",
            "last_day": "d_1857",
            "calendar_day_count": len(aggregate),
        },
        "selected_series_count": len(ids),
        "training_observation_count": len(train),
        "aggregate_total_units": aggregate_total,
        "aggregate_mean_daily_units": statistics.mean(aggregate_values),
        "aggregate_std_daily_units": statistics.stdev(aggregate_values),
        "overall_zero_sales_percentage": 100.0 * zero_count / denominator,
        "overall_zero_sales_days": zero_count,
        "overall_zero_sales_denominator": denominator,
        "product_statistics": product_records,
        "weekday_statistics": weekday_records,
        "calculation_conventions": {
            "standard_deviation": "Sample standard deviation, ddof=1",
            "product_daily_denominator": TRAIN_DAYS,
            "overall_zero_sales_denominator": (
                "Selected series count multiplied by 1857 training days"
            ),
            "weekday_mean_denominator": (
                "Number of training dates belonging to that weekday"
            ),
            "rolling_average": {
                "window_days": ROLLING_DAYS,
                "minimum_observations": ROLLING_DAYS,
                "centered": False,
                "initial_unavailable_values": ROLLING_DAYS - 1,
            },
        },
        "integrity_checks": {
            "training_days_per_series": TRAIN_DAYS,
            "duplicate_product_day_records": 0,
            "training_observation_count_reconciled": True,
            "product_and_aggregate_totals_reconciled": True,
            "manifest_training_metrics_reconciled": True,
            "zero_sales_denominators_verified": True,
            "weekday_date_coverage_verified": True,
            "calendar_mapping_and_continuity_verified": True,
        },
        "coverage": {
            "demand_analysis": "Training observations only",
            "held_out_sales_parsed_or_summarized": False,
            "non_demand_structural_checks": (
                "Identifiers, dates, day numbers, keys, and split labels "
                "are checked across the prepared dataset."
            ),
            "raw_or_prepared_inputs_modified": False,
        },
        "limitations": [
            f"The {len(ids)} products form a high-volume selected subset, "
            "not a representative sample of the department or M5 dataset.",
            "Zero recorded sales do not necessarily mean zero customer demand.",
            "Inventory availability and lost sales are not observed.",
            "Weekly patterns are descriptive rather than causal and do not "
            "establish statistical significance.",
            "Validation and test sales are excluded from demand validation, "
            "descriptive statistics, grouping, plotting, and conclusions.",
            "No missing-value imputation or source-data cleaning is performed.",
        ],
        "chart_files": [
            (FIGURE_DIR / name).relative_to(ROOT).as_posix()
            for name in FIGURE_NAMES
        ],
    }
    return aggregate, products, weekday_records, summary


def save_figure(figure: Any, path: Path) -> None:
    try:
        figure.savefig(
            path,
            format="png",
            dpi=160,
            bbox_inches="tight",
            metadata={"Software": "Retail demand training-only EDA"},
        )
    finally:
        plt.close(figure)


def create_charts(
    directory: Path,
    aggregate: pd.DataFrame,
    products: pd.DataFrame,
    weekdays: list[dict[str, Any]],
) -> None:
    with plt.rc_context(
        {
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        figure, axis = plt.subplots(figsize=(13, 5))
        axis.plot(
            aggregate["date"],
            aggregate["total_units"].astype(float),
            color="#6b94b5",
            linewidth=0.8,
            alpha=0.7,
            label="Daily aggregate units",
        )
        axis.plot(
            aggregate["date"],
            aggregate["trailing_28_day_average"],
            color="#b34323",
            linewidth=1.8,
            label="Trailing 28-day average",
        )
        axis.set(
            title="Selected products: aggregate daily sales — training only",
            xlabel="Training date",
            ylabel="Units sold per day",
        )
        axis.set_ylim(bottom=0)
        locator = mdates.AutoDateLocator(minticks=5, maxticks=9)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        axis.grid(axis="y", alpha=0.2)
        axis.legend()
        figure.tight_layout()
        save_figure(figure, directory / FIGURE_NAMES[0])

        ordered = products.sort_values("selection_rank")
        labels = ordered["item_id"].tolist()
        # Use full series IDs if item IDs alone would be ambiguous.
        if len(labels) != len(set(labels)):
            labels = ordered["id"].tolist()
        positions = list(range(len(ordered)))
        height = max(6.0, 0.32 * len(ordered) + 1.8)
        figure, axes = plt.subplots(
            1, 2, figsize=(13, height), sharey=True
        )
        axes[0].barh(
            positions, ordered["mean_daily_units"], color="#3478a4"
        )
        axes[1].barh(
            positions, ordered["zero_sales_percentage"], color="#bd793a"
        )
        axes[0].set_yticks(positions, labels=labels)
        axes[0].invert_yaxis()
        axes[0].set(
            title="Mean daily sales",
            xlabel="Units per training day",
            ylabel="Product identifier, selection-rank order",
        )
        axes[1].set(
            title="Zero-sales days",
            xlabel="Percentage of 1,857 training days",
        )
        axes[0].set_xlim(left=0)
        axes[1].set_xlim(0, 100)
        for axis in axes:
            axis.grid(axis="x", alpha=0.2)
            axis.set_axisbelow(True)
        figure.suptitle("Selected-product demand profiles — training only")
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        save_figure(figure, directory / FIGURE_NAMES[1])

        figure, axis = plt.subplots(figsize=(10, 5))
        axis.bar(
            [row["weekday"] for row in weekdays],
            [row["mean_aggregate_daily_units"] for row in weekdays],
            color="#3478a4",
        )
        axis.set(
            title="Mean aggregate daily sales by weekday — training only",
            xlabel="Calendar weekday",
            ylabel="Mean units sold per day across selected products",
        )
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        figure.tight_layout()
        save_figure(figure, directory / FIGURE_NAMES[2])


def verify_png(path: Path) -> None:
    require(path.is_file(), f"Chart was not written: {path}")
    with path.open("rb") as handle:
        require(
            handle.read(len(PNG_SIGNATURE)) == PNG_SIGNATURE,
            f"Invalid PNG signature: {path}",
        )
    # Decode the image as well as checking its signature.
    image = plt.imread(path)
    require(
        image.ndim in (2, 3) and image.shape[0] > 0 and image.shape[1] > 0,
        f"PNG has invalid dimensions: {path}",
    )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish_outputs(
    aggregate: pd.DataFrame,
    products: pd.DataFrame,
    weekdays: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_root = ROOT / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)

    replaced: list[str] = []
    try:
        with tempfile.TemporaryDirectory(
            prefix=".eda-stage-", dir=output_root
        ) as temporary:
            stage = Path(temporary)
            create_charts(stage, aggregate, products, weekdays)

            staged_json = stage / SUMMARY_PATH.name
            staged_json.write_text(
                json.dumps(
                    summary,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                ) + "\n",
                encoding="utf-8",
            )
            for name in FIGURE_NAMES:
                verify_png(stage / name)
            with staged_json.open("r", encoding="utf-8") as handle:
                require(
                    json.load(handle) == summary,
                    "Staged JSON does not reconcile with calculated summary.",
                )

            destinations = [
                (stage / name, FIGURE_DIR / name)
                for name in FIGURE_NAMES
            ] + [(staged_json, SUMMARY_PATH)]
            expected_hashes = {
                destination: digest(source)
                for source, destination in destinations
            }

            FIGURE_DIR.mkdir(parents=True, exist_ok=True)
            SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

            # Individual replacements are atomic on the same filesystem.
            # Replacing all four outputs is not one atomic transaction.
            for source, destination in destinations:
                source.replace(destination)
                replaced.append(destination.name)

            for _, destination in destinations:
                require(
                    destination.is_file()
                    and digest(destination) == expected_hashes[destination],
                    f"Published output verification failed: {destination}",
                )
            for name in FIGURE_NAMES:
                verify_png(FIGURE_DIR / name)
            with SUMMARY_PATH.open("r", encoding="utf-8") as handle:
                require(
                    json.load(handle) == summary,
                    "Published JSON differs from the calculated summary.",
                )
    except Exception as exc:
        raise RuntimeError(
            "EDA output publication failed. Outputs replaced during this "
            f"attempt: {', '.join(replaced) or 'none'}. "
            "Existing files may belong to different runs; resolve the error "
            f"and rerun successfully before using them. Cause: {exc}"
        ) from exc


def main() -> int:
    try:
        train, structure, manifest, metadata = load_inputs()
        ids = validate_inputs(train, structure, manifest, metadata)
        del structure

        aggregate, products, weekdays, summary = calculate_analysis(
            train, manifest, ids
        )
        publish_outputs(aggregate, products, weekdays, summary)
    except Exception as exc:
        print(
            f"Training EDA FAILED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        import traceback
        traceback.print_exc()
        print(
            "Existing EDA outputs must not be treated as successful results "
            "of this run.",
            file=sys.stderr,
        )
        return 1

    print("Training EDA: PASS")
    print(f"Selected series: {summary['selected_series_count']}")
    print(f"Training days per series: {TRAIN_DAYS}")
    print(f"Training observations: {summary['training_observation_count']}")
    print(f"Aggregate training units: {summary['aggregate_total_units']}")
    print(
        "Overall zero-sales percentage: "
        f"{summary['overall_zero_sales_percentage']:.2f}%"
    )
    print("Training integrity and reconciliation checks passed.")
    print("All three PNG charts and JSON summary were written and verified.")
    for name in FIGURE_NAMES:
        print(f"Chart: {FIGURE_DIR / name}")
    print(f"Summary: {SUMMARY_PATH}")
    print("Exit code: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())