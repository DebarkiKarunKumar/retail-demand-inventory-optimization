"""Global direct horizon-aware Ridge forecasting, validation period only.

Run only after dependency and execution approval:
    python -m src.forecast_ridge

The CSV parser transiently tokenizes records. Test sales fields are never
extracted, converted, retained, hashed, summarized, or evaluated.

Historical origins use 258 integer-spaced anchors spanning 29..1829.
Interior anchor i receives offset (i % 7) - 3; endpoints remain fixed.
The resulting deterministic schedule represents all seven weekday offsets.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
import tempfile
import warnings
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from importlib.metadata import version
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)

from src.evaluate import calculate_metrics, evaluate_records


ROOT = Path(__file__).resolve().parents[1]
STORE = "CA_1"
DEPARTMENT = "FOODS_3"
PRODUCT_COUNT = 20
TRAIN_END = 1857
VALIDATION_START = 1858
VALIDATION_END = 1885
HISTORY_END = 1913
HORIZON = 28
ORIGIN_FIRST = 29
ORIGIN_UPPER_BOUND = 1829
ORIGIN_COUNT = 258


def generate_origins() -> tuple[int, ...]:
    """Return the predeclared, sales-independent historical origin schedule.

    Anchor i = 29 + floor(1800 * i / 257), for i=0..257.
    Interior offset = (i % 7) - 3. Endpoints have zero offset.

    Interior origins remain within three days of their evenly spaced integer
    anchors. All arithmetic is integer-only; no random state is involved.
    """
    span = ORIGIN_UPPER_BOUND - ORIGIN_FIRST
    origins = []
    for index in range(ORIGIN_COUNT):
        anchor = ORIGIN_FIRST + span * index // (ORIGIN_COUNT - 1)
        offset = (
            0
            if index in (0, ORIGIN_COUNT - 1)
            else (index % 7) - 3
        )
        origins.append(anchor + offset)
    return tuple(origins)


ORIGINS = generate_origins()
MODEL = "ridge_global"
BASELINE = "seasonal_naive_7"
LEVELS = ("pooled_product_day", "aggregate_daily")
METRICS = ("mae", "rmse", "wape_pct", "bias_units")

CONFIG = {
    "alpha": 1.0,
    "fit_intercept": True,
    "solver": "lsqr",
    "tol": 1e-6,
    "max_iter": 10000,
}

SALES_COLUMNS = (
    "id",
    "item_id",
    "store_id",
    "dept_id",
    "d",
    "day_number",
    "date",
    "sales",
    "split",
)
MANIFEST_COLUMNS = (
    "selection_rank",
    "id",
    "item_id",
    "store_id",
    "dept_id",
    "train_total_units",
    "train_positive_days",
    "train_zero_sales_pct",
)
BASELINE_COLUMNS = (
    "id",
    "item_id",
    "model",
    "forecast_origin_day",
    "horizon",
    "d",
    "date",
    "actual_sales",
    "predicted_sales",
)
PREDICTION_COLUMNS = (
    "id",
    "item_id",
    "model",
    "forecast_origin_day",
    "horizon",
    "d",
    "date",
    "actual_sales",
    "raw_predicted_sales",
    "predicted_sales",
)
PRODUCT_COLUMNS = (
    "id",
    "item_id",
    "model",
    "observations",
    *METRICS,
)
SUMMARY_COLUMNS = (
    "model",
    "evaluation_level",
    "observations",
    *METRICS,
)
COMPARISON_COLUMNS = (
    "evaluation_level",
    "observations",
    "ridge_model",
    "baseline_model",
    *(
        name
        for metric in METRICS
        for name in (
            f"ridge_{metric}",
            f"baseline_{metric}",
            f"difference_{metric}",
        )
    ),
    "relative_wape_improvement_pct",
)
CATEGORICAL = ("id", "horizon", "target_weekday")
NUMERIC = (
    "origin_sales",
    "origin_lag_7",
    "origin_lag_14",
    "origin_lag_28",
    "mean_7",
    "std_7",
    "mean_14",
    "std_14",
    "mean_28",
    "std_28",
    "zero_pct_28",
    "same_weekday_mean_28",
)
FEATURES = CATEGORICAL + NUMERIC

FEATURE_DEFINITIONS = {
    "id": "Selected series ID; fixed one-hot vocabulary.",
    "horizon": "Categorical integer h in 1..28.",
    "target_weekday": "calendar[o+h].weekday(); Monday=0, Sunday=6.",
    "origin_sales": "y[o]",
    "origin_lag_7": "y[o-7]",
    "origin_lag_14": "y[o-14]",
    "origin_lag_28": "y[o-28]",
    "mean_7": "mean(y[o-6:o], inclusive)",
    "std_7": "population std(y[o-6:o], inclusive), ddof=0",
    "mean_14": "mean(y[o-13:o], inclusive)",
    "std_14": "population std(y[o-13:o], inclusive), ddof=0",
    "mean_28": "mean(y[o-27:o], inclusive)",
    "std_28": "population std(y[o-27:o], inclusive), ddof=0",
    "zero_pct_28": "100 * count(y[s]==0, s=o-27..o) / 28",
    "same_weekday_mean_28": (
        "mean(y[s] for s=o-27..o where "
        "calendar[s].weekday()==calendar[o+h].weekday()); exactly four values"
    ),
}

NUMBER_PATTERN = re.compile(
    r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def numeric(value: Any, context: str, text: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{context}: booleans are invalid.")
    if isinstance(value, str):
        require(
            text and NUMBER_PATTERN.fullmatch(value) is not None,
            f"{context}: malformed numeric token.",
        )
    else:
        require(
            isinstance(value, (Real, Decimal)),
            f"{context}: expected a real numeric value.",
        )
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(
            f"{context}: cannot represent numeric value."
        ) from exc
    require(math.isfinite(result), f"{context}: value must be finite.")
    return result


def units(value: Any, context: str, text: bool = False) -> int:
    numeric(value, context, text=text)
    try:
        exact = (
            Decimal(value)
            if isinstance(value, (str, Decimal))
            else Decimal(str(value))
        )
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{context}: invalid unit sales.") from exc
    require(exact >= 0, f"{context}: negative unit sales.")
    require(
        exact == exact.to_integral_value(),
        f"{context}: fractional unit sales.",
    )
    result = int(exact)
    require(
        int(float(result)) == result,
        f"{context}: integer cannot be represented exactly for modeling.",
    )
    return result


def json_constant(token: str) -> None:
    raise ValueError(f"Invalid JSON constant: {token}")


def header(
    reader: Any,
    required: tuple[str, ...],
    source: str,
) -> list[str]:
    names = next(reader, [])
    require(bool(names), f"{source}: missing CSV header.")
    require(
        all(name.strip() for name in names),
        f"{source}: blank header.",
    )
    require(
        len(names) == len(set(names)),
        f"{source}: duplicate headers.",
    )
    missing = [name for name in required if name not in names]
    require(not missing, f"{source}: missing columns {missing}.")
    return names


def read_table(
    path: Path,
    required: tuple[str, ...],
) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        names = header(reader, required, path.name)
        records = []
        for number, row in enumerate(reader, start=1):
            require(
                len(row) == len(names),
                f"{path.name}: malformed record {number}.",
            )
            records.append(dict(zip(names, row)))
    return pd.DataFrame(records, columns=names, dtype=object)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_record(hasher: Any, record: Any) -> None:
    payload = json.dumps(
        record,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    hasher.update((payload + "\n").encode("utf-8"))


def load_selection(
    root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    directory = root / "data/processed"
    manifest = read_table(
        directory / "selected_products.csv",
        MANIFEST_COLUMNS,
    )
    require(
        len(manifest) == PRODUCT_COUNT,
        "Approved run requires exactly 20 selected products.",
    )
    for column in ("id", "item_id", "store_id", "dept_id"):
        require(
            all(
                isinstance(value, str) and value.strip()
                for value in manifest[column]
            ),
            f"Manifest has missing {column}.",
        )
    require(
        not manifest["id"].duplicated().any(),
        "Duplicate manifest IDs.",
    )
    require(
        manifest["store_id"].eq(STORE).all()
        and manifest["dept_id"].eq(DEPARTMENT).all(),
        "Manifest scope must be CA_1 / FOODS_3.",
    )
    for column in (
        "selection_rank",
        "train_total_units",
        "train_positive_days",
    ):
        manifest[column] = pd.Series(
            [
                units(value, f"Manifest {column}", text=True)
                for value in manifest[column]
            ],
            dtype=object,
        )
    require(
        manifest["selection_rank"].tolist()
        == list(range(1, PRODUCT_COUNT + 1)),
        "Manifest must be ordered by ranks 1..20.",
    )
    manifest["train_zero_sales_pct"] = [
        numeric(value, "Manifest zero percentage", text=True)
        for value in manifest["train_zero_sales_pct"]
    ]
    require(
        manifest["train_zero_sales_pct"].between(0, 100).all(),
        "Manifest zero percentages must be within 0..100.",
    )

    with (directory / "selection_metadata.json").open(
        "r",
        encoding="utf-8-sig",
    ) as handle:
        metadata = json.load(handle, parse_constant=json_constant)
    require(
        isinstance(metadata, dict),
        "Selection metadata must be an object.",
    )
    require(
        type(metadata.get("actual_selected_series_count")) is int
        and metadata["actual_selected_series_count"] == PRODUCT_COUNT,
        "Selection metadata count disagrees with approved scope.",
    )
    require(
        metadata.get("selected_series_ids_in_rank_order")
        == manifest["id"].tolist(),
        "Selection metadata IDs disagree with manifest rank order.",
    )
    require(
        metadata.get("store_id") == STORE
        and metadata.get("dept_id") == DEPARTMENT,
        "Selection metadata scope mismatch.",
    )
    return manifest, metadata


def load_sales(
    path: Path,
    manifest: pd.DataFrame,
    period: str,
) -> tuple[
    dict[str, list[int]],
    dict[int, date],
    dict[str, Any],
]:
    """Read one demand period plus relevant calendar structure.

    Routing precedes access to the sales field. Test rows are discarded before
    identifiers, dates, or sales are retained. Validation sales are not accessed
    during the training pass.
    """
    require(
        period in ("train", "validation"),
        "Unsupported demand period.",
    )
    selected = manifest.set_index("id").to_dict("index")
    demand: dict[str, dict[int, int]] = {
        series_id: {} for series_id in selected
    }
    structural_days: dict[str, list[int]] = {
        series_id: [] for series_id in selected
    }
    calendar: dict[int, date] = {}
    seen: set[tuple[str, int]] = set()
    fingerprint = hashlib.sha256()

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        names = header(reader, SALES_COLUMNS, path.name)
        positions = {name: names.index(name) for name in SALES_COLUMNS}

        for number, row in enumerate(reader, start=1):
            require(
                len(row) == len(names),
                f"Sales record {number}: incorrect field count.",
            )
            day_token = row[positions["day_number"]]
            require(
                day_token.isascii() and day_token.isdecimal(),
                f"Sales record {number}: invalid day_number.",
            )
            day = int(day_token)
            require(
                1 <= day <= HISTORY_END,
                f"Sales record {number}: day out of range.",
            )
            expected_split = (
                "train"
                if day <= TRAIN_END
                else "validation"
                if day <= VALIDATION_END
                else "test"
            )
            require(
                row[positions["split"]] == expected_split,
                f"Sales record {number}: incorrect split.",
            )
            if day > VALIDATION_END:
                continue

            series_id = row[positions["id"]]
            require(
                series_id in selected,
                f"Unexpected series ID {series_id!r}.",
            )
            key = (series_id, day)
            require(
                key not in seen,
                f"Duplicate product-day record {key}.",
            )
            seen.add(key)
            previous = structural_days[series_id]
            require(
                not previous or day > previous[-1],
                f"{series_id}: days are not chronologically ordered.",
            )
            previous.append(day)

            for field in ("item_id", "store_id", "dept_id"):
                require(
                    row[positions[field]] == selected[series_id][field],
                    f"{series_id}: {field} differs from manifest.",
                )
            require(
                row[positions["d"]] == f"d_{day}",
                f"{series_id}: incorrect d ID.",
            )
            date_token = row[positions["date"]]
            require(
                re.fullmatch(
                    r"[0-9]{4}-[0-9]{2}-[0-9]{2}",
                    date_token,
                ) is not None,
                f"{series_id}, d_{day}: invalid date format.",
            )
            parsed_date = date.fromisoformat(date_token)
            require(
                day not in calendar or calendar[day] == parsed_date,
                f"d_{day}: inconsistent calendar mapping across products.",
            )
            calendar[day] = parsed_date
            consumed = [
                series_id,
                selected[series_id]["item_id"],
                STORE,
                DEPARTMENT,
                day,
                date_token,
                expected_split,
            ]
            if expected_split == period:
                value = units(
                    row[positions["sales"]],
                    f"{series_id}, d_{day}",
                    text=True,
                )
                demand[series_id][day] = value
                consumed.append(value)
            digest_record(fingerprint, consumed)

    expected_structure = list(range(1, VALIDATION_END + 1))
    for series_id, days in structural_days.items():
        require(
            days == expected_structure,
            f"{series_id}: missing relevant product-day structure.",
        )
    require(
        sorted(calendar) == expected_structure,
        "Calendar does not cover days 1..1885.",
    )
    require(
        all(
            calendar[day] - calendar[day - 1] == timedelta(days=1)
            for day in range(2, VALIDATION_END + 1)
        ),
        "Calendar dates must advance by exactly one day.",
    )

    first, last = (
        (1, TRAIN_END)
        if period == "train"
        else (VALIDATION_START, VALIDATION_END)
    )
    result = {}
    for series_id in sorted(selected):
        require(
            sorted(demand[series_id]) == list(range(first, last + 1)),
            f"{series_id}: incomplete {period} demand coverage.",
        )
        result[series_id] = [
            demand[series_id][day]
            for day in range(first, last + 1)
        ]
    return result, calendar, {
        "sha256": fingerprint.hexdigest(),
        "scope": (
            "Logical consumed records: calendar/identifiers for d_1..d_1885 "
            f"and {period} sales only; test sales excluded."
        ),
        "demand_observations": sum(
            len(values) for values in result.values()
        ),
    }


def history_prefix(history: Any, through: int) -> np.ndarray:
    require(
        isinstance(through, Integral)
        and not isinstance(through, (bool, np.bool_))
        and through >= 29,
        "Feature origin must be an integer of at least 29.",
    )
    require(
        isinstance(history, (list, tuple, np.ndarray)),
        "History must be an ordered one-dimensional sequence.",
    )
    if isinstance(history, np.ndarray):
        require(
            history.ndim == 1,
            "History must be one-dimensional.",
        )
    require(
        len(history) >= through,
        "Missing historical observations.",
    )
    # Slice before validation: never inspect demand after the feature origin.
    values = [
        units(value, f"History day {index + 1}")
        for index, value in enumerate(history[:through])
    ]
    result = np.asarray(values, dtype=np.float64)
    require(
        np.isfinite(result).all(),
        "History cannot be represented finitely.",
    )
    return result


def feature_rows(
    series_id: str,
    history: Any,
    calendar: dict[int, date],
    origin: int,
) -> list[dict[str, Any]]:
    """Construct horizons 1..28 using demand only through origin."""
    require(
        isinstance(series_id, str) and series_id.strip(),
        "Missing product ID.",
    )
    values = history_prefix(history, origin)
    needed = range(origin - 27, origin + HORIZON + 1)
    require(
        all(day in calendar for day in needed),
        "Missing feature calendar dates.",
    )
    require(
        all(type(calendar[day]) is date for day in needed),
        "Feature calendar values must be datetime.date objects.",
    )
    require(
        all(
            calendar[day] - calendar[day - 1] == timedelta(days=1)
            for day in range(origin - 26, origin + HORIZON + 1)
        ),
        "Feature calendar is not consecutive.",
    )

    base = {
        "origin_sales": values[origin - 1],
        "origin_lag_7": values[origin - 8],
        "origin_lag_14": values[origin - 15],
        "origin_lag_28": values[origin - 29],
    }
    for window in (7, 14, 28):
        recent = values[origin - window:origin]
        base[f"mean_{window}"] = float(np.mean(recent))
        base[f"std_{window}"] = float(np.std(recent, ddof=0))
    base["zero_pct_28"] = float(
        100.0
        * np.count_nonzero(values[origin - 28:origin] == 0)
        / 28
    )

    weekday_means = {}
    for weekday in range(7):
        positions = [
            day - 1
            for day in range(origin - 27, origin + 1)
            if calendar[day].weekday() == weekday
        ]
        require(
            len(positions) == 4,
            "Expected four historical observations per weekday.",
        )
        weekday_means[weekday] = float(np.mean(values[positions]))

    rows = []
    for horizon in range(1, HORIZON + 1):
        weekday = calendar[origin + horizon].weekday()
        rows.append(
            {
                "id": series_id,
                "horizon": horizon,
                "target_weekday": weekday,
                **base,
                "same_weekday_mean_28": weekday_means[weekday],
            }
        )
    return rows


def build_examples(
    histories: dict[str, list[int]],
    calendar: dict[int, date],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    require(
        ORIGINS == generate_origins()
        and len(ORIGINS) == ORIGIN_COUNT
        and len(set(ORIGINS)) == ORIGIN_COUNT
        and tuple(sorted(ORIGINS)) == ORIGINS
        and ORIGINS[0] == ORIGIN_FIRST
        and ORIGINS[-1] == ORIGIN_UPPER_BOUND
        and all(
            ORIGIN_FIRST <= origin <= ORIGIN_UPPER_BOUND
            and origin + HORIZON <= TRAIN_END
            for origin in ORIGINS
        )
        and {(origin - 1) % 7 for origin in ORIGINS}
        == set(range(7)),
        "Historical origin configuration is inconsistent.",
    )
    rows, targets, keys = [], [], []
    for series_id in sorted(histories):
        require(
            len(histories[series_id]) == TRAIN_END,
            f"{series_id}: need 1857 training days.",
        )
        checked = history_prefix(histories[series_id], TRAIN_END)
        for origin in ORIGINS:
            rows.extend(
                feature_rows(series_id, checked, calendar, origin)
            )
            for horizon in range(1, HORIZON + 1):
                target_day = origin + horizon
                require(
                    target_day <= TRAIN_END,
                    "Training target exceeds d_1857.",
                )
                targets.append(checked[target_day - 1])
                keys.append(
                    (series_id, origin, horizon, target_day)
                )

    key_frame = pd.DataFrame(
        keys,
        columns=["id", "origin_day", "horizon", "target_day"],
    )
    expected = len(histories) * len(ORIGINS) * HORIZON
    require(
        len(rows) == expected,
        "Incorrect training-example count.",
    )
    require(
        not key_frame.duplicated(
            ["id", "origin_day", "horizon"]
        ).any(),
        "Duplicate historical training-example keys.",
    )
    require(
        key_frame["target_day"].le(TRAIN_END).all(),
        "Training-target boundary violation.",
    )
    return (
        pd.DataFrame(rows, columns=FEATURES),
        np.asarray(targets, dtype=np.float64),
        key_frame,
    )


def validate_features(
    frame: pd.DataFrame,
    ids: tuple[str, ...],
) -> pd.DataFrame:
    require(
        isinstance(frame, pd.DataFrame),
        "Features must be a DataFrame.",
    )
    require(
        tuple(frame.columns) == FEATURES,
        "Feature schema/order mismatch.",
    )
    require(len(frame) > 0, "Feature frame is empty.")
    require(
        all(
            isinstance(value, str) and value in ids
            for value in frame["id"]
        ),
        "Missing or unknown product ID.",
    )
    for column, allowed in (
        ("horizon", set(range(1, 29))),
        ("target_weekday", set(range(7))),
    ):
        require(
            all(
                isinstance(value, Integral)
                and not isinstance(value, (bool, np.bool_))
                and value in allowed
                for value in frame[column]
            ),
            f"Unexpected category in {column}.",
        )
    for column in NUMERIC:
        require(
            pd.api.types.is_numeric_dtype(frame[column])
            and not pd.api.types.is_bool_dtype(frame[column]),
            f"{column}: expected numeric features.",
        )
    numbers = frame[list(NUMERIC)].to_numpy(dtype=np.float64)
    require(
        np.isfinite(numbers).all(),
        "Non-finite numeric features.",
    )
    require(
        (numbers >= 0).all(),
        "Negative demand features.",
    )
    return frame


def make_pipeline(ids: tuple[str, ...]) -> Pipeline:
    require(
        bool(ids)
        and len(ids) == len(set(ids))
        and all(
            isinstance(value, str) and value.strip()
            for value in ids
        ),
        "Invalid fixed product vocabulary.",
    )
    encoder = OneHotEncoder(
        categories=[
            list(ids),
            list(range(1, 29)),
            list(range(7)),
        ],
        handle_unknown="error",
        sparse_output=True,
        dtype=np.float64,
    )
    preprocessing = ColumnTransformer(
        [
            ("categories", encoder, list(CATEGORICAL)),
            ("numeric", StandardScaler(), list(NUMERIC)),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
    return Pipeline(
        [
            (
                "validate",
                FunctionTransformer(
                    validate_features,
                    kw_args={"ids": ids},
                    validate=False,
                ),
            ),
            ("preprocess", preprocessing),
            ("ridge", Ridge(**CONFIG)),
        ]
    )


def solver_iterations(estimator: Any) -> list[int] | None:
    """Return optional solver diagnostics without requiring their presence."""
    iterations = getattr(estimator, "n_iter_", None)
    if iterations is None:
        return None
    return (
        np.asarray(iterations)
        .astype(int)
        .reshape(-1)
        .tolist()
    )


def fit_model(
    features: pd.DataFrame,
    targets: np.ndarray,
    ids: tuple[str, ...],
) -> Pipeline:
    require(
        targets.ndim == 1 and len(targets) == len(features),
        "Invalid training target shape.",
    )
    require(
        np.isfinite(targets).all()
        and (targets >= 0).all()
        and np.equal(targets, np.floor(targets)).all(),
        "Training targets must be finite nonnegative integers.",
    )
    model = make_pipeline(ids)
    # A convergence warning is promoted to failure, never suppressed.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(features, targets)

    require(
        model.named_steps["preprocess"].sparse_output_,
        "Expected sparse preprocessing output.",
    )
    estimator = model.named_steps["ridge"]
    require(
        np.isfinite(
            np.asarray(estimator.coef_, dtype=float)
        ).all()
        and np.isfinite(
            np.asarray(estimator.intercept_, dtype=float)
        ).all(),
        "Ridge fitted coefficients or intercept are non-finite.",
    )

    # Validate fitted predictions independently of optional iteration metadata.
    fitted_predictions = np.asarray(
        model.predict(features),
        dtype=float,
    )
    require(
        fitted_predictions.shape == targets.shape
        and np.isfinite(fitted_predictions).all(),
        "Ridge fitted predictions have invalid shape or non-finite values.",
    )
    return model


def clip_predictions(
    raw: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(raw, dtype=np.float64)
    require(
        values.ndim == 1 and len(values) > 0,
        "Invalid prediction shape.",
    )
    require(
        np.isfinite(values).all(),
        "Raw predictions must be finite.",
    )
    clipped = np.maximum(values, 0.0)
    count = int(np.count_nonzero(values < 0))
    return clipped, {
        "negative_raw_prediction_count": count,
        "percentage_clipped": 100.0 * count / len(values),
        "minimum_raw_prediction": float(values.min()),
        "total_upward_adjustment": float(
            np.sum(clipped - values)
        ),
    }


def forecast_validation(
    model: Pipeline,
    histories: dict[str, list[int]],
    calendar: dict[int, date],
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    items = manifest.set_index("id")["item_id"].to_dict()
    require(
        set(histories) == set(items),
        "Forecast product IDs disagree with manifest.",
    )
    rows, keys = [], []
    for series_id in sorted(histories):
        require(
            len(histories[series_id]) == TRAIN_END,
            "Incorrect validation-origin history.",
        )
        rows.extend(
            feature_rows(
                series_id,
                histories[series_id],
                calendar,
                TRAIN_END,
            )
        )
        for horizon in range(1, HORIZON + 1):
            day = TRAIN_END + horizon
            keys.append(
                {
                    "id": series_id,
                    "item_id": items[series_id],
                    "model": MODEL,
                    "forecast_origin_day": "d_1857",
                    "horizon": horizon,
                    "d": f"d_{day}",
                    "date": calendar[day].isoformat(),
                }
            )
    raw = np.asarray(
        model.predict(pd.DataFrame(rows, columns=FEATURES)),
        dtype=float,
    )
    clipped, diagnostics = clip_predictions(raw)
    frame = pd.DataFrame(keys)
    frame["raw_predicted_sales"] = raw
    frame["predicted_sales"] = clipped
    return (
        frame.sort_values(["model", "id", "horizon"])
        .reset_index(drop=True),
        diagnostics,
    )


def attach_actuals(
    forecasts: pd.DataFrame,
    actuals: dict[str, list[int]],
) -> pd.DataFrame:
    require(
        set(forecasts["id"]) == set(actuals),
        "Validation actual ID mismatch.",
    )
    require(
        all(len(values) == HORIZON for values in actuals.values()),
        "Each product requires 28 validation actuals.",
    )
    frame = forecasts.copy()
    frame["actual_sales"] = [
        units(
            actuals[row.id][row.horizon - 1],
            "Validation actual",
        )
        for row in frame.itertuples(index=False)
    ]
    return frame[list(PREDICTION_COLUMNS)]


def metric_tables(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    products, summaries = [], []
    for model_name, model_rows in predictions.groupby(
        "model",
        sort=True,
    ):
        for series_id, group in model_rows.groupby("id", sort=True):
            metrics = calculate_metrics(
                group["actual_sales"].tolist(),
                group["predicted_sales"].tolist(),
            )
            products.append(
                {
                    "id": series_id,
                    "item_id": group["item_id"].iloc[0],
                    "model": model_name,
                    **metrics,
                }
            )
        records = [
            {
                "id": row.id,
                "day_number": TRAIN_END + row.horizon,
                "actual": row.actual_sales,
                "predicted": row.predicted_sales,
            }
            for row in model_rows.itertuples(index=False)
        ]
        for level in LEVELS:
            result = evaluate_records(records, level)
            summaries.append(
                {
                    "model": model_name,
                    "evaluation_level": level,
                    "observations": result["observations"],
                    **{
                        name: result[name]
                        for name in METRICS
                    },
                }
            )
    return (
        pd.DataFrame(
            products,
            columns=PRODUCT_COLUMNS,
            dtype=object,
        )
        .sort_values(["model", "id"])
        .reset_index(drop=True),
        pd.DataFrame(
            summaries,
            columns=SUMMARY_COLUMNS,
            dtype=object,
        )
        .sort_values(["model", "evaluation_level"])
        .reset_index(drop=True),
    )


def load_and_align_baseline(
    root: Path,
    ridge_predictions: pd.DataFrame,
) -> pd.DataFrame:
    path = root / "outputs/evaluation/baseline_predictions.csv"
    raw = read_table(path, BASELINE_COLUMNS)
    selected = raw.loc[raw["model"].eq(BASELINE)].copy()
    require(
        len(selected) == len(ridge_predictions),
        "Baseline prediction count mismatch.",
    )
    selected["horizon"] = [
        units(value, "Baseline horizon", text=True)
        for value in selected["horizon"]
    ]
    require(
        selected["horizon"].between(1, HORIZON).all(),
        "Baseline contains non-validation horizons.",
    )
    require(
        not selected.duplicated(["id", "horizon"]).any(),
        "Duplicate baseline prediction keys.",
    )
    selected = (
        selected.sort_values(["id", "horizon"])
        .reset_index(drop=True)
    )
    ridge = (
        ridge_predictions.sort_values(["id", "horizon"])
        .reset_index(drop=True)
    )
    for column in (
        "id",
        "item_id",
        "horizon",
        "forecast_origin_day",
        "d",
        "date",
    ):
        require(
            selected[column].tolist() == ridge[column].tolist(),
            f"Baseline alignment mismatch in {column}.",
        )

    # Access demand only after validation keys and dates have matched.
    for column in ("actual_sales", "predicted_sales"):
        selected[column] = pd.Series(
            [
                units(
                    value,
                    f"Baseline {column}",
                    text=True,
                )
                for value in selected[column]
            ],
            dtype=object,
        )
    require(
        selected["actual_sales"].tolist()
        == ridge["actual_sales"].tolist(),
        "Baseline validation actuals differ from current validation actuals.",
    )

    summary_path = (
        root / "outputs/evaluation/baseline_metrics_summary.csv"
    )
    stored = read_table(summary_path, SUMMARY_COLUMNS)
    stored = stored.loc[stored["model"].eq(BASELINE)].copy()
    require(
        len(stored) == 2
        and set(stored["evaluation_level"]) == set(LEVELS)
        and not stored["evaluation_level"].duplicated().any(),
        "Stored baseline summary coverage mismatch.",
    )
    parsed = []
    for record in stored.to_dict("records"):
        record["observations"] = units(
            record["observations"],
            "Baseline observations",
            text=True,
        )
        for metric in METRICS:
            record[metric] = (
                None
                if metric == "wape_pct" and record[metric] == ""
                else numeric(
                    record[metric],
                    f"Baseline {metric}",
                    text=True,
                )
            )
        parsed.append(record)
    stored = pd.DataFrame(
        parsed,
        columns=SUMMARY_COLUMNS,
        dtype=object,
    )
    _, recalculated = metric_tables(selected)

    for level in LEVELS:
        saved = stored.loc[
            stored["evaluation_level"].eq(level)
        ].iloc[0]
        checked = recalculated.loc[
            recalculated["evaluation_level"].eq(level)
        ].iloc[0]
        require(
            saved["observations"] == checked["observations"],
            "Baseline observation mismatch.",
        )
        for metric in METRICS:
            a, b = saved[metric], checked[metric]
            require(
                (a is None and b is None)
                or (
                    a is not None
                    and b is not None
                    and math.isclose(
                        a,
                        b,
                        rel_tol=1e-10,
                        abs_tol=1e-10,
                    )
                ),
                f"Stored baseline {level} {metric} "
                "disagrees with its predictions.",
            )
    # Use stored, reconciled metrics rather than substituting new values.
    return (
        stored.sort_values("evaluation_level")
        .reset_index(drop=True)
    )


def compare_models(
    ridge_summary: pd.DataFrame,
    baseline_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    rows = []
    for level in LEVELS:
        ridge = ridge_summary.loc[
            ridge_summary["evaluation_level"].eq(level)
        ].iloc[0]
        baseline = baseline_summary.loc[
            baseline_summary["evaluation_level"].eq(level)
        ].iloc[0]
        require(
            ridge["observations"] == baseline["observations"],
            "Comparison observation counts differ.",
        )
        row = {
            "evaluation_level": level,
            "observations": ridge["observations"],
            "ridge_model": MODEL,
            "baseline_model": BASELINE,
        }
        for metric in METRICS:
            r, b = ridge[metric], baseline[metric]
            row[f"ridge_{metric}"] = r
            row[f"baseline_{metric}"] = b
            row[f"difference_{metric}"] = (
                None if r is None or b is None else r - b
            )
        r, b = ridge["wape_pct"], baseline["wape_pct"]
        row["relative_wape_improvement_pct"] = (
            None
            if r is None or b is None or b == 0
            else 100.0 * (b - r) / b
        )
        rows.append(row)

    comparison = pd.DataFrame(
        rows,
        columns=COMPARISON_COLUMNS,
        dtype=object,
    )
    primary = comparison.loc[
        comparison["evaluation_level"].eq("pooled_product_day")
    ].iloc[0]
    r, b = primary["ridge_wape_pct"], primary["baseline_wape_pct"]
    selected = (
        MODEL
        if r is not None and b is not None and r < b
        else BASELINE
    )
    return (
        comparison.sort_values("evaluation_level")
        .reset_index(drop=True),
        selected,
    )


def verify_outputs(
    predictions: pd.DataFrame,
    products: pd.DataFrame,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    actuals: dict[str, list[int]],
    calendar: dict[int, date],
    manifest: pd.DataFrame,
) -> dict[str, Any]:
    require(
        tuple(predictions.columns) == PREDICTION_COLUMNS,
        "Prediction schema mismatch.",
    )
    require(
        tuple(products.columns) == PRODUCT_COLUMNS,
        "Product metric schema mismatch.",
    )
    require(
        tuple(summary.columns) == SUMMARY_COLUMNS,
        "Summary schema mismatch.",
    )
    require(
        tuple(comparison.columns) == COMPARISON_COLUMNS,
        "Comparison schema mismatch.",
    )
    ids = set(manifest["id"])
    require(
        len(predictions) == PRODUCT_COUNT * HORIZON,
        "Expected 560 predictions.",
    )
    require(
        set(predictions["id"]) == ids,
        "Prediction product coverage mismatch.",
    )
    require(
        predictions["model"].eq(MODEL).all()
        and predictions["forecast_origin_day"].eq("d_1857").all(),
        "Prediction model/origin mismatch.",
    )
    require(
        not predictions.duplicated(
            ["id", "model", "horizon"]
        ).any(),
        "Duplicate prediction keys.",
    )
    require(
        not predictions.isna().any().any(),
        "Missing prediction fields.",
    )
    for series_id, group in predictions.groupby("id", sort=True):
        require(
            group["horizon"].tolist() == list(range(1, 29)),
            f"{series_id}: incorrect horizon coverage.",
        )
        for row in group.itertuples(index=False):
            require(
                isinstance(row.horizon, Integral)
                and not isinstance(row.horizon, (bool, np.bool_)),
                "Forecast horizons must be integers.",
            )
            day = TRAIN_END + row.horizon
            require(
                row.d == f"d_{day}"
                and row.date == calendar[day].isoformat(),
                "Output target calendar mismatch.",
            )
            require(
                row.actual_sales
                == actuals[series_id][row.horizon - 1],
                "Output validation actual mismatch.",
            )
            require(
                math.isfinite(row.raw_predicted_sales)
                and row.predicted_sales
                == max(0.0, row.raw_predicted_sales),
                "Output clipping mismatch.",
            )

    require(
        len(products) == PRODUCT_COUNT
        and set(products["id"]) == ids
        and not products["id"].duplicated().any()
        and products["model"].eq(MODEL).all()
        and products["observations"].eq(HORIZON).all(),
        "Per-product metric coverage/count mismatch.",
    )
    require(
        len(summary) == 2
        and set(summary["evaluation_level"]) == set(LEVELS)
        and not summary["evaluation_level"].duplicated().any()
        and summary["model"].eq(MODEL).all(),
        "Summary metric coverage mismatch.",
    )
    for record in summary.to_dict("records"):
        expected = (
            PRODUCT_COUNT * HORIZON
            if record["evaluation_level"] == "pooled_product_day"
            else HORIZON
        )
        require(
            record["observations"] == expected,
            "Summary observation count mismatch.",
        )
    require(
        len(comparison) == 2
        and set(comparison["evaluation_level"]) == set(LEVELS),
        "Comparison coverage mismatch.",
    )
    for table in (products, summary):
        for record in table.to_dict("records"):
            for metric in METRICS:
                value = record[metric]
                require(
                    (metric == "wape_pct" and value is None)
                    or (
                        isinstance(value, Real)
                        and math.isfinite(value)
                        and (metric == "bias_units" or value >= 0)
                    ),
                    f"Invalid metric output: {metric}.",
                )
    return {
        "status": "PASS",
        "prediction_rows": len(predictions),
        "product_metric_rows": len(products),
        "summary_metric_rows": len(summary),
        "comparison_rows": len(comparison),
        "duplicate_prediction_keys": 0,
        "historical_feature_boundaries_verified": True,
        "fixed_origin_and_horizons_verified": True,
        "validation_actuals_reconciled": True,
        "baseline_keys_dates_actuals_and_metrics_reconciled": True,
        "metric_implementation": "src.evaluate",
        "test_sales_loaded_or_evaluated": False,
    }


def publish_outputs(
    root: Path,
    predictions: pd.DataFrame,
    products: pd.DataFrame,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    metadata: dict[str, Any],
) -> list[Path]:
    directory = root / "outputs/evaluation"
    directory.mkdir(parents=True, exist_ok=True)
    names = (
        "ridge_predictions.csv",
        "ridge_metrics_by_product.csv",
        "ridge_metrics_summary.csv",
        "ridge_baseline_comparison.csv",
        "ridge_run_metadata.json",
    )
    destinations = [directory / name for name in names]
    replaced = []
    try:
        with tempfile.TemporaryDirectory(
            prefix=".ridge-stage-",
            dir=directory,
        ) as temp:
            staged = [Path(temp) / name for name in names]
            for frame, path in zip(
                (predictions, products, summary, comparison),
                staged[:4],
            ):
                frame.to_csv(
                    path,
                    index=False,
                    na_rep="",
                    lineterminator="\n",
                )
                reread = read_table(path, tuple(frame.columns))
                require(
                    tuple(reread.columns) == tuple(frame.columns)
                    and len(reread) == len(frame),
                    f"Staged CSV verification failed: {path.name}",
                )
                expected_cells = [
                    [
                        "" if value is None else str(value)
                        for value in row
                    ]
                    for row in frame.itertuples(
                        index=False,
                        name=None,
                    )
                ]
                require(
                    reread.values.tolist() == expected_cells,
                    f"Staged CSV values differ: {path.name}",
                )
            staged[4].write_text(
                json.dumps(
                    metadata,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                ) + "\n",
                encoding="utf-8",
            )
            with staged[4].open("r", encoding="utf-8") as handle:
                require(
                    json.load(handle) == metadata,
                    "Staged metadata differs.",
                )

            fingerprints = []
            for path in staged:
                require(
                    path.stat().st_size > 0,
                    f"Empty staged output: {path.name}",
                )
                fingerprints.append(sha256(path))

            # Each replacement is atomic individually, not as a five-file set.
            for source, destination in zip(staged, destinations):
                source.replace(destination)
                replaced.append(destination.name)
            for path, expected in zip(destinations, fingerprints):
                require(
                    path.is_file() and sha256(path) == expected,
                    f"Published output verification failed: {path.name}",
                )
    except Exception as exc:
        raise RuntimeError(
            "Ridge publication failed. Files replaced this attempt: "
            f"{', '.join(replaced) or 'none'}. Outputs may be from different "
            f"runs; rerun successfully before using them. Cause: {exc}"
        ) from exc
    return destinations


def run_pipeline(root: Path = ROOT) -> dict[str, Any]:
    root = Path(root)
    paths = {
        "sales": root / "data/processed/selected_sales.csv",
        "manifest": root / "data/processed/selected_products.csv",
        "selection_metadata": (
            root / "data/processed/selection_metadata.json"
        ),
        "baseline_predictions": (
            root / "outputs/evaluation/baseline_predictions.csv"
        ),
        "baseline_metrics_summary": (
            root / "outputs/evaluation/baseline_metrics_summary.csv"
        ),
    }
    for path in paths.values():
        require(path.is_file(), f"Required input missing: {path}")

    # Hash the prepared sales artifact only through scoped logical records.
    # A full sales-file hash would also read excluded test-demand bytes.
    fingerprints = {
        name: {
            "filename": path.relative_to(root).as_posix(),
            "sha256": sha256(path),
            "scope": "Complete artifact",
        }
        for name, path in paths.items()
        if name != "sales"
    }
    manifest, _ = load_selection(root)
    histories, calendar, train_fingerprint = load_sales(
        paths["sales"],
        manifest,
        "train",
    )
    indexed_manifest = manifest.set_index("id")
    for series_id, values in histories.items():
        require(
            sum(values)
            == indexed_manifest.at[series_id, "train_total_units"]
            and sum(value > 0 for value in values)
            == indexed_manifest.at[series_id, "train_positive_days"]
            and math.isclose(
                100.0
                * sum(value == 0 for value in values)
                / TRAIN_END,
                indexed_manifest.at[
                    series_id,
                    "train_zero_sales_pct",
                ],
                rel_tol=0,
                abs_tol=1e-9,
            ),
            f"{series_id}: training demand disagrees with manifest.",
        )
        require(
            sum(value > 0 for value in values) >= 28,
            f"{series_id}: selected series fails training activity threshold.",
        )

    features, targets, keys = build_examples(histories, calendar)
    require(
        len(features) == 144480,
        "Approved run requires 144480 training examples.",
    )
    ids = tuple(sorted(histories))
    model = fit_model(features, targets, ids)
    forecasts, clipping = forecast_validation(
        model,
        histories,
        calendar,
        manifest,
    )

    # First access to validation actuals occurs after forecasts are frozen.
    actuals, second_calendar, validation_fingerprint = load_sales(
        paths["sales"],
        manifest,
        "validation",
    )
    require(
        calendar == second_calendar,
        "Calendar changed between filtered passes.",
    )
    predictions = attach_actuals(forecasts, actuals)
    products, summary = metric_tables(predictions)
    baseline_summary = load_and_align_baseline(root, predictions)
    comparison, selected_model = compare_models(
        summary,
        baseline_summary,
    )
    integrity = verify_outputs(
        predictions,
        products,
        summary,
        comparison,
        actuals,
        calendar,
        manifest,
    )
    for name, path in paths.items():
        if name != "sales":
            require(
                sha256(path) == fingerprints[name]["sha256"],
                f"Input artifact changed during execution: {path.name}",
            )
    fingerprints["sales"] = {
        "filename": paths["sales"].relative_to(root).as_posix(),
        "training_pass": train_fingerprint,
        "validation_pass": validation_fingerprint,
        "full_file_hashed": False,
        "reason": "Test-demand exclusion includes fingerprinting.",
    }

    preprocessing = model.named_steps["preprocess"]
    scaler = preprocessing.named_transformers_["numeric"]
    estimator = model.named_steps["ridge"]
    metadata = {
        "dataset": "Walmart M5 Forecasting Accuracy",
        "model": MODEL,
        "model_configuration": CONFIG,
        "scope": {
            "store_id": STORE,
            "dept_id": DEPARTMENT,
        },
        "selected_series_count": len(ids),
        "selected_series_ids": list(ids),
        "feature_definitions": FEATURE_DEFINITIONS,
        "feature_order": list(FEATURES),
        "training_example_count": len(features),
        "historical_origins": {
            "first": ORIGINS[0],
            "last_actual": ORIGINS[-1],
            "inclusive_upper_bound": ORIGIN_UPPER_BOUND,
            "step_days": None,
            "count_per_product": len(ORIGINS),
            "schedule": (
                "For i=0..257: anchor=29+(1800*i)//257; "
                "origin=anchor+(i%7)-3 for interior indices; "
                "first and last origins remain 29 and 1829."
            ),
            "origin_weekday_offsets": sorted(
                {(origin - 1) % 7 for origin in ORIGINS}
            ),
            "actual_calendar_origin_weekdays": sorted(
                {
                    calendar[origin].weekday()
                    for origin in ORIGINS
                }
            ),
            "arithmetic_note": (
                "258 distinct approximately evenly distributed origins; "
                "integer arithmetic only; variable spacing; "
                "all targets remain at or before d_1857."
            ),
        },
        "training_targets": {
            "first_day": int(keys["target_day"].min()),
            "last_day": int(keys["target_day"].max()),
            "maximum_permitted_day": TRAIN_END,
        },
        "validation_origin": {
            "day": "d_1857",
            "date": calendar[TRAIN_END].isoformat(),
        },
        "validation_period": {
            "first_day": "d_1858",
            "last_day": "d_1885",
            "first_date": calendar[VALIDATION_START].isoformat(),
            "last_date": calendar[VALIDATION_END].isoformat(),
        },
        "forecast_horizon": HORIZON,
        "package_versions": {
            name: version(name)
            for name in (
                "pandas",
                "numpy",
                "scikit-learn",
                "scipy",
            )
        },
        "python_version": sys.version,
        "preprocessing": {
            "categorical_features": list(CATEGORICAL),
            "fixed_categories": {
                "id": list(ids),
                "horizon": list(range(1, 29)),
                "target_weekday": list(range(7)),
            },
            "unknown_categories": "Rejected",
            "numeric_features": list(NUMERIC),
            "numeric_scaler": (
                "StandardScaler; fit on historical examples only"
            ),
            "numeric_fitted_means": {
                name: float(value)
                for name, value in zip(NUMERIC, scaler.mean_)
            },
            "numeric_fitted_scales": {
                name: float(value)
                for name, value in zip(NUMERIC, scaler.scale_)
            },
            "sparse_output": bool(preprocessing.sparse_output_),
            "transformed_feature_count": len(
                preprocessing.get_feature_names_out()
            ),
            "target_transformed": False,
            "sample_weights": None,
            "solver_iterations": solver_iterations(estimator),
        },
        "clipping": clipping,
        "input_fingerprints": fingerprints,
        "integrity_checks": integrity,
        "selected_model": selected_model,
        "selection_rule": (
            "Choose ridge_global only when its pooled product-day WAPE is "
            "defined and strictly below the defined seasonal_naive_7 WAPE; "
            "otherwise retain seasonal_naive_7."
        ),
        "comparison_difference_convention": (
            "Ridge minus seasonal naive"
        ),
        "relative_wape_improvement": (
            "100 * (baseline - ridge) / baseline; null if undefined"
        ),
        "no_test_demand_access": (
            "Test sales fields were not extracted, converted, retained, "
            "hashed, summarized, used as features, or evaluated. "
            "CSV parsing transiently tokenizes records before routing."
        ),
        "limitations": [
            "High-volume selected products are not representative "
            "of the department.",
            "Recorded sales need not equal unconstrained customer demand.",
            "Global Ridge has shared additive effects and "
            "squared-error fitting.",
            "Historical origins are approximately evenly distributed with "
            "deterministic variable spacing; all seven origin weekdays occur.",
            "Weekday coverage removes the fixed-weekly-origin confounding, "
            "but global Ridge still imposes shared additive feature effects.",
            "Overlapping historical examples are dependent.",
            "No prices, promotions, inventory availability, "
            "or lost-sales information.",
            "Negative raw predictions are preserved; only fixed "
            "zero-clipped values are scored.",
            "No hyperparameter tuning or final-test evaluation.",
            "Sales fingerprints cover consumed logical records, "
            "not the full CSV.",
            "Prepared inputs are assumed stable during execution.",
            "Individual output replacement is atomic; "
            "the five-file set is not.",
        ],
    }
    outputs = publish_outputs(
        root,
        predictions,
        products,
        summary,
        comparison,
        metadata,
    )
    return {
        "metadata": metadata,
        "output_paths": outputs,
    }


def main() -> int:
    try:
        result = run_pipeline()
    except Exception as exc:
        print(
            f"Ridge pipeline FAILED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print(
            "Existing Ridge outputs are not evidence "
            "of a successful current run.",
            file=sys.stderr,
        )
        return 1

    metadata = result["metadata"]
    print("Ridge validation pipeline: PASS")
    print(
        f"Training examples: {metadata['training_example_count']}"
    )
    print(
        f"Predictions: "
        f"{metadata['integrity_checks']['prediction_rows']}"
    )
    print(f"Selected model: {metadata['selected_model']}")
    print(
        "Clipped predictions: "
        f"{metadata['clipping']['negative_raw_prediction_count']}"
    )
    print("Test sales were not retained or evaluated.")
    for path in result["output_paths"]:
        print(f"Output: {path}")
    print("Exit code: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())