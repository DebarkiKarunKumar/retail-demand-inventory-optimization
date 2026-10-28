"""Global direct XGBoost forecasting with chronological historical backtests.

Run only after review, dependency approval, and successful synthetic tests:
    python -m src.forecast_xgboost

Feature definitions come from src.prepare_xgb_features.

Demand is loaded separately for each permitted interval. CSV parsing
transiently tokenizes records, but excluded sales fields are never extracted,
converted, retained, or evaluated.

Existing feature, Ridge, and seasonal-naive artifacts remain unchanged.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from importlib.metadata import version
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from xgboost import XGBRegressor

import src.prepare_xgb_features as feature_module
from src.evaluate import calculate_metrics, evaluate_records


ROOT = Path(__file__).resolve().parents[1]
STORE = "CA_1"
DEPARTMENT = "FOODS_3"
PRODUCT_COUNT = 20
TRAIN_END = 1857
VALIDATION_END = 1885
HISTORY_END = 1913
HORIZON = 28
BACKTEST_ORIGINS = (1689, 1745, 1801)

MODEL = "xgboost"
SEASONAL_MODEL = "seasonal_naive_7"
RIDGE_MODEL = "ridge_global"
CONFIGURATION_ORDER = ("A_squared_error", "B_tweedie")
LEVELS = ("pooled_product_day", "aggregate_daily")
METRICS = ("mae", "rmse", "wape_pct", "bias_units")

COMMON_PARAMETERS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_lambda": 5,
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": 4,
}
CONFIGURATIONS = {
    "A_squared_error": {
        **COMMON_PARAMETERS,
        "objective": "reg:squarederror",
    },
    "B_tweedie": {
        **COMMON_PARAMETERS,
        "objective": "reg:tweedie",
        "tweedie_variance_power": 1.3,
    },
}

CATEGORICAL_PREDICTORS = (
    "id",
    "target_weekday",
    "target_month",
    "target_event_type_1",
    "target_event_type_2",
)
NUMERIC_PREDICTORS = (
    *feature_module.DEMAND_FEATURES,
    "horizon",
    "target_snap_CA",
    "target_has_event",
    *feature_module.PRICE_FEATURES,
)
PREDICTOR_COLUMNS = CATEGORICAL_PREDICTORS + NUMERIC_PREDICTORS
MISSING_ALLOWED_COLUMNS = frozenset(feature_module.PRICE_FEATURES[:5])

BASELINE_PREDICTION_COLUMNS = (
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
BACKTEST_METRIC_COLUMNS = (
    "configuration",
    "objective",
    "scope",
    "forecast_origin_day",
    "fitting_cutoff",
    "evaluation_level",
    "observations",
    *METRICS,
)
COMPARISON_COLUMNS = (
    *SUMMARY_COLUMNS,
    "difference_wape_pct_vs_seasonal",
    "relative_wape_improvement_pct_vs_seasonal",
    "difference_wape_pct_vs_ridge",
    "relative_wape_improvement_pct_vs_ridge",
)

CSV_KEY_COLUMNS = {
    "xgb_backtest_metrics.csv": (
        "configuration",
        "scope",
        "forecast_origin_day",
        "evaluation_level",
    ),
    "xgb_predictions.csv": ("id", "model", "horizon"),
    "xgb_metrics_by_product.csv": ("id", "model"),
    "xgb_metrics_summary.csv": ("model", "evaluation_level"),
    "xgb_model_comparison.csv": ("model", "evaluation_level"),
}
CSV_STRING_COLUMNS = frozenset({
    "id",
    "item_id",
    "model",
    "configuration",
    "objective",
    "scope",
    "forecast_origin_day",
    "evaluation_level",
    "d",
    "date",
})
CSV_INTEGER_COLUMNS = frozenset({
    "horizon",
    "actual_sales",
    "observations",
    "fitting_cutoff",
})
CSV_FLOAT_COLUMNS = frozenset({
    *METRICS,
    "raw_predicted_sales",
    "predicted_sales",
    "difference_wape_pct_vs_seasonal",
    "relative_wape_improvement_pct_vs_seasonal",
    "difference_wape_pct_vs_ridge",
    "relative_wape_improvement_pct_vs_ridge",
})
CSV_NULLABLE_COLUMNS = frozenset({
    "fitting_cutoff",
    "wape_pct",
    "difference_wape_pct_vs_seasonal",
    "relative_wape_improvement_pct_vs_seasonal",
    "difference_wape_pct_vs_ridge",
    "relative_wape_improvement_pct_vs_ridge",
})

DemandReader = Callable[[int, int], dict[str, list[int]]]
EstimatorFactory = Callable[[str], Any]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_feature_interface() -> None:
    required_functions = (
        "load_selection",
        "load_calendar",
        "training_weeks",
        "load_prices",
        "build_feature_rows",
        "build_training_examples",
        "integer_value",
        "csv_header",
        "reject_json_constant",
    )
    for name in required_functions:
        require(
            callable(getattr(feature_module, name, None)),
            f"Feature-module interface mismatch: missing callable {name}.",
        )
    require(
        feature_module.TRAIN_END == TRAIN_END
        and feature_module.VALIDATION_END == VALIDATION_END
        and feature_module.HORIZON == HORIZON
        and feature_module.MIN_ORIGIN == 57,
        "Feature-module chronological constants differ from this stage.",
    )
    require(
        set(PREDICTOR_COLUMNS).issubset(feature_module.FEATURE_COLUMNS)
        and len(PREDICTOR_COLUMNS) == len(set(PREDICTOR_COLUMNS))
        and not {"origin_day", "target_day", "target_sales"}.intersection(
            PREDICTOR_COLUMNS
        ),
        "Feature-module predictor interface mismatch.",
    )


def generate_origins(
    cutoff: int,
    *,
    max_cutoff: int = TRAIN_END,
) -> tuple[int, ...]:
    """Return deterministic, approximately weekly, weekday-diverse origins.

    Eligible interval: [57, cutoff-28].
    Anchor count: floor((upper-57)/7)+1.

    Integer-spaced anchors span the eligible interval. Interior anchor i
    receives offset (i % 7)-3; endpoints remain fixed.

    When there are at least two anchors, their gaps are at least seven days.
    Offset differences can reduce actual origin gaps below seven days.
    They cannot reduce a gap to zero: an interior offset decreases by at
    most six days, and endpoint offset changes have magnitude at most three.

    Actual origins are strictly increasing. No minimum seven-day origin
    gap is claimed. No sales or performance values select origins.

    The default maximum fitting cutoff remains d_1857.
    """
    require(
        isinstance(max_cutoff, Integral)
        and not isinstance(max_cutoff, (bool, np.bool_))
        and feature_module.MIN_ORIGIN + HORIZON
        <= max_cutoff <= VALIDATION_END,
        "Maximum fitting cutoff must permit complete labels "
        "and cannot exceed 1885.",
    )
    require(
        isinstance(cutoff, Integral)
        and not isinstance(cutoff, (bool, np.bool_))
        and feature_module.MIN_ORIGIN + HORIZON
        <= cutoff <= max_cutoff,
        (
            "Fitting cutoff must permit a 57-day origin "
            "and 28 training labels."
            if max_cutoff == TRAIN_END
            else "Fitting cutoff exceeds the explicitly permitted boundary "
            "or cannot provide complete historical labels."
        ),
    )
    first = feature_module.MIN_ORIGIN
    upper = int(cutoff) - HORIZON
    span = upper - first
    count = span // 7 + 1

    if count == 1:
        origins = (first,)
    else:
        origins = tuple(
            first
            + span * index // (count - 1)
            + (
                0
                if index in (0, count - 1)
                else index % 7 - 3
            )
            for index in range(count)
        )

    require(
        len(origins) == count
        and len(set(origins)) == count
        and all(
            later > earlier
            for earlier, later in zip(origins, origins[1:])
        ),
        "Historical origins must be unique and strictly increasing.",
    )
    require(
        all(
            first <= origin <= upper
            and origin - 56 >= 1
            and origin + HORIZON <= cutoff
            for origin in origins
        ),
        "Historical origin violates lookback or fitting-label boundaries.",
    )
    require(
        origins[0] == first,
        "Incorrect first historical origin.",
    )
    if count > 1:
        require(
            origins[-1] == upper,
            "Incorrect final historical origin.",
        )

    if cutoff in (*BACKTEST_ORIGINS, TRAIN_END, VALIDATION_END):
        require(
            {(origin - 1) % 7 for origin in origins}
            == set(range(7)),
            "Approved fitting cutoff lacks all seven origin weekday offsets.",
        )
    return origins


def read_table(path: Path, required: tuple[str, ...]) -> pd.DataFrame:
    records = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        names = feature_module.csv_header(reader, required, path.name)
        for number, row in enumerate(reader, start=1):
            require(
                len(row) == len(names),
                f"{path.name}: incorrect field count at record {number}.",
            )
            records.append(dict(zip(names, row)))
    return pd.DataFrame(records, columns=names, dtype=object)


def finite_number(raw: Any, context: str, text: bool = False) -> float:
    if isinstance(raw, (bool, np.bool_)):
        raise ValueError(f"{context}: booleans are invalid.")
    if isinstance(raw, str):
        require(
            text and feature_module.NUMBER_PATTERN.fullmatch(raw) is not None,
            f"{context}: malformed numeric token.",
        )
    else:
        require(
            isinstance(raw, Real),
            f"{context}: expected a real number.",
        )
    try:
        value = float(raw)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{context}: invalid number.") from exc
    require(math.isfinite(value), f"{context}: non-finite number.")
    return value


def load_demand_window(
    path: Path,
    manifest: pd.DataFrame,
    calendar: dict[int, dict[str, Any]],
    first_day: int,
    last_day: int,
    *,
    max_permitted_day: int = VALIDATION_END,
) -> dict[str, list[int]]:
    """Read requested demand without extracting excluded sales fields.

    The default remains days 1..1885. Stage 5A scoring must explicitly
    authorize max_permitted_day=1913.
    """
    require(
        isinstance(max_permitted_day, Integral)
        and not isinstance(max_permitted_day, (bool, np.bool_))
        and 1 <= max_permitted_day <= HISTORY_END,
        "Maximum permitted demand day must be an integer in 1..1913.",
    )
    require(
        isinstance(first_day, Integral)
        and isinstance(last_day, Integral)
        and not isinstance(first_day, (bool, np.bool_))
        and not isinstance(last_day, (bool, np.bool_))
        and 1 <= first_day <= last_day <= max_permitted_day,
        f"Demand interval must lie within days 1..{max_permitted_day}.",
    )
    selected = manifest.set_index("id").to_dict("index")
    observed = {
        series_id: {}
        for series_id in selected
    }
    last_seen = {}

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        names = feature_module.csv_header(
            reader,
            feature_module.SALES_COLUMNS,
            path.name,
        )
        positions = {
            name: names.index(name)
            for name in feature_module.SALES_COLUMNS
        }

        for number, row in enumerate(reader, start=1):
            require(
                len(row) == len(names),
                f"Sales record {number}: incorrect field count.",
            )
            day = feature_module.integer_value(
                row[positions["day_number"]],
                "Sales day",
                text=True,
            )
            require(
                1 <= day <= HISTORY_END,
                "Sales day outside approved history.",
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
                f"Sales record {number}: split/day mismatch.",
            )

            # Check the requested interval before any sales-field access.
            if not first_day <= day <= last_day:
                continue

            series_id = row[positions["id"]]
            require(
                series_id in selected,
                f"Unexpected selected ID {series_id!r}.",
            )
            require(
                day not in observed[series_id],
                f"Duplicate demand key ({series_id}, {day}).",
            )
            require(
                day > last_seen.get(series_id, 0),
                f"{series_id}: demand days are not ordered.",
            )
            last_seen[series_id] = day
            for field in ("item_id", "store_id", "dept_id"):
                require(
                    row[positions[field]]
                    == selected[series_id][field],
                    f"{series_id}: {field} differs from manifest.",
                )
            require(
                row[positions["d"]] == f"d_{day}",
                f"{series_id}: incorrect day ID.",
            )
            require(
                day in calendar
                and row[positions["date"]]
                == calendar[day]["date"].isoformat(),
                f"{series_id}, d_{day}: calendar mismatch.",
            )
            observed[series_id][day] = (
                feature_module.integer_value(
                    row[positions["sales"]],
                    f"{series_id}, d_{day}",
                    text=True,
                )
            )

    result = {}
    expected_days = list(range(first_day, last_day + 1))
    for series_id in sorted(selected):
        require(
            sorted(observed[series_id]) == expected_days,
            f"{series_id}: incomplete requested demand interval.",
        )
        result[series_id] = [
            observed[series_id][day]
            for day in expected_days
        ]
    return result


def validate_predictors(
    frame: pd.DataFrame,
    ids: tuple[str, ...],
) -> pd.DataFrame:
    require(
        isinstance(frame, pd.DataFrame),
        "Predictors must be a DataFrame.",
    )
    require(
        tuple(frame.columns) == PREDICTOR_COLUMNS,
        "Predictor schema/order mismatch.",
    )
    require(len(frame) > 0, "Predictor frame is empty.")
    require(
        all(
            isinstance(value, str) and value in ids
            for value in frame["id"]
        ),
        "Missing or unknown product ID.",
    )

    for column in ("target_event_type_1", "target_event_type_2"):
        require(
            all(
                isinstance(value, str) and value.strip()
                for value in frame[column]
            ),
            f"{column}: missing explicit event category.",
        )

    for column, allowed in (
        ("horizon", set(range(1, 29))),
        ("target_weekday", set(range(7))),
        ("target_month", set(range(1, 13))),
        ("target_snap_CA", {0, 1}),
        ("target_has_event", {0, 1}),
    ):
        require(
            all(
                isinstance(value, Integral)
                and not isinstance(value, (bool, np.bool_))
                and value in allowed
                for value in frame[column]
            ),
            f"{column}: invalid category or integer value.",
        )

    for column in NUMERIC_PREDICTORS:
        require(
            pd.api.types.is_numeric_dtype(frame[column])
            and not pd.api.types.is_bool_dtype(frame[column]),
            f"{column}: expected numeric values.",
        )
        values = frame[column].to_numpy(dtype=float)
        require(
            not np.isinf(values).any(),
            f"{column}: infinite values.",
        )
        if column not in MISSING_ALLOWED_COLUMNS:
            require(
                np.isfinite(values).all(),
                f"{column}: missing numeric values.",
            )

    for column in feature_module.DEMAND_FEATURES:
        require(
            frame[column].ge(0).all(),
            f"{column}: negative demand feature.",
        )
    require(
        frame["price_history_missing_indicator"].isin([0, 1]).all(),
        "Invalid missing-price indicator.",
    )
    require(
        frame["recorded_price_weeks_to_origin"].ge(0).all(),
        "Negative recorded-price-week count.",
    )
    return frame


def numeric_matrix(frame: Any) -> np.ndarray:
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = np.asarray(frame, dtype=np.float32)
    except (FloatingPointError, ValueError, TypeError) as exc:
        raise ValueError(
            "Numeric predictors cannot be represented safely."
        ) from exc
    require(
        not np.isinf(result).any(),
        "Infinite encoded numeric predictors.",
    )
    return result


def validate_design_matrix(matrix: Any) -> np.ndarray:
    result = np.asarray(matrix, dtype=np.float32)
    require(
        result.ndim == 2,
        "Encoded predictors must be two-dimensional.",
    )
    require(
        not np.isinf(result).any(),
        "Encoded predictors contain infinity.",
    )
    return result


def make_estimator(configuration: str) -> XGBRegressor:
    require(
        configuration in CONFIGURATIONS,
        "Unknown model configuration.",
    )
    return XGBRegressor(**CONFIGURATIONS[configuration])


def make_pipeline(
    configuration: str,
    ids: tuple[str, ...],
    estimator_factory: EstimatorFactory | None = None,
) -> Pipeline:
    require(
        configuration in CONFIGURATIONS,
        "Unknown model configuration.",
    )
    factory = (
        make_estimator
        if estimator_factory is None
        else estimator_factory
    )

    # Dense encoding preserves zeros. Omitted sparse tree inputs would be
    # treated as missing, conflating genuine zeros with missing prices.
    preprocessing = ColumnTransformer(
        [
            (
                "categories",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float32,
                ),
                list(CATEGORICAL_PREDICTORS),
            ),
            (
                "numeric",
                FunctionTransformer(
                    numeric_matrix,
                    validate=False,
                ),
                list(NUMERIC_PREDICTORS),
            ),
        ],
        sparse_threshold=0.0,
        remainder="drop",
    )
    return Pipeline([
        (
            "validate",
            FunctionTransformer(
                validate_predictors,
                kw_args={"ids": ids},
                validate=False,
            ),
        ),
        ("preprocess", preprocessing),
        (
            "matrix",
            FunctionTransformer(
                validate_design_matrix,
                validate=False,
            ),
        ),
        ("model", factory(configuration)),
    ])


def predictors_from_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [
        name for name in PREDICTOR_COLUMNS
        if name not in frame.columns
    ]
    require(not missing, f"Missing feature columns: {missing}.")
    return frame[list(PREDICTOR_COLUMNS)].copy()


def build_fit_examples(
    histories: dict[str, list[int]],
    manifest: pd.DataFrame,
    calendar: dict[int, dict[str, Any]],
    prices: dict[tuple[str, str, int], float],
    cutoff: int,
    *,
    max_fitting_cutoff: int = TRAIN_END,
    max_forecast_day: int = VALIDATION_END,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    require(
        set(histories) == set(manifest["id"])
        and all(
            len(values) == cutoff
            for values in histories.values()
        ),
        "Fitting histories must contain exactly days 1 through "
        "the fitting cutoff.",
    )

    origin_boundary = {}
    if max_fitting_cutoff != TRAIN_END:
        origin_boundary["max_cutoff"] = max_fitting_cutoff
    origins = generate_origins(cutoff, **origin_boundary)

    require(
        isinstance(max_forecast_day, Integral)
        and not isinstance(max_forecast_day, (bool, np.bool_))
        and cutoff <= max_forecast_day <= HISTORY_END,
        "Maximum forecast day must cover the fitting cutoff "
        "and cannot exceed 1913.",
    )

    # Preserve the original default helper call and historical week coverage.
    weeks = (
        feature_module.training_weeks(calendar)
        if max_fitting_cutoff == TRAIN_END
        else feature_module.training_weeks(
            calendar,
            cutoff=max_fitting_cutoff,
        )
    )
    available_weeks = {
        week["wm_yr_wk"]
        for week in weeks
        if week["first_date"] <= calendar[cutoff]["date"]
    }
    partition_prices = {
        key: value
        for key, value in prices.items()
        if key[2] in available_weeks
    }

    example_boundaries = {}
    if max_fitting_cutoff != TRAIN_END:
        example_boundaries["fitting_cutoff"] = cutoff
    if max_forecast_day != VALIDATION_END:
        example_boundaries["max_forecast_day"] = max_forecast_day

    examples = feature_module.build_training_examples(
        histories,
        manifest.set_index("id")["item_id"].to_dict(),
        calendar,
        partition_prices,
        origins,
        **example_boundaries,
    )
    require(
        len(examples)
        == len(manifest) * len(origins) * HORIZON,
        "Incorrect fitting-example count.",
    )
    require(
        not examples.duplicated(
            ["id", "origin_day", "horizon"]
        ).any(),
        "Duplicate fitting-example keys.",
    )
    require(
        examples["target_day"].le(cutoff).all()
        and examples["target_day"].eq(
            examples["origin_day"] + examples["horizon"]
        ).all(),
        "Fitting labels exceed the fitting cutoff or are misaligned.",
    )
    targets = np.asarray([
        feature_module.integer_value(value, "Fitting target")
        for value in examples["target_sales"]
    ], dtype=float)

    diagnostics = {
        "fitting_cutoff": cutoff,
        "historical_origin_count": len(origins),
        "historical_origins": list(origins),
        "first_historical_origin": origins[0],
        "last_historical_origin": origins[-1],
        "origin_weekday_offsets": sorted(
            {(origin - 1) % 7 for origin in origins}
        ),
        "training_example_count": len(examples),
        "maximum_training_target_day": int(
            examples["target_day"].max()
        ),
        "selected_product_count": len(manifest),
        "missing_price_cells": {
            name: int(examples[name].isna().sum())
            for name in feature_module.PRICE_FEATURES
        },
    }
    return (
        predictors_from_features(examples),
        targets,
        diagnostics,
    )


def clip_predictions(
    raw: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(raw, dtype=float)
    require(
        values.ndim == 1 and len(values) > 0,
        "Invalid raw prediction shape.",
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


def prepare_partition(
    origin: int,
    manifest: pd.DataFrame,
    calendar: dict[int, dict[str, Any]],
    prices: dict[tuple[str, str, int], float],
    read_demand: DemandReader,
    *,
    max_fitting_cutoff: int = TRAIN_END,
    max_forecast_day: int = VALIDATION_END,
) -> dict[str, Any]:
    """Prepare shared input matrices without reading future actuals."""
    origin_boundary = {}
    if max_fitting_cutoff != TRAIN_END:
        origin_boundary["max_cutoff"] = max_fitting_cutoff
    generate_origins(origin, **origin_boundary)

    require(
        isinstance(max_forecast_day, Integral)
        and not isinstance(max_forecast_day, (bool, np.bool_))
        and 1 <= max_forecast_day <= HISTORY_END,
        "Maximum forecast day must be an integer in 1..1913.",
    )
    require(
        origin + HORIZON <= max_forecast_day,
        (
            "Forecast horizon would enter final test."
            if max_forecast_day == VALIDATION_END
            else "Forecast horizon exceeds the explicitly permitted boundary."
        ),
    )

    histories = read_demand(1, origin)

    fitting_boundaries = {}
    if max_fitting_cutoff != TRAIN_END:
        fitting_boundaries["max_fitting_cutoff"] = max_fitting_cutoff
    if max_forecast_day != VALIDATION_END:
        fitting_boundaries["max_forecast_day"] = max_forecast_day

    features, targets, diagnostics = build_fit_examples(
        histories,
        manifest,
        calendar,
        prices,
        origin,
        **fitting_boundaries,
    )
    require(
        np.isfinite(targets).all(),
        "Non-finite fitting targets.",
    )

    ids = tuple(sorted(manifest["id"]))
    item_ids = manifest.set_index("id")["item_id"].to_dict()
    weeks = (
        feature_module.training_weeks(calendar)
        if max_fitting_cutoff == TRAIN_END
        else feature_module.training_weeks(
            calendar,
            cutoff=max_fitting_cutoff,
        )
    )
    forecast_features = []
    keys = []

    feature_boundaries = {}
    if max_fitting_cutoff != TRAIN_END:
        feature_boundaries["max_origin"] = origin
    if max_forecast_day != VALIDATION_END:
        feature_boundaries["max_forecast_day"] = max_forecast_day

    for series_id in ids:
        forecast_features.extend(
            feature_module.build_feature_rows(
                series_id,
                item_ids[series_id],
                histories[series_id],
                origin,
                calendar,
                prices,
                weeks,
                **feature_boundaries,
            )
        )
        for horizon in range(1, HORIZON + 1):
            target = origin + horizon
            keys.append({
                "id": series_id,
                "item_id": item_ids[series_id],
                "model": MODEL,
                "forecast_origin_day": f"d_{origin}",
                "horizon": horizon,
                "d": f"d_{target}",
                "date": calendar[target]["date"].isoformat(),
            })

    prediction_features = predictors_from_features(
        pd.DataFrame(
            forecast_features,
            columns=feature_module.FEATURE_COLUMNS,
        )
    )
    require(
        len(prediction_features) == len(manifest) * HORIZON,
        "Incorrect prediction-feature count.",
    )
    return {
        "origin": origin,
        "ids": ids,
        "features": features,
        "targets": targets,
        "prediction_features": prediction_features,
        "keys": pd.DataFrame(keys),
        "diagnostics": diagnostics,
    }


def fit_prepared_partition(
    partition: dict[str, Any],
    configuration: str,
    estimator_factory: EstimatorFactory | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit a separate preprocessing/model pipeline on prepared inputs."""
    require(
        configuration in CONFIGURATIONS,
        "Unknown model configuration.",
    )
    features = partition["features"]
    targets = partition["targets"]
    prediction_features = partition["prediction_features"]

    if configuration == "B_tweedie":
        require(
            np.any(targets > 0),
            "Tweedie fitting requires some positive training demand.",
        )

    pipeline = make_pipeline(
        configuration,
        partition["ids"],
        estimator_factory,
    )
    pipeline.fit(features, targets)
    raw = np.asarray(
        pipeline.predict(prediction_features),
        dtype=float,
    )
    clipped, clipping = clip_predictions(raw)
    require(
        len(clipped) == len(partition["keys"]),
        "Incorrect frozen forecast count.",
    )
    forecasts = partition["keys"].copy()
    forecasts["raw_predicted_sales"] = raw
    forecasts["predicted_sales"] = clipped

    encoder = (
        pipeline.named_steps["preprocess"]
        .named_transformers_["categories"]
    )
    unknown_categories = {}
    for name, learned in zip(
        CATEGORICAL_PREDICTORS,
        encoder.categories_,
    ):
        unknown_categories[name] = sorted(
            set(prediction_features[name].tolist())
            - set(learned.tolist()),
            key=str,
        )

    diagnostics = {
        **partition["diagnostics"],
        "configuration": configuration,
        "objective": CONFIGURATIONS[configuration]["objective"],
        "forecast_origin_day": f"d_{partition['origin']}",
        "prediction_count": len(forecasts),
        "clipping": clipping,
        "categorical_vocabulary": {
            name: values.tolist()
            for name, values in zip(
                CATEGORICAL_PREDICTORS,
                encoder.categories_,
            )
        },
        "unseen_prediction_categories": unknown_categories,
        "preprocessing_fit_observations": len(features),
        "encoded_feature_count": int(
            pipeline.named_steps["preprocess"]
            .transform(features.iloc[:1]).shape[1]
        ),
    }
    return (
        forecasts.sort_values(
            ["model", "id", "horizon"]
        ).reset_index(drop=True),
        diagnostics,
    )


def fit_and_forecast(
    origin: int,
    configuration: str,
    manifest: pd.DataFrame,
    calendar: dict[int, dict[str, Any]],
    prices: dict[tuple[str, str, int], float],
    read_demand: DemandReader,
    estimator_factory: EstimatorFactory | None = None,
    *,
    max_fitting_cutoff: int = TRAIN_END,
    max_forecast_day: int = VALIDATION_END,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Prepare one partition and freeze forecasts before reading actuals."""
    boundaries = {}
    if max_fitting_cutoff != TRAIN_END:
        boundaries["max_fitting_cutoff"] = max_fitting_cutoff
    if max_forecast_day != VALIDATION_END:
        boundaries["max_forecast_day"] = max_forecast_day

    partition = prepare_partition(
        origin,
        manifest,
        calendar,
        prices,
        read_demand,
        **boundaries,
    )
    return fit_prepared_partition(
        partition,
        configuration,
        estimator_factory,
    )


def attach_actuals(
    forecasts: pd.DataFrame,
    actuals: dict[str, list[int]],
    origin: int,
) -> pd.DataFrame:
    require(
        set(forecasts["id"]) == set(actuals),
        "Actual/forecast ID mismatch.",
    )
    require(
        all(
            len(values) == HORIZON
            for values in actuals.values()
        ),
        "Each product requires exactly 28 actual observations.",
    )
    require(
        not forecasts.duplicated(
            ["id", "model", "horizon"]
        ).any(),
        "Duplicate forecast keys.",
    )
    result = forecasts.copy()
    result["actual_sales"] = [
        feature_module.integer_value(
            actuals[row.id][row.horizon - 1],
            "Evaluation actual",
        )
        for row in result.itertuples(index=False)
    ]
    require(
        result["forecast_origin_day"].eq(f"d_{origin}").all(),
        "Forecast origin mismatch during alignment.",
    )
    return result[list(PREDICTION_COLUMNS)]


def metric_tables(
    predictions: pd.DataFrame,
    origin: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate aligned 28-day forecasts using src.evaluate."""
    require(
        isinstance(origin, Integral)
        and not isinstance(origin, (bool, np.bool_)),
        "Evaluation origin must be an integer.",
    )
    require(
        set(BASELINE_PREDICTION_COLUMNS).issubset(
            predictions.columns
        ),
        "Metric input is missing required prediction columns.",
    )
    require(len(predictions) > 0, "Metric input is empty.")
    require(
        not predictions.duplicated(
            ["id", "model", "horizon"]
        ).any(),
        "Duplicate product/model/horizon metric input.",
    )

    products = []
    summaries = []
    for model_name, model_rows in predictions.groupby(
        "model",
        sort=True,
    ):
        for series_id, group in model_rows.groupby(
            "id",
            sort=True,
        ):
            group = group.sort_values("horizon")
            require(
                len(group) == HORIZON
                and group["horizon"].tolist()
                == list(range(1, HORIZON + 1))
                and all(
                    isinstance(value, Integral)
                    and not isinstance(value, (bool, np.bool_))
                    for value in group["horizon"]
                ),
                f"{model_name}/{series_id}: expected exactly "
                "28 integer horizons.",
            )
            require(
                group["item_id"].nunique(dropna=False) == 1
                and isinstance(group["item_id"].iloc[0], str)
                and bool(group["item_id"].iloc[0].strip()),
                f"{model_name}/{series_id}: inconsistent item identifier.",
            )
            require(
                group["forecast_origin_day"].eq(
                    f"d_{origin}"
                ).all()
                and group["d"].tolist()
                == [
                    f"d_{origin + horizon}"
                    for horizon in range(1, HORIZON + 1)
                ],
                f"{model_name}/{series_id}: evaluation "
                "origin/day misalignment.",
            )

            result = calculate_metrics(
                group["actual_sales"].tolist(),
                group["predicted_sales"].tolist(),
            )
            require(
                result["observations"] == len(group),
                "Evaluation module returned an inconsistent "
                "observation count.",
            )
            products.append({
                "id": series_id,
                "item_id": group["item_id"].iloc[0],
                "model": model_name,
                **{
                    name: result[name]
                    for name in METRICS
                },
                "observations": len(group),
            })

        records = [
            {
                "id": row.id,
                "day_number": int(origin) + row.horizon,
                "actual": row.actual_sales,
                "predicted": row.predicted_sales,
            }
            for row in model_rows.itertuples(index=False)
        ]
        for level in LEVELS:
            result = evaluate_records(records, level)
            expected = (
                len(model_rows)
                if level == "pooled_product_day"
                else HORIZON
            )
            require(
                result["observations"] == expected,
                f"{level}: inconsistent evaluation observation count.",
            )
            summaries.append({
                "model": model_name,
                "evaluation_level": level,
                "observations": result["observations"],
                **{
                    name: result[name]
                    for name in METRICS
                },
            })

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


def select_configuration(
    backtest_predictions: pd.DataFrame,
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Calculate combined metrics from individual backtest product-days."""
    require(
        set(backtest_predictions["configuration"])
        == set(CONFIGURATION_ORDER),
        "Both predeclared configurations must have backtest predictions.",
    )
    require(
        not backtest_predictions.duplicated([
            "configuration",
            "origin_day",
            "id",
            "horizon",
        ]).any(),
        "Duplicate combined-backtest prediction keys.",
    )

    combined = {}
    reference_actuals = None
    for configuration in CONFIGURATION_ORDER:
        rows = backtest_predictions.loc[
            backtest_predictions["configuration"].eq(configuration)
        ].sort_values(["origin_day", "id", "horizon"])
        signature = [
            (
                row.origin_day,
                row.id,
                row.horizon,
                row.actual_sales,
            )
            for row in rows.itertuples(index=False)
        ]
        if reference_actuals is None:
            reference_actuals = signature
        else:
            require(
                signature == reference_actuals,
                "Configuration backtests do not share "
                "identical keys/actuals.",
            )

        # Backtest windows are disjoint and have calendar gaps. Directly
        # evaluate their concatenated pairs rather than inventing consecutive
        # day IDs or averaging fold percentages.
        combined[configuration] = calculate_metrics(
            rows["actual_sales"].tolist(),
            rows["predicted_sales"].tolist(),
        )

    a = combined["A_squared_error"]["wape_pct"]
    b = combined["B_tweedie"]["wape_pct"]
    require(
        (a is None) == (b is None),
        "Inconsistent undefined WAPE across identical backtest actuals.",
    )
    selected = (
        "B_tweedie"
        if a is not None and b is not None and b < a
        else "A_squared_error"
    )
    return selected, combined


def run_backtests(
    manifest: pd.DataFrame,
    calendar: dict[int, dict[str, Any]],
    prices: dict[tuple[str, str, int], float],
    read_demand: DemandReader,
    origins: tuple[int, ...] = BACKTEST_ORIGINS,
    estimator_factory: EstimatorFactory | None = None,
) -> dict[str, Any]:
    require(
        bool(origins)
        and origins == tuple(sorted(set(origins))),
        "Backtest origins must be unique and chronological.",
    )
    metric_rows = []
    prediction_frames = []
    diagnostics = []

    for origin in origins:
        require(
            origin + HORIZON <= TRAIN_END,
            "Backtest enters held-out periods.",
        )

        # Prepare once per origin. Each configuration still fits its own
        # preprocessing and estimator independently.
        partition = prepare_partition(
            origin,
            manifest,
            calendar,
            prices,
            read_demand,
        )
        frozen = {}
        for configuration in CONFIGURATION_ORDER:
            forecasts, information = fit_prepared_partition(
                partition,
                configuration,
                estimator_factory,
            )
            frozen[configuration] = forecasts
            diagnostics.append(information)

        del partition

        # Both configurations have frozen every forecast before actuals
        # from this backtest's horizon are obtained.
        actuals = read_demand(
            origin + 1,
            origin + HORIZON,
        )
        for configuration in CONFIGURATION_ORDER:
            predictions = attach_actuals(
                frozen[configuration],
                actuals,
                origin,
            )
            _, summary = metric_tables(predictions, origin)
            for record in summary.to_dict("records"):
                metric_rows.append({
                    "configuration": configuration,
                    "objective": CONFIGURATIONS[
                        configuration
                    ]["objective"],
                    "scope": "fold",
                    "forecast_origin_day": f"d_{origin}",
                    "fitting_cutoff": origin,
                    "evaluation_level": record["evaluation_level"],
                    "observations": record["observations"],
                    **{
                        name: record[name]
                        for name in METRICS
                    },
                })
            predictions["configuration"] = configuration
            predictions["origin_day"] = origin
            prediction_frames.append(predictions)

    all_predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )
    selected, combined = select_configuration(all_predictions)
    for configuration in CONFIGURATION_ORDER:
        metric_rows.append({
            "configuration": configuration,
            "objective": CONFIGURATIONS[
                configuration
            ]["objective"],
            "scope": "combined",
            "forecast_origin_day": "combined",
            "fitting_cutoff": None,
            "evaluation_level": "pooled_product_day",
            **combined[configuration],
        })

    metrics = pd.DataFrame(
        metric_rows,
        columns=BACKTEST_METRIC_COLUMNS,
        dtype=object,
    ).sort_values([
        "configuration",
        "scope",
        "forecast_origin_day",
        "evaluation_level",
    ]).reset_index(drop=True)

    return {
        "metrics": metrics,
        "selected_configuration": selected,
        "combined_metrics": combined,
        "diagnostics": diagnostics,
        "predictions": all_predictions,
    }


def load_benchmark(
    predictions_path: Path,
    summary_path: Path,
    model_name: str,
    current_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = read_table(
        predictions_path,
        BASELINE_PREDICTION_COLUMNS,
    )
    benchmark = raw.loc[raw["model"].eq(model_name)].copy()
    require(
        len(benchmark) == len(current_predictions),
        f"{model_name}: prediction count mismatch.",
    )
    benchmark["horizon"] = [
        feature_module.integer_value(
            value,
            "Benchmark horizon",
            text=True,
        )
        for value in benchmark["horizon"]
    ]
    require(
        benchmark["horizon"].between(1, HORIZON).all()
        and not benchmark.duplicated(
            ["id", "horizon"]
        ).any(),
        f"{model_name}: invalid prediction keys.",
    )
    benchmark = (
        benchmark.sort_values(["id", "horizon"])
        .reset_index(drop=True)
    )
    current = (
        current_predictions.sort_values(["id", "horizon"])
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
            benchmark[column].tolist()
            == current[column].tolist(),
            f"{model_name}: alignment mismatch in {column}.",
        )

    benchmark["actual_sales"] = pd.Series([
        feature_module.integer_value(
            value,
            "Benchmark actual",
            text=True,
        )
        for value in benchmark["actual_sales"]
    ], dtype=object)
    require(
        benchmark["actual_sales"].tolist()
        == current["actual_sales"].tolist(),
        f"{model_name}: validation actual mismatch.",
    )
    benchmark["predicted_sales"] = [
        finite_number(
            value,
            "Benchmark prediction",
            text=True,
        )
        for value in benchmark["predicted_sales"]
    ]
    require(
        benchmark["predicted_sales"].ge(0).all(),
        f"{model_name}: negative exported predictions.",
    )

    products, recalculated = metric_tables(
        benchmark,
        TRAIN_END,
    )
    stored = read_table(summary_path, SUMMARY_COLUMNS)
    stored = stored.loc[
        stored["model"].eq(model_name)
    ].copy()
    require(
        len(stored) == len(LEVELS)
        and set(stored["evaluation_level"]) == set(LEVELS)
        and not stored["evaluation_level"].duplicated().any(),
        f"{model_name}: stored metric coverage mismatch.",
    )

    records = []
    for record in stored.to_dict("records"):
        record = {
            name: record[name]
            for name in SUMMARY_COLUMNS
        }
        record["observations"] = feature_module.integer_value(
            record["observations"],
            "Benchmark observations",
            text=True,
        )
        for name in METRICS:
            record[name] = (
                None
                if name == "wape_pct" and record[name] == ""
                else finite_number(
                    record[name],
                    f"Benchmark {name}",
                    text=True,
                )
            )
        checked = recalculated.loc[
            recalculated["evaluation_level"].eq(
                record["evaluation_level"]
            )
        ].iloc[0]
        require(
            record["observations"] == checked["observations"],
            f"{model_name}: stored observation count mismatch.",
        )
        for name in METRICS:
            saved = record[name]
            observed = checked[name]
            require(
                (saved is None and observed is None)
                or (
                    saved is not None
                    and observed is not None
                    and math.isclose(
                        saved,
                        observed,
                        rel_tol=1e-10,
                        abs_tol=1e-10,
                    )
                ),
                f"{model_name}: stored {name} "
                "disagrees with predictions.",
            )
        records.append(record)

    return (
        pd.DataFrame(
            records,
            columns=SUMMARY_COLUMNS,
            dtype=object,
        )
        .sort_values("evaluation_level")
        .reset_index(drop=True),
        products,
    )


def comparison_table(
    xgb_summary: pd.DataFrame,
    seasonal_summary: pd.DataFrame,
    ridge_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for table in (
        seasonal_summary,
        ridge_summary,
        xgb_summary,
    ):
        for record in table.to_dict("records"):
            level = record["evaluation_level"]
            for label, benchmark in (
                ("seasonal", seasonal_summary),
                ("ridge", ridge_summary),
            ):
                reference = benchmark.loc[
                    benchmark["evaluation_level"].eq(level)
                ].iloc[0]
                require(
                    record["observations"]
                    == reference["observations"],
                    "Comparison observation counts differ.",
                )
                value = record["wape_pct"]
                base = reference["wape_pct"]
                record[f"difference_wape_pct_vs_{label}"] = (
                    None
                    if value is None or base is None
                    else value - base
                )
                record[
                    f"relative_wape_improvement_pct_vs_{label}"
                ] = (
                    None
                    if value is None or base is None or base == 0
                    else 100.0 * (base - value) / base
                )
            rows.append(record)

    return pd.DataFrame(
        rows,
        columns=COMPARISON_COLUMNS,
        dtype=object,
    ).sort_values(
        ["evaluation_level", "model"]
    ).reset_index(drop=True)


def product_exceptions(
    xgb_products: pd.DataFrame,
    benchmark_products: pd.DataFrame,
) -> dict[str, Any]:
    benchmark = benchmark_products.set_index("id")
    worse = []
    undefined = []
    for record in xgb_products.to_dict("records"):
        value = record["wape_pct"]
        base = benchmark.at[record["id"], "wape_pct"]
        if value is None or base is None:
            undefined.append(record["id"])
        elif value > base:
            worse.append(record["id"])
    return {
        "products_with_worse_xgb_wape": sorted(worse),
        "products_with_undefined_wape_comparison": sorted(undefined),
    }


def verify_outputs(
    backtests: pd.DataFrame,
    predictions: pd.DataFrame,
    products: pd.DataFrame,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    manifest: pd.DataFrame,
    actuals: dict[str, list[int]],
    calendar: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    count = len(manifest)
    require(
        count == PRODUCT_COUNT,
        "Approved outputs require 20 selected products.",
    )
    require(
        not manifest["id"].duplicated().any(),
        "Manifest IDs are duplicated.",
    )
    for frame, schema in (
        (backtests, BACKTEST_METRIC_COLUMNS),
        (predictions, PREDICTION_COLUMNS),
        (products, PRODUCT_COLUMNS),
        (summary, SUMMARY_COLUMNS),
        (comparison, COMPARISON_COLUMNS),
    ):
        require(
            tuple(frame.columns) == schema,
            "Output schema mismatch.",
        )

    ids = set(manifest["id"])
    require(
        set(actuals) == ids
        and all(
            len(values) == HORIZON
            for values in actuals.values()
        ),
        "Validation actual coverage mismatch.",
    )
    require(
        len(predictions) == count * HORIZON,
        "Prediction count mismatch.",
    )
    require(
        set(predictions["id"]) == ids
        and predictions["model"].eq(MODEL).all()
        and predictions["forecast_origin_day"].eq(
            "d_1857"
        ).all()
        and not predictions.duplicated(
            ["id", "model", "horizon"]
        ).any()
        and not predictions.isna().any().any(),
        "Prediction identity, origin, uniqueness, "
        "or completeness failure.",
    )
    require(
        predictions.reset_index(drop=True).equals(
            predictions.sort_values(
                ["model", "id", "horizon"]
            ).reset_index(drop=True)
        ),
        "Predictions are not deterministically ordered.",
    )

    items = manifest.set_index("id")["item_id"].to_dict()
    for series_id, group in predictions.groupby(
        "id",
        sort=True,
    ):
        require(
            len(group) == HORIZON
            and group["horizon"].tolist()
            == list(range(1, HORIZON + 1)),
            f"{series_id}: invalid horizon coverage.",
        )
        for row in group.itertuples(index=False):
            require(
                isinstance(row.horizon, Integral)
                and not isinstance(
                    row.horizon,
                    (bool, np.bool_),
                ),
                "Forecast horizons must be integers.",
            )
            target = TRAIN_END + row.horizon
            require(
                target <= VALIDATION_END
                and row.item_id == items[series_id]
                and row.d == f"d_{target}"
                and row.date
                == calendar[target]["date"].isoformat()
                and row.actual_sales
                == actuals[series_id][row.horizon - 1]
                and math.isfinite(row.raw_predicted_sales)
                and math.isfinite(row.predicted_sales)
                and row.predicted_sales
                == max(0.0, row.raw_predicted_sales),
                "Prediction calendar/source/clipping "
                "reconciliation failed.",
            )

    require(
        len(products) == count
        and set(products["id"]) == ids
        and not products.duplicated(["id", "model"]).any()
        and products["model"].eq(MODEL).all()
        and products["observations"].eq(HORIZON).all(),
        "Per-product metric coverage/count mismatch.",
    )
    for row in products.itertuples(index=False):
        require(
            row.item_id == items[row.id],
            "Per-product metric item identifier mismatch.",
        )

    require(
        len(summary) == len(LEVELS)
        and set(summary["evaluation_level"]) == set(LEVELS)
        and not summary.duplicated(
            ["model", "evaluation_level"]
        ).any()
        and summary["model"].eq(MODEL).all(),
        "Summary coverage mismatch.",
    )
    for record in summary.to_dict("records"):
        expected = (
            count * HORIZON
            if record["evaluation_level"] == "pooled_product_day"
            else HORIZON
        )
        require(
            record["observations"] == expected,
            "Summary observation mismatch.",
        )

    expected_comparison_keys = {
        (model, level)
        for model in (MODEL, SEASONAL_MODEL, RIDGE_MODEL)
        for level in LEVELS
    }
    require(
        len(comparison) == len(expected_comparison_keys)
        and not comparison.duplicated(
            ["model", "evaluation_level"]
        ).any()
        and set(zip(
            comparison["model"],
            comparison["evaluation_level"],
        )) == expected_comparison_keys,
        "Comparison coverage mismatch.",
    )
    for record in comparison.to_dict("records"):
        expected = (
            count * HORIZON
            if record["evaluation_level"] == "pooled_product_day"
            else HORIZON
        )
        require(
            record["observations"] == expected,
            "Comparison observation mismatch.",
        )

    expected_backtest_keys = {
        (configuration, "fold", f"d_{origin}", level)
        for configuration in CONFIGURATION_ORDER
        for origin in BACKTEST_ORIGINS
        for level in LEVELS
    } | {
        (
            configuration,
            "combined",
            "combined",
            "pooled_product_day",
        )
        for configuration in CONFIGURATION_ORDER
    }
    key_columns = [
        "configuration",
        "scope",
        "forecast_origin_day",
        "evaluation_level",
    ]
    require(
        len(backtests) == len(expected_backtest_keys)
        and not backtests.duplicated(key_columns).any()
        and set(
            backtests[key_columns].itertuples(
                index=False,
                name=None,
            )
        ) == expected_backtest_keys,
        "Backtest metric coverage mismatch.",
    )
    for record in backtests.to_dict("records"):
        require(
            record["objective"]
            == CONFIGURATIONS[
                record["configuration"]
            ]["objective"],
            "Backtest objective mismatch.",
        )
        if record["scope"] == "combined":
            require(
                pd.isna(record["fitting_cutoff"])
                and record["observations"]
                == count * HORIZON * len(BACKTEST_ORIGINS),
                "Combined-backtest observation/cutoff mismatch.",
            )
        else:
            origin = int(record["forecast_origin_day"][2:])
            require(
                record["fitting_cutoff"] == origin
                and origin + HORIZON <= TRAIN_END,
                "Backtest fitting cutoff/window mismatch.",
            )
            expected = (
                count * HORIZON
                if record["evaluation_level"]
                == "pooled_product_day"
                else HORIZON
            )
            require(
                record["observations"] == expected,
                "Backtest fold observation mismatch.",
            )

    for frame in (backtests, products, summary, comparison):
        for record in frame.to_dict("records"):
            for name in METRICS:
                value = record[name]
                require(
                    (name == "wape_pct" and value is None)
                    or (
                        isinstance(value, Real)
                        and math.isfinite(value)
                        and (name == "bias_units" or value >= 0)
                    ),
                    f"Invalid output metric {name}.",
                )

    return {
        "status": "PASS",
        "prediction_rows": len(predictions),
        "per_product_metric_rows": len(products),
        "summary_metric_rows": len(summary),
        "comparison_rows": len(comparison),
        "backtest_metric_rows": len(backtests),
        "duplicate_prediction_keys": 0,
        "validation_actuals_reconciled": True,
        "benchmark_keys_dates_actuals_and_metrics_reconciled": True,
        "metric_implementation": "src.evaluate",
        "test_sales_accessed": False,
    }


def verify_staged_csv(
    expected: pd.DataFrame,
    path: Path,
    key_columns: tuple[str, ...],
) -> None:
    """Verify schema, row order, keys, and typed round-trip cell values."""
    require(
        len(expected.columns) == len(set(expected.columns)),
        "Expected export contains duplicate columns.",
    )
    require(
        set(key_columns).issubset(expected.columns),
        "Export key columns are missing.",
    )
    require(
        not expected.duplicated(list(key_columns)).any(),
        f"{path.name}: expected export has duplicate keys.",
    )
    actual = read_table(path, tuple(expected.columns))
    require(
        tuple(actual.columns) == tuple(expected.columns),
        f"{path.name}: round-trip schema mismatch.",
    )
    require(
        len(actual) == len(expected),
        f"{path.name}: round-trip row-count mismatch.",
    )
    require(
        not actual.duplicated(list(key_columns)).any(),
        f"{path.name}: round-trip keys are duplicated.",
    )

    for column in expected.columns:
        require(
            column in (
                CSV_STRING_COLUMNS
                | CSV_INTEGER_COLUMNS
                | CSV_FLOAT_COLUMNS
            ),
            f"{path.name}: no verification rule for {column}.",
        )
        for row_number, (source, token) in enumerate(
            zip(
                expected[column].tolist(),
                actual[column].tolist(),
            ),
            start=1,
        ):
            context = f"{path.name}, row {row_number}, {column}"
            missing = bool(pd.isna(source))
            if missing:
                require(
                    column in CSV_NULLABLE_COLUMNS
                    and token == "",
                    f"{context}: unexpected null or null serialization.",
                )
                continue

            require(
                token != "",
                f"{context}: unexpected empty field.",
            )
            if column in CSV_STRING_COLUMNS:
                require(
                    isinstance(source, str)
                    and token == source,
                    f"{context}: identifier/string mismatch.",
                )
            elif column in CSV_INTEGER_COLUMNS:
                require(
                    not isinstance(source, (bool, np.bool_)),
                    f"{context}: boolean integer value.",
                )
                try:
                    source_number = Decimal(str(source))
                    exported_number = Decimal(token)
                except (InvalidOperation, ValueError) as exc:
                    raise ValueError(
                        f"{context}: malformed integer serialization."
                    ) from exc
                require(
                    source_number.is_finite()
                    and exported_number.is_finite()
                    and source_number
                    == source_number.to_integral_value()
                    and exported_number
                    == exported_number.to_integral_value()
                    and exported_number == source_number,
                    f"{context}: exact integer mismatch.",
                )
            else:
                source_number = finite_number(
                    source,
                    context,
                )
                exported_number = finite_number(
                    token,
                    context,
                    text=True,
                )
                require(
                    math.isclose(
                        exported_number,
                        source_number,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ),
                    f"{context}: floating-point mismatch.",
                )


def publish_outputs(
    root: Path,
    backtests: pd.DataFrame,
    predictions: pd.DataFrame,
    products: pd.DataFrame,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    metadata: dict[str, Any],
) -> list[Path]:
    directory = root / "outputs/evaluation"
    directory.mkdir(parents=True, exist_ok=True)
    names = (
        "xgb_backtest_metrics.csv",
        "xgb_predictions.csv",
        "xgb_metrics_by_product.csv",
        "xgb_metrics_summary.csv",
        "xgb_model_comparison.csv",
        "xgb_run_metadata.json",
    )
    destinations = [directory / name for name in names]
    replaced = []

    try:
        with tempfile.TemporaryDirectory(
            prefix=".xgb-stage-",
            dir=directory,
        ) as temp:
            staged = [Path(temp) / name for name in names]
            for frame, path in zip(
                (
                    backtests,
                    predictions,
                    products,
                    summary,
                    comparison,
                ),
                staged[:5],
            ):
                frame.to_csv(
                    path,
                    index=False,
                    na_rep="",
                    lineterminator="\n",
                )
                verify_staged_csv(
                    frame,
                    path,
                    CSV_KEY_COLUMNS[path.name],
                )

            payload = json.dumps(
                metadata,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ) + "\n"
            staged[5].write_text(payload, encoding="utf-8")
            with staged[5].open("r", encoding="utf-8") as handle:
                require(
                    json.load(handle) == metadata,
                    "Staged JSON differs.",
                )

            staged_bytes = []
            for path in staged:
                require(
                    path.stat().st_size > 0,
                    f"Empty output {path.name}.",
                )
                staged_bytes.append(path.read_bytes())

            # Each replacement is atomic individually, not as a six-file set.
            for source, destination in zip(staged, destinations):
                source.replace(destination)
                replaced.append(destination.name)
            for path, expected in zip(destinations, staged_bytes):
                require(
                    path.is_file()
                    and path.read_bytes() == expected,
                    f"Published output verification failed: {path.name}",
                )
    except Exception as exc:
        raise RuntimeError(
            "XGBoost output publication failed. Replaced this attempt: "
            f"{', '.join(replaced) or 'none'}. Outputs may be from "
            "different runs; rerun successfully before use. "
            f"Cause: {exc}"
        ) from exc
    return destinations


def load_feature_audit(
    root: Path,
    manifest: pd.DataFrame,
) -> dict[str, Any]:
    path = root / "outputs/evaluation/xgb_feature_audit.json"
    with path.open("r", encoding="utf-8-sig") as handle:
        audit = json.load(
            handle,
            parse_constant=feature_module.reject_json_constant,
        )
    require(
        isinstance(audit, dict)
        and audit.get("status") == "PASS"
        and type(audit.get("selected_product_count")) is int
        and audit["selected_product_count"] == len(manifest)
        and audit.get("selected_ids_in_rank_order")
        == manifest["id"].tolist()
        and audit.get("feature_names")
        == list(feature_module.FEATURE_COLUMNS),
        "Feature-audit report is missing, incompatible, or inconsistent.",
    )
    return audit


def run_pipeline(root: Path = ROOT) -> dict[str, Any]:
    check_feature_interface()
    root = Path(root)
    paths = {
        "sales": root / "data/processed/selected_sales.csv",
        "manifest": root / "data/processed/selected_products.csv",
        "selection_metadata": (
            root / "data/processed/selection_metadata.json"
        ),
        "calendar": root / "data/raw/calendar.csv",
        "prices": root / "data/raw/sell_prices.csv",
        "feature_audit": (
            root / "outputs/evaluation/xgb_feature_audit.json"
        ),
        "seasonal_predictions": (
            root / "outputs/evaluation/baseline_predictions.csv"
        ),
        "seasonal_summary": (
            root / "outputs/evaluation/baseline_metrics_summary.csv"
        ),
        "ridge_predictions": (
            root / "outputs/evaluation/ridge_predictions.csv"
        ),
        "ridge_summary": (
            root / "outputs/evaluation/ridge_metrics_summary.csv"
        ),
    }
    for path in paths.values():
        require(path.is_file(), f"Required input missing: {path}")

    manifest = feature_module.load_selection(root)
    require(
        len(manifest) == PRODUCT_COUNT,
        "Approved run requires 20 products.",
    )
    load_feature_audit(root, manifest)
    calendar = feature_module.load_calendar(paths["calendar"])
    weeks = feature_module.training_weeks(calendar)
    prices = feature_module.load_prices(
        paths["prices"],
        manifest,
        weeks,
    )

    def read_demand(
        first: int,
        last: int,
    ) -> dict[str, list[int]]:
        return load_demand_window(
            paths["sales"],
            manifest,
            calendar,
            first,
            last,
        )

    backtesting = run_backtests(
        manifest,
        calendar,
        prices,
        read_demand,
    )
    for configuration in CONFIGURATION_ORDER:
        require(
            backtesting["combined_metrics"][
                configuration
            ]["observations"]
            == PRODUCT_COUNT * HORIZON * len(BACKTEST_ORIGINS),
            "Expected 1680 combined backtest observations "
            "per configuration.",
        )

    chosen = backtesting["selected_configuration"]
    forecasts, final_diagnostics = fit_and_forecast(
        TRAIN_END,
        chosen,
        manifest,
        calendar,
        prices,
        read_demand,
    )

    # First access to actual validation demand occurs after forecasts exist.
    actuals = read_demand(
        TRAIN_END + 1,
        VALIDATION_END,
    )
    predictions = attach_actuals(
        forecasts,
        actuals,
        TRAIN_END,
    )
    products, summary = metric_tables(
        predictions,
        TRAIN_END,
    )

    seasonal_summary, seasonal_products = load_benchmark(
        paths["seasonal_predictions"],
        paths["seasonal_summary"],
        SEASONAL_MODEL,
        predictions,
    )
    ridge_summary, ridge_products = load_benchmark(
        paths["ridge_predictions"],
        paths["ridge_summary"],
        RIDGE_MODEL,
        predictions,
    )
    comparison = comparison_table(
        summary,
        seasonal_summary,
        ridge_summary,
    )
    integrity = verify_outputs(
        backtesting["metrics"],
        predictions,
        products,
        summary,
        comparison,
        manifest,
        actuals,
        calendar,
    )

    metadata = {
        "dataset": "Walmart M5 Forecasting Accuracy",
        "model": MODEL,
        "scope": {
            "store_id": STORE,
            "dept_id": DEPARTMENT,
        },
        "source_paths": {
            name: path.relative_to(root).as_posix()
            for name, path in paths.items()
        },
        "selected_product_ids_in_rank_order": (
            manifest["id"].tolist()
        ),
        "selected_product_count": len(manifest),
        "feature_names": list(PREDICTOR_COLUMNS),
        "excluded_bookkeeping_and_label_columns": [
            "origin_day",
            "target_day",
            "target_sales",
        ],
        "feature_implementation": "src.prepare_xgb_features",
        "categorical_encoding": {
            "method": (
                "Dense OneHotEncoder; fitted independently per "
                "configuration and partition"
            ),
            "categorical_predictors": list(CATEGORICAL_PREDICTORS),
            "unknown_product_ids": "Rejected",
            "unseen_calendar_categories": (
                "Encoded as an all-zero indicator block and explicitly "
                "recorded in each fit's diagnostics."
            ),
            "numeric_scaling": "None",
            "matrix_dtype": "float32",
            "missing_prices": "Preserved as NaN; rows retained",
            "dense_matrix_reason": (
                "Preserve genuine zero values distinctly from missing values"
            ),
        },
        "historical_origin_schedule": {
            "minimum_origin": feature_module.MIN_ORIGIN,
            "algorithm": (
                "upper=cutoff-28; n=(upper-57)//7+1; "
                "integer anchors span 57..upper; interior index i receives "
                "(i%7)-3; endpoints unchanged; singleton schedule is (57,)."
            ),
            "spacing_note": (
                "Approximately weekly distribution; actual origin gaps "
                "may be shorter than seven days but are strictly positive."
            ),
            "selection_uses_demand_or_performance": False,
            "fit_schedules": (
                backtesting["diagnostics"]
                + [final_diagnostics]
            ),
        },
        "backtest_origins": list(BACKTEST_ORIGINS),
        "backtest_windows": [
            {
                "forecast_origin": origin,
                "fitting_label_cutoff": origin,
                "evaluation_first_day": origin + 1,
                "evaluation_last_day": origin + HORIZON,
            }
            for origin in BACKTEST_ORIGINS
        ],
        "model_configurations": CONFIGURATIONS,
        "selected_configuration": chosen,
        "selected_objective": CONFIGURATIONS[chosen]["objective"],
        "configuration_selection": {
            "metric": "Combined pooled product-day WAPE",
            "combined_metrics": backtesting["combined_metrics"],
            "tie_break": "A_squared_error",
            "undefined_wape_rule": (
                "Both undefined: retain A_squared_error"
            ),
            "validation_used_for_configuration_selection": False,
        },
        "final_fit": final_diagnostics,
        "chronological_windows": {
            "training": [1, TRAIN_END],
            "validation": [TRAIN_END + 1, VALIDATION_END],
            "excluded_final_test": [
                VALIDATION_END + 1,
                HISTORY_END,
            ],
            "forecast_horizon": HORIZON,
        },
        "metric_definitions": {
            "implementation": "src.evaluate",
            "mae": "mean(abs(predicted - actual))",
            "rmse": "sqrt(mean((predicted - actual)**2))",
            "wape_pct": (
                "sum(abs(predicted-actual))/sum(actual)*100"
            ),
            "undefined_wape": (
                "None internally, empty CSV field, null in JSON"
            ),
            "bias_units": (
                "mean(predicted - actual); positive means overforecast"
            ),
            "combined_backtests": (
                "calculate_metrics on all product-day pairs, not an "
                "average of fold percentages; aggregate-daily metrics "
                "are reported per fold."
            ),
        },
        "product_level_exceptions": {
            SEASONAL_MODEL: product_exceptions(
                products,
                seasonal_products,
            ),
            RIDGE_MODEL: product_exceptions(
                products,
                ridge_products,
            ),
        },
        "integrity_checks": integrity,
        "package_versions": {
            name: version(name)
            for name in (
                "pandas",
                "numpy",
                "scikit-learn",
                "xgboost",
            )
        },
        "python_version": sys.version,
        "data_availability_assumptions": [
            "Demand is observed through each fitting/forecast origin only.",
            "Both configurations freeze forecasts before each backtest's "
            "actuals are read.",
            "Validation actuals are read only after final forecasts "
            "are frozen.",
            "Calendar/event/SNAP fields are treated as known in advance.",
            "Historical price availability follows "
            "src.prepare_xgb_features: current-week prices require "
            "a confirmed week start.",
            "Test sales fields are never accessed, converted, retained, "
            "or evaluated.",
            "Input artifacts are assumed stable during execution.",
        ],
        "runtime_and_memory": {
            "shared_preparation": (
                "Fitting history and feature matrices are prepared once "
                "per backtest origin and reused across configurations."
            ),
            "independent_preprocessing": True,
            "demand_csv_passes": (
                2 * len(BACKTEST_ORIGINS) + 2
            ),
            "full_raw_sales_matrix_loaded": False,
            "training_matrices_exported": False,
        },
        "benchmark_interface_assumptions": {
            "required_prediction_columns": list(
                BASELINE_PREDICTION_COLUMNS
            ),
            "required_summary_columns": list(SUMMARY_COLUMNS),
            "model_names": [SEASONAL_MODEL, RIDGE_MODEL],
            "extra_prediction_columns": "Permitted",
            "undefined_wape_csv_representation": "Empty field",
        },
        "limitations": [
            "The selected high-volume products are not representative.",
            "Historical backtests condition on a cohort selected using "
            "training information through d_1857; they are not "
            "selection-independent historical population simulations.",
            "Overlapping supervised targets are dependent.",
            "The two configurations are predeclared, not proven optimal.",
            "Only three historical windows choose the objective.",
            "Missing prices do not establish product launch dates.",
            "Recorded sales may be constrained by unobserved inventory.",
            "Unseen calendar categories have no separately learned effect.",
            "Multithreaded floating-point operations may vary slightly "
            "across software builds/hardware despite fixed random state.",
            "Validation comparison does not retune or replace "
            "the selected objective.",
            "The prerequisite audit is not a source-file fingerprint; "
            "inputs are revalidated for relevant scopes.",
            "Individual output replacements are atomic; "
            "the six-file set is not.",
            "No final-test evaluation, SHAP analysis, "
            "or inventory optimization.",
        ],
    }

    outputs = publish_outputs(
        root,
        backtesting["metrics"],
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
            f"XGBoost experiment FAILED: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print(
            "Existing XGBoost outputs are not evidence "
            "of a successful current run.",
            file=sys.stderr,
        )
        return 1

    metadata = result["metadata"]
    print("XGBoost validation experiment: PASS")
    print(
        "Selected configuration: "
        f"{metadata['selected_configuration']}"
    )
    print(
        f"Selected objective: {metadata['selected_objective']}"
    )
    print(
        "Validation prediction rows: "
        f"{metadata['integrity_checks']['prediction_rows']}"
    )
    print("Configuration selection used historical backtests only.")
    print("No final-test sales were accessed or evaluated.")
    for path in result["output_paths"]:
        print(f"Output: {path}")
    print("Exit code: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())