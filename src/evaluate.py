"""Forecast evaluation metrics and synthetic-data-compatible aggregation.

No files are read or written.

evaluate_records() return formats:

    per_product:
        {
            "level": "per_product",
            "products": {
                "<series id>": {
                    "observations": int,
                    "mae": float,
                    "rmse": float,
                    "wape_pct": float | None,
                    "bias_units": float,
                }
            },
        }

    pooled_product_day / aggregate_daily:
        {
            "level": "<requested level>",
            "observations": int,
            "mae": float,
            "rmse": float,
            "wape_pct": float | None,
            "bias_units": float,
        }

Every aggregation level requires the same consecutive day-number coverage
for every product. This detects internal gaps and inconsistent coverage.
Without an expected external date range, common missing boundary days cannot
be inferred. Stage 3B must separately verify its expected evaluation horizon.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from decimal import Decimal
from numbers import Integral, Real
from typing import Any

import numpy as np


LEVELS = ("per_product", "pooled_product_day", "aggregate_daily")
REQUIRED_RECORD_FIELDS = ("id", "day_number", "actual", "predicted")


def _numeric_value(value: Any, context: str) -> float:
    """Accept real numeric values, excluding booleans and numeric strings."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{context}: booleans are not numeric observations.")

    if not isinstance(value, (Real, Decimal)):
        raise ValueError(
            f"{context}: expected a real numeric value; "
            f"received {type(value).__name__}."
        )

    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{context}: value cannot be represented as a finite float."
        ) from exc

    if not math.isfinite(converted):
        raise ValueError(f"{context}: values must be finite, not NaN or infinity.")
    if converted < 0:
        raise ValueError(f"{context}: values must be nonnegative.")
    return converted


def _numeric_sequence(values: Any, name: str) -> list[float]:
    """Validate a one-dimensional ordered numeric iterable."""
    if isinstance(values, (str, bytes, bytearray, Mapping, set, frozenset)):
        raise ValueError(f"{name}: expected an ordered numeric sequence.")

    if isinstance(values, np.ndarray) and values.ndim != 1:
        raise ValueError(f"{name}: expected a one-dimensional numeric sequence.")

    try:
        iterator = iter(values)
    except TypeError as exc:
        raise ValueError(f"{name}: expected an iterable numeric sequence.") from exc

    result = []
    try:
        for index, value in enumerate(iterator):
            result.append(_numeric_value(value, f"{name}[{index}]"))
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"{name}: numeric sequence could not be read.") from exc

    if not result:
        raise ValueError(f"{name}: sequence must not be empty.")
    return result


def _validated_pair(
    actual: Any, predicted: Any
) -> tuple[list[float], list[float]]:
    actual_values = _numeric_sequence(actual, "actual")
    predicted_values = _numeric_sequence(predicted, "predicted")
    if len(actual_values) != len(predicted_values):
        raise ValueError("actual and predicted must have equal lengths.")
    return actual_values, predicted_values


def _checked_float(value: float, context: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{context}: result exceeds finite float precision.")
    return float(value)


def _calculate_validated(
    actual: list[float], predicted: list[float]
) -> dict[str, Any]:
    """Calculate metrics after validating both inputs once."""
    observations = len(actual)
    errors = [
        forecast - observed
        for observed, forecast in zip(actual, predicted)
    ]
    absolute_errors = [abs(error) for error in errors]

    # Divide before summing to avoid unnecessary overflow in a large total.
    mae_value = math.fsum(error / observations for error in absolute_errors)
    bias_value = math.fsum(error / observations for error in errors)

    # Scaling avoids overflow from squaring otherwise valid finite errors.
    error_scale = max(absolute_errors)
    if error_scale == 0:
        rmse_value = 0.0
    else:
        mean_scaled_square = math.fsum(
            (error / error_scale) ** 2 for error in errors
        ) / observations
        rmse_value = error_scale * math.sqrt(mean_scaled_square)

    actual_scale = max(actual)
    if actual_scale == 0:
        wape_value = None
    elif error_scale == 0:
        wape_value = 0.0
    else:
        # Normalize the two sums independently to reduce overflow/underflow.
        scaled_actual_total = math.fsum(
            value / actual_scale for value in actual
        )
        scaled_error_total = math.fsum(
            value / error_scale for value in absolute_errors
        )
        wape_value = _checked_float(
            (scaled_error_total / scaled_actual_total)
            * (error_scale / actual_scale)
            * 100.0,
            "WAPE",
        )

    return {
        "observations": observations,
        "mae": _checked_float(mae_value, "MAE"),
        "rmse": _checked_float(rmse_value, "RMSE"),
        "wape_pct": wape_value,
        "bias_units": _checked_float(bias_value, "Forecast bias"),
    }


def calculate_metrics(actual: Any, predicted: Any) -> dict[str, Any]:
    """Return observation count, MAE, RMSE, WAPE percentage, and signed bias.

    Inputs must be nonempty, equal-length, one-dimensional numeric sequences.
    Values must be finite and nonnegative. Booleans and strings are rejected.

    WAPE is None when total actual demand is zero. Positive bias means
    overforecasting; negative bias means underforecasting.
    """
    actual_values, predicted_values = _validated_pair(actual, predicted)
    return _calculate_validated(actual_values, predicted_values)


def mae(actual: Any, predicted: Any) -> float:
    """Mean absolute error, in the same units as the observations."""
    actual_values, predicted_values = _validated_pair(actual, predicted)
    count = len(actual_values)
    return _checked_float(
        math.fsum(
            abs(forecast - observed) / count
            for observed, forecast in zip(actual_values, predicted_values)
        ),
        "MAE",
    )


def rmse(actual: Any, predicted: Any) -> float:
    """Root mean squared error, in observation units."""
    actual_values, predicted_values = _validated_pair(actual, predicted)
    errors = [
        forecast - observed
        for observed, forecast in zip(actual_values, predicted_values)
    ]
    scale = max(abs(error) for error in errors)
    if scale == 0:
        return 0.0
    return _checked_float(
        scale * math.sqrt(
            math.fsum((error / scale) ** 2 for error in errors) / len(errors)
        ),
        "RMSE",
    )


def wape(actual: Any, predicted: Any) -> float | None:
    """Absolute error / actual demand * 100; None for zero actual demand."""
    return calculate_metrics(actual, predicted)["wape_pct"]


def forecast_bias(actual: Any, predicted: Any) -> float:
    """Mean(predicted - actual): positive values indicate overforecasting."""
    actual_values, predicted_values = _validated_pair(actual, predicted)
    count = len(actual_values)
    return _checked_float(
        math.fsum(
            (forecast - observed) / count
            for observed, forecast in zip(actual_values, predicted_values)
        ),
        "Forecast bias",
    )


def _validated_records(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, dict[int, tuple[float, float]]]:
    if isinstance(records, (str, bytes, bytearray, Mapping, set, frozenset)):
        raise ValueError("records must be an ordered iterable of record mappings.")

    try:
        iterator = iter(records)
    except TypeError as exc:
        raise ValueError("records must be an iterable of record mappings.") from exc

    products: dict[str, dict[int, tuple[float, float]]] = {}
    try:
        for index, record in enumerate(iterator):
            if not isinstance(record, Mapping):
                raise ValueError(f"Record {index} must be a mapping.")

            missing = [
                field for field in REQUIRED_RECORD_FIELDS if field not in record
            ]
            if missing:
                raise ValueError(f"Record {index} is missing fields: {missing}.")

            series_id = record["id"]
            if not isinstance(series_id, str) or not series_id.strip():
                raise ValueError(f"Record {index}: id must be a nonempty string.")

            day = record["day_number"]
            if (
                isinstance(day, (bool, np.bool_))
                or not isinstance(day, Integral)
                or day < 1
            ):
                raise ValueError(
                    f"Record {index}: day_number must be a positive integer."
                )
            day = int(day)

            product_days = products.setdefault(series_id, {})
            if day in product_days:
                raise ValueError(
                    f"Duplicate (id, day_number) record: ({series_id!r}, {day})."
                )

            product_days[day] = (
                _numeric_value(record["actual"], f"Record {index} actual"),
                _numeric_value(record["predicted"], f"Record {index} predicted"),
            )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("records could not be interpreted as record mappings.") from exc

    if not products:
        raise ValueError("records must not be empty.")

    ordered_ids = sorted(products)
    reference_days = sorted(products[ordered_ids[0]])
    if any(
        current != previous + 1
        for previous, current in zip(reference_days, reference_days[1:])
    ):
        raise ValueError("Incomplete coverage: day numbers must be consecutive.")

    for series_id in ordered_ids:
        if sorted(products[series_id]) != reference_days:
            raise ValueError(
                "Inconsistent or incomplete product-day coverage for "
                f"{series_id!r}; every product must have the same days."
            )
    return products


def _finite_sum(values: Iterable[float], context: str) -> float:
    try:
        result = math.fsum(values)
    except OverflowError as exc:
        raise ValueError(f"{context}: daily total exceeds finite float precision.") from exc
    return _checked_float(result, context)


def evaluate_records(
    records: Iterable[Mapping[str, Any]], level: str
) -> dict[str, Any]:
    """Evaluate unique product-day records at a specified aggregation level.

    Records contain id, day_number, actual, and predicted.

    All products must cover the same consecutive days. No product count,
    forecast horizon, or real dataset dimensions are hardcoded.

    Pooled metrics use individual product-day errors. Aggregate daily metrics
    first sum actual and predicted values separately for each day.
    """
    if not isinstance(level, str) or level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}.")

    products = _validated_records(records)
    ids = sorted(products)
    days = sorted(products[ids[0]])

    if level == "per_product":
        results = {}
        for series_id in ids:
            pairs = [products[series_id][day] for day in days]
            results[series_id] = _calculate_validated(
                [pair[0] for pair in pairs],
                [pair[1] for pair in pairs],
            )
        return {"level": level, "products": results}

    if level == "pooled_product_day":
        pairs = [
            products[series_id][day]
            for series_id in ids
            for day in days
        ]
        metrics = _calculate_validated(
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
        )
        return {"level": level, **metrics}

    daily_actual = []
    daily_predicted = []
    for day in days:
        daily_actual.append(
            _finite_sum(
                (products[series_id][day][0] for series_id in ids),
                f"Day {day} actual",
            )
        )
        daily_predicted.append(
            _finite_sum(
                (products[series_id][day][1] for series_id in ids),
                f"Day {day} predicted",
            )
        )
    return {
        "level": level,
        **_calculate_validated(daily_actual, daily_predicted),
    }