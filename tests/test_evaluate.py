"""Focused Stage 3A tests using synthetic data only."""

import math
from decimal import Decimal

import numpy as np
import pytest

from src.evaluate import (
    calculate_metrics,
    evaluate_records,
    forecast_bias,
    mae,
    rmse,
    wape,
)


METRIC_FUNCTIONS = (mae, rmse, wape, forecast_bias, calculate_metrics)
METRIC_FIELDS = {
    "observations", "mae", "rmse", "wape_pct", "bias_units"
}


def synthetic_records():
    # A errors: +2, -2; B errors: -2, +2.
    # Daily errors cancel after aggregation, but pooled errors remain.
    return [
        {"id": "B", "day_number": 11, "actual": 30, "predicted": 32},
        {"id": "A", "day_number": 10, "actual": 10, "predicted": 12},
        {"id": "B", "day_number": 10, "actual": 10, "predicted": 8},
        {"id": "A", "day_number": 11, "actual": 10, "predicted": 8},
    ]


def test_perfect_predictions():
    result = calculate_metrics([1, 2, 3], [1, 2, 3])
    assert set(result) == METRIC_FIELDS
    assert result["observations"] == 3
    for name in ("mae", "rmse", "wape_pct", "bias_units"):
        assert result[name] == pytest.approx(0.0)


def test_hand_calculated_metrics():
    # Errors = [2, -2, 6]; absolute sum = 10; squared sum = 44.
    actual = [10, 20, 30]
    predicted = [12, 18, 36]
    result = calculate_metrics(actual, predicted)

    assert result["observations"] == 3
    assert result["mae"] == pytest.approx(10 / 3)
    assert result["rmse"] == pytest.approx(math.sqrt(44 / 3))
    assert result["wape_pct"] == pytest.approx(100 * 10 / 60)
    assert result["bias_units"] == pytest.approx(2)

    assert mae(actual, predicted) == pytest.approx(result["mae"])
    assert rmse(actual, predicted) == pytest.approx(result["rmse"])
    assert wape(actual, predicted) == pytest.approx(result["wape_pct"])
    assert forecast_bias(actual, predicted) == pytest.approx(result["bias_units"])


def test_positive_bias_means_overforecasting():
    assert forecast_bias([2, 4], [5, 7]) == pytest.approx(3)


def test_negative_bias_means_underforecasting():
    assert forecast_bias([5, 7], [2, 4]) == pytest.approx(-3)


def test_zero_actual_demand_has_undefined_wape():
    assert wape([0, 0], [1, 3]) is None
    assert calculate_metrics([0, 0], [1, 3])["wape_pct"] is None


def test_other_metrics_defined_when_actual_total_is_zero():
    result = calculate_metrics([0, 0], [1, 3])
    assert result["observations"] == 2
    assert result["mae"] == pytest.approx(2)
    assert result["rmse"] == pytest.approx(math.sqrt(5))
    assert result["bias_units"] == pytest.approx(2)


def test_all_zero_values():
    result = calculate_metrics([0, 0], [0, 0])
    assert result["wape_pct"] is None
    assert result["mae"] == pytest.approx(0)
    assert result["rmse"] == pytest.approx(0)
    assert result["bias_units"] == pytest.approx(0)


@pytest.mark.parametrize("function", METRIC_FUNCTIONS)
def test_unequal_lengths_rejected(function):
    with pytest.raises(ValueError, match="equal lengths"):
        function([1, 2], [1])


@pytest.mark.parametrize("function", METRIC_FUNCTIONS)
@pytest.mark.parametrize(
    "actual,predicted",
    [([], []), ([], [1]), ([1], [])],
)
def test_empty_inputs_rejected(function, actual, predicted):
    with pytest.raises(ValueError, match="empty"):
        function(actual, predicted)


@pytest.mark.parametrize("function", METRIC_FUNCTIONS)
@pytest.mark.parametrize(
    "bad",
    [
        None,
        float("nan"),
        float("inf"),
        float("-inf"),
        "2",
        "invalid",
        complex(1, 0),
        Decimal("NaN"),
        np.nan,
    ],
)
@pytest.mark.parametrize("side", ["actual", "predicted"])
def test_invalid_values_rejected(function, bad, side):
    actual = [bad] if side == "actual" else [1]
    predicted = [bad] if side == "predicted" else [1]
    with pytest.raises(ValueError):
        function(actual, predicted)


@pytest.mark.parametrize("function", METRIC_FUNCTIONS)
@pytest.mark.parametrize("side", ["actual", "predicted"])
def test_negative_values_rejected(function, side):
    actual = [-1] if side == "actual" else [1]
    predicted = [-1] if side == "predicted" else [1]
    with pytest.raises(ValueError, match="nonnegative"):
        function(actual, predicted)


@pytest.mark.parametrize("function", METRIC_FUNCTIONS)
@pytest.mark.parametrize("bad", [True, False, np.bool_(True)])
@pytest.mark.parametrize("side", ["actual", "predicted"])
def test_booleans_rejected(function, bad, side):
    actual = [bad] if side == "actual" else [1]
    predicted = [bad] if side == "predicted" else [1]
    with pytest.raises(ValueError, match="booleans"):
        function(actual, predicted)


@pytest.mark.parametrize(
    "malformed",
    [
        None,
        12,
        "123",
        {"a": 1},
        {1, 2},
        [[1, 2]],
        np.array([[1, 2]]),
        np.array(1),
    ],
)
def test_malformed_sequences_rejected(malformed):
    with pytest.raises(ValueError):
        calculate_metrics(malformed, [1])


def test_numpy_numeric_arrays_accepted():
    result = calculate_metrics(
        np.array([1, 2, 3], dtype=np.int64),
        np.array([2, 2, 4], dtype=np.float64),
    )
    assert result["observations"] == 3
    assert result["mae"] == pytest.approx(2 / 3)
    assert result["rmse"] == pytest.approx(math.sqrt(2 / 3))
    assert result["wape_pct"] == pytest.approx(100 * 2 / 6)
    assert result["bias_units"] == pytest.approx(2 / 3)


def test_decimal_values_accepted():
    result = calculate_metrics(
        [Decimal("1.5"), Decimal("2.5")],
        [Decimal("2.0"), Decimal("3.0")],
    )
    assert result["mae"] == pytest.approx(0.5)
    assert result["bias_units"] == pytest.approx(0.5)


def test_large_finite_errors_do_not_overflow_when_squared():
    result = calculate_metrics([0, 0], [1e200, 1e200])
    assert result["rmse"] == pytest.approx(1e200)
    assert result["mae"] == pytest.approx(1e200)
    assert result["wape_pct"] is None


def test_per_product_metrics_and_deterministic_order():
    result = evaluate_records(synthetic_records(), "per_product")
    assert result["level"] == "per_product"
    assert list(result["products"]) == ["A", "B"]

    for metrics in result["products"].values():
        assert set(metrics) == METRIC_FIELDS
        assert metrics["observations"] == 2
        assert metrics["mae"] == pytest.approx(2)
        assert metrics["rmse"] == pytest.approx(2)
        assert metrics["bias_units"] == pytest.approx(0)

    assert result["products"]["A"]["wape_pct"] == pytest.approx(20)
    assert result["products"]["B"]["wape_pct"] == pytest.approx(10)


def test_pooled_metrics_use_individual_product_day_errors():
    result = evaluate_records(synthetic_records(), "pooled_product_day")
    assert result["observations"] == 4
    assert result["mae"] == pytest.approx(2)
    assert result["rmse"] == pytest.approx(2)
    assert result["wape_pct"] == pytest.approx(100 * 8 / 60)
    assert result["bias_units"] == pytest.approx(0)

    per_product = evaluate_records(synthetic_records(), "per_product")["products"]
    average_product_wape = sum(
        metrics["wape_pct"] for metrics in per_product.values()
    ) / len(per_product)
    assert result["wape_pct"] != pytest.approx(average_product_wape)


def test_pooled_rmse_is_not_average_product_rmse():
    records = [
        {"id": "A", "day_number": 1, "actual": 10, "predicted": 10},
        {"id": "A", "day_number": 2, "actual": 10, "predicted": 10},
        {"id": "B", "day_number": 1, "actual": 10, "predicted": 14},
        {"id": "B", "day_number": 2, "actual": 10, "predicted": 14},
    ]
    pooled = evaluate_records(records, "pooled_product_day")
    products = evaluate_records(records, "per_product")["products"]
    average_rmse = sum(item["rmse"] for item in products.values()) / 2

    assert pooled["rmse"] == pytest.approx(math.sqrt(8))
    assert average_rmse == pytest.approx(2)
    assert pooled["rmse"] != pytest.approx(average_rmse)


def test_aggregate_daily_metrics_use_summed_daily_totals():
    result = evaluate_records(synthetic_records(), "aggregate_daily")
    # Daily actual and predicted totals both equal [20, 40].
    assert result["observations"] == 2
    for name in ("mae", "rmse", "wape_pct", "bias_units"):
        assert result[name] == pytest.approx(0)


def test_aggregate_daily_nonzero_hand_calculated_metrics():
    records = [
        {"id": "A", "day_number": 1, "actual": 1, "predicted": 2},
        {"id": "B", "day_number": 1, "actual": 3, "predicted": 5},
        {"id": "A", "day_number": 2, "actual": 2, "predicted": 1},
        {"id": "B", "day_number": 2, "actual": 4, "predicted": 3},
    ]
    # Daily actual = [4, 6], predicted = [7, 4], errors = [3, -2].
    result = evaluate_records(records, "aggregate_daily")
    assert result["observations"] == 2
    assert result["mae"] == pytest.approx(2.5)
    assert result["rmse"] == pytest.approx(math.sqrt(6.5))
    assert result["wape_pct"] == pytest.approx(50)
    assert result["bias_units"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "level", ["per_product", "pooled_product_day", "aggregate_daily"]
)
def test_incomplete_product_day_coverage_rejected(level):
    records = synthetic_records()[:-1]
    with pytest.raises(ValueError, match="coverage"):
        evaluate_records(records, level)


def test_shared_internal_day_gap_rejected():
    records = [
        {"id": "A", "day_number": 1, "actual": 1, "predicted": 1},
        {"id": "A", "day_number": 3, "actual": 1, "predicted": 1},
        {"id": "B", "day_number": 1, "actual": 1, "predicted": 1},
        {"id": "B", "day_number": 3, "actual": 1, "predicted": 1},
    ]
    with pytest.raises(ValueError, match="consecutive"):
        evaluate_records(records, "aggregate_daily")


@pytest.mark.parametrize(
    "level", ["per_product", "pooled_product_day", "aggregate_daily"]
)
def test_duplicate_product_day_records_rejected(level):
    records = synthetic_records()
    records.append(dict(records[0]))
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_records(records, level)


@pytest.mark.parametrize(
    "level", ["per_product", "pooled_product_day", "aggregate_daily"]
)
def test_undefined_wape_consistent_across_aggregation_levels(level):
    records = [
        {"id": "A", "day_number": 1, "actual": 0, "predicted": 1},
        {"id": "A", "day_number": 2, "actual": 0, "predicted": 1},
        {"id": "B", "day_number": 1, "actual": 0, "predicted": 1},
        {"id": "B", "day_number": 2, "actual": 0, "predicted": 1},
    ]
    result = evaluate_records(records, level)
    metrics_list = (
        list(result["products"].values())
        if level == "per_product"
        else [result]
    )
    expected_error = 2 if level == "aggregate_daily" else 1
    for metrics in metrics_list:
        assert metrics["wape_pct"] is None
        assert metrics["mae"] == pytest.approx(expected_error)
        assert metrics["rmse"] == pytest.approx(expected_error)
        assert metrics["bias_units"] == pytest.approx(expected_error)


@pytest.mark.parametrize(
    "bad_record",
    [
        None,
        {},
        {"id": "", "day_number": 1, "actual": 1, "predicted": 1},
        {"id": "A", "day_number": True, "actual": 1, "predicted": 1},
        {"id": "A", "day_number": 1.0, "actual": 1, "predicted": 1},
        {"id": "A", "day_number": 0, "actual": 1, "predicted": 1},
        {"id": "A", "day_number": 1, "actual": -1, "predicted": 1},
        {"id": "A", "day_number": 1, "actual": 1, "predicted": "1"},
        {"id": "A", "day_number": 1, "actual": None, "predicted": 1},
        {"id": "A", "day_number": 1, "actual": 1, "predicted": float("inf")},
        {"id": "A", "day_number": 1, "actual": float("nan"), "predicted": 1},
        {"id": "A", "day_number": 1, "actual": True, "predicted": 1},
    ],
)
def test_invalid_records_rejected(bad_record):
    with pytest.raises(ValueError):
        evaluate_records([bad_record], "aggregate_daily")


def test_empty_records_rejected():
    with pytest.raises(ValueError, match="empty"):
        evaluate_records([], "aggregate_daily")


def test_unknown_level_rejected():
    with pytest.raises(ValueError, match="level"):
        evaluate_records(synthetic_records(), "unknown")


@pytest.mark.parametrize(
    "level", ["per_product", "pooled_product_day", "aggregate_daily"]
)
def test_record_order_does_not_change_results(level):
    records = synthetic_records()
    assert evaluate_records(records, level) == evaluate_records(
        list(reversed(records)), level
    )


def test_aggregate_daily_overflow_has_clear_error():
    records = [
        {"id": "A", "day_number": 1, "actual": 1e308, "predicted": 1},
        {"id": "B", "day_number": 1, "actual": 1e308, "predicted": 1},
    ]
    with pytest.raises(ValueError, match="finite float"):
        evaluate_records(records, "aggregate_daily")