"""Synthetic Stage 3B tests. No real M5 files are required."""

import csv
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import src.forecast_baselines as baselines
from src.evaluate import calculate_metrics, evaluate_records


@pytest.mark.parametrize("last", [0, 12, 37])
def test_naive_repeats_final_training_value(last):
    result = baselines.naive_forecast([2, 4, last])
    assert result == [last] * 28
    assert all(type(value) is int for value in result)


def test_seasonal_naive_repeats_final_seven_values():
    last_week = [5, 7, 9, 6, 8, 10, 12]
    result = baselines.seasonal_naive_forecast([99, 98] + last_week)
    assert result == last_week * 4
    assert all(type(value) is int for value in result)


@pytest.mark.parametrize(
    "function",
    [baselines.naive_forecast, baselines.seasonal_naive_forecast],
)
def test_forecasts_have_exactly_28_values(function):
    assert len(function(list(range(14)))) == 28


@pytest.mark.parametrize(
    "function",
    [baselines.naive_forecast, baselines.seasonal_naive_forecast],
)
def test_zero_sales_are_valid(function):
    assert function([0] * 7) == [0] * 28


@pytest.mark.parametrize(
    "function",
    [baselines.naive_forecast, baselines.seasonal_naive_forecast],
)
@pytest.mark.parametrize(
    "bad", [None, -1, 1.5, "2", True, np.bool_(True), float("nan"), float("inf")]
)
def test_invalid_history_values_rejected(function, bad):
    history = [1] * 7
    history[0] = bad
    with pytest.raises(ValueError):
        function(history)


@pytest.mark.parametrize(
    "function,history",
    [
        (baselines.naive_forecast, []),
        (baselines.seasonal_naive_forecast, []),
        (baselines.seasonal_naive_forecast, [1] * 6),
    ],
)
def test_insufficient_history_rejected(function, history):
    with pytest.raises(ValueError, match="Insufficient"):
        function(history)


@pytest.mark.parametrize("bad", [None, 4, "123", [[1, 2]], np.array([[1, 2]])])
def test_malformed_history_rejected(bad):
    with pytest.raises(ValueError):
        baselines.naive_forecast(bad)


def test_numpy_integer_training_array_accepted():
    assert baselines.naive_forecast(np.array([1, 2, 3])) == [3] * 28


def test_fixed_origin_weekly_forecast_does_not_advance_to_actuals():
    history = [1, 2, 3, 4, 5, 6, 7]
    forecast = baselines.seasonal_naive_forecast(history)
    assert forecast[:7] == history
    assert forecast[7:14] == history
    assert forecast[14:21] == history
    assert forecast[21:] == history
    assert baselines.naive_forecast(history) == [7] * 28


def write_sales(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(baselines.SALES_COLUMNS)
        writer.writerows(rows)


def minimal_manifest():
    return pd.DataFrame(
        [{"id": "A", "item_id": "ITEM_A", "store_id": "CA_1", "dept_id": "FOODS_3"}]
    )


def period_rows(first, last, split, sales=3):
    start = date(2011, 1, 1)
    return [
        [
            "A", "ITEM_A", "CA_1", "FOODS_3", f"d_{day}", day,
            (start + timedelta(days=day - 1)).isoformat(), sales, split,
        ]
        for day in range(first, last + 1)
    ]


def test_test_sales_excluded_from_loading_path(tmp_path, monkeypatch):
    path = tmp_path / "sales.csv"
    rows = period_rows(1, 1857, "train")
    rows += period_rows(1858, 1885, "validation")
    # Deliberately malformed test demand must never reach numeric validation.
    rows += period_rows(1886, 1913, "test", sales="UNTOUCHED_TEST_DEMAND")
    write_sales(path, rows)

    original = baselines.unit_value
    parsed = []

    def spy(value, context, allow_text=False):
        parsed.append(value)
        assert value != "UNTOUCHED_TEST_DEMAND"
        return original(value, context, allow_text)

    monkeypatch.setattr(baselines, "unit_value", spy)
    train = baselines.load_period(path, minimal_manifest(), "train")
    validation = baselines.load_period(path, minimal_manifest(), "validation")
    assert len(train) == 1857
    assert len(validation) == 28
    assert len(parsed) == 1885
    assert train["split"].eq("train").all()
    assert validation["split"].eq("validation").all()
    assert validation["day_number"].max() == 1885


@pytest.mark.parametrize("problem", ["duplicate", "missing", "unordered"])
def test_loader_rejects_invalid_product_day_coverage(tmp_path, problem):
    rows = period_rows(1858, 1885, "validation")
    if problem == "duplicate":
        rows.append(list(rows[0]))
    elif problem == "missing":
        rows.pop()
    else:
        rows[0], rows[1] = rows[1], rows[0]
    path = tmp_path / "sales.csv"
    write_sales(path, rows)
    with pytest.raises(ValueError):
        baselines.load_period(path, minimal_manifest(), "validation")


def test_loader_rejects_inconsistent_split_boundary(tmp_path):
    rows = period_rows(1858, 1885, "validation")
    rows[0][-1] = "train"
    path = tmp_path / "sales.csv"
    write_sales(path, rows)
    with pytest.raises(ValueError, match="split"):
        baselines.load_period(path, minimal_manifest(), "validation")


def synthetic_forecasts():
    records = []
    for model in baselines.MODELS:
        for series_id in ("A", "B"):
            for horizon in range(1, 29):
                records.append(
                    {
                        "id": series_id,
                        "item_id": f"ITEM_{series_id}",
                        "model": model,
                        "forecast_origin_day": "d_1857",
                        "horizon": horizon,
                        "d": f"d_{1857 + horizon}",
                        "predicted_sales": 12 if series_id == "A" else 8,
                    }
                )
    return pd.DataFrame(records, columns=baselines.FORECAST_COLUMNS, dtype=object)


def synthetic_validation(actual_a=10, actual_b=10):
    records = []
    for series_id, actual in (("A", actual_a), ("B", actual_b)):
        for horizon in range(1, 29):
            records.append(
                {
                    "id": series_id,
                    "item_id": f"ITEM_{series_id}",
                    "day_number": 1857 + horizon,
                    "date": pd.Timestamp("2020-01-01") + np.timedelta64(horizon - 1, "D"),
                    "sales": actual,
                }
            )
    return pd.DataFrame(records, dtype=object)


def test_alignment_maps_horizons_and_validation_changes_only_actuals():
    forecasts = synthetic_forecasts()
    original = forecasts.copy(deep=True)
    first = baselines.align_forecasts(forecasts, synthetic_validation(10, 10))
    changed = baselines.align_forecasts(forecasts, synthetic_validation(100, 200))

    assert forecasts.equals(original)
    assert first["predicted_sales"].tolist() == changed["predicted_sales"].tolist()
    assert first["actual_sales"].tolist() != changed["actual_sales"].tolist()
    assert len(first) == 2 * 2 * 28
    assert tuple(first.columns) == baselines.PREDICTION_COLUMNS
    assert not first.duplicated(["id", "model", "horizon"]).any()
    for row in first.itertuples(index=False):
        assert row.d == f"d_{1857 + row.horizon}"
        assert row.forecast_origin_day == "d_1857"
    assert set(first["d"]) == {f"d_{day}" for day in range(1858, 1886)}


def test_missing_validation_alignment_rejected():
    with pytest.raises(ValueError, match="Missing"):
        baselines.align_forecasts(
            synthetic_forecasts(), synthetic_validation().iloc[1:]
        )


def test_duplicate_validation_alignment_rejected():
    validation = synthetic_validation()
    validation = pd.concat([validation, validation.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        baselines.align_forecasts(synthetic_forecasts(), validation)


def test_evaluation_delegates_to_existing_metric_functions(monkeypatch):
    predictions = baselines.align_forecasts(
        synthetic_forecasts(), synthetic_validation()
    )
    metric_calls = []
    aggregation_calls = []

    def metric_spy(actual, predicted):
        metric_calls.append(len(actual))
        return calculate_metrics(actual, predicted)

    def aggregation_spy(records, level):
        aggregation_calls.append((level, len(records)))
        return evaluate_records(records, level)

    monkeypatch.setattr(baselines, "calculate_metrics", metric_spy)
    monkeypatch.setattr(baselines, "evaluate_records", aggregation_spy)
    products, summary = baselines.evaluate_predictions(predictions)

    assert metric_calls == [28] * 4
    assert aggregation_calls == [
        ("pooled_product_day", 56),
        ("aggregate_daily", 56),
        ("pooled_product_day", 56),
        ("aggregate_daily", 56),
    ]
    assert tuple(products.columns) == baselines.PRODUCT_COLUMNS
    assert tuple(summary.columns) == baselines.SUMMARY_COLUMNS
    assert len(products) == 4
    assert len(summary) == 4
    for model in baselines.MODELS:
        pooled = summary.loc[
            summary["model"].eq(model)
            & summary["evaluation_level"].eq("pooled_product_day")
        ].iloc[0]
        aggregate = summary.loc[
            summary["model"].eq(model)
            & summary["evaluation_level"].eq("aggregate_daily")
        ].iloc[0]
        assert pooled["observations"] == 56
        assert pooled["mae"] == pytest.approx(2)
        assert pooled["rmse"] == pytest.approx(2)
        assert pooled["wape_pct"] == pytest.approx(20)
        assert aggregate["observations"] == 28
        assert aggregate["mae"] == pytest.approx(0)
        assert aggregate["rmse"] == pytest.approx(0)


def build_synthetic_project(root):
    processed = root / "data/processed"
    processed.mkdir(parents=True)
    start = date(2011, 1, 1)
    manifest_rows = []
    selected_ids = []

    with (processed / "selected_sales.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(baselines.SALES_COLUMNS)
        for index in range(20):
            series_id = f"SERIES_{index:02d}"
            item_id = f"ITEM_{index:02d}"
            selected_ids.append(series_id)
            history = [index + 1] * 1850 + [1, 2, 3, 4, 5, 6, 7]
            manifest_rows.append(
                [
                    index + 1, series_id, item_id, "CA_1", "FOODS_3",
                    sum(history), 1857, 0.0,
                ]
            )
            for day in range(1, 1914):
                if day <= 1857:
                    sales = history[day - 1]
                    split = "train"
                elif day <= 1885:
                    sales = index + 10
                    split = "validation"
                else:
                    sales = "DO_NOT_PARSE_TEST_SALES"
                    split = "test"
                writer.writerow(
                    [
                        series_id, item_id, "CA_1", "FOODS_3",
                        f"d_{day}", day,
                        (start + timedelta(days=day - 1)).isoformat(),
                        sales, split,
                    ]
                )

    with (processed / "selected_products.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(baselines.MANIFEST_COLUMNS)
        writer.writerows(manifest_rows)

    (processed / "selection_metadata.json").write_text(
        json.dumps(
            {
                "actual_selected_series_count": 20,
                "selected_series_ids_in_rank_order": selected_ids,
                "store_id": "CA_1",
                "dept_id": "FOODS_3",
            }
        ),
        encoding="utf-8",
    )


def test_complete_synthetic_pipeline_outputs_and_input_immutability(tmp_path):
    build_synthetic_project(tmp_path)
    input_paths = sorted((tmp_path / "data/processed").iterdir())
    original_bytes = {path: path.read_bytes() for path in input_paths}

    result = baselines.run_pipeline(tmp_path)
    assert len(result["output_paths"]) == 5
    assert all(path.is_file() and path.stat().st_size > 0 for path in result["output_paths"])
    assert all(path.read_bytes() == original_bytes[path] for path in input_paths)

    evaluation = tmp_path / "outputs/evaluation"
    predictions = pd.read_csv(evaluation / "baseline_predictions.csv")
    products = pd.read_csv(evaluation / "baseline_metrics_by_product.csv")
    summary = pd.read_csv(evaluation / "baseline_metrics_summary.csv")

    assert tuple(predictions.columns) == baselines.PREDICTION_COLUMNS
    assert tuple(products.columns) == baselines.PRODUCT_COLUMNS
    assert tuple(summary.columns) == baselines.SUMMARY_COLUMNS
    assert len(predictions) == 1120
    assert len(products) == 40
    assert len(summary) == 4
    assert predictions.groupby(["id", "model"]).size().eq(28).all()
    assert not predictions.duplicated(["id", "model", "horizon"]).any()
    assert predictions["forecast_origin_day"].eq("d_1857").all()
    assert products["observations"].eq(28).all()
    assert summary.loc[
        summary["evaluation_level"].eq("pooled_product_day"), "observations"
    ].eq(560).all()
    assert summary.loc[
        summary["evaluation_level"].eq("aggregate_daily"), "observations"
    ].eq(28).all()

    metadata = json.loads(
        (evaluation / "baseline_run_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["integrity_checks"]["status"] == "PASS"
    assert metadata["no_test_demand_access"]["test_demand_retained"] is False
    assert metadata["no_test_demand_access"]["test_demand_evaluated"] is False
    chart = tmp_path / "outputs/figures/baseline_forecast_comparison.png"
    assert chart.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_pipeline_generates_forecasts_before_loading_validation(tmp_path, monkeypatch):
    build_synthetic_project(tmp_path)
    original_load = baselines.load_period
    original_generate = baselines.generate_forecasts
    events = []

    def load_spy(path, manifest, period):
        if period == "validation":
            assert events == ["train_loaded", "forecasts_generated"]
        result = original_load(path, manifest, period)
        events.append(f"{period}_loaded")
        return result

    def generate_spy(training, manifest):
        assert events == ["train_loaded"]
        result = original_generate(training, manifest)
        events.append("forecasts_generated")
        return result

    monkeypatch.setattr(baselines, "load_period", load_spy)
    monkeypatch.setattr(baselines, "generate_forecasts", generate_spy)
    baselines.run_pipeline(tmp_path)
    assert events == ["train_loaded", "forecasts_generated", "validation_loaded"]


def test_manifest_scope_requires_exactly_20_series(tmp_path):
    build_synthetic_project(tmp_path)
    path = tmp_path / "data/processed/selected_products.csv"
    manifest = pd.read_csv(path)
    manifest.iloc[:-1].to_csv(path, index=False)
    with pytest.raises(ValueError, match="exactly 20"):
        baselines.load_manifest_and_metadata(tmp_path)