"""Synthetic Stage 4C-B tests; no real M5 files are required."""

import csv
import hashlib
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

import src.forecast_xgboost as experiment
from src.evaluate import calculate_metrics, evaluate_records


def calendar_mapping():
    start = date(2011, 1, 3)
    return {
        day: {
            "date": start + timedelta(days=day - 1),
            "wm_yr_wk": 1000 + (day - 1) // 7,
            "snap_CA": int(day % 2 == 0),
            "event_type_1": "None",
            "event_type_2": "None",
            "has_event": 0,
        }
        for day in range(1, 1886)
    }


def manifest_for(count=2):
    return pd.DataFrame([
        {
            "selection_rank": index + 1,
            "id": f"SERIES_{index:02d}",
            "item_id": f"ITEM_{index:02d}",
            "store_id": "CA_1",
            "dept_id": "FOODS_3",
        }
        for index in range(count)
    ])


class SyntheticRegressor(RegressorMixin, BaseEstimator):
    """Deterministic test double; production uses XGBRegressor."""

    def __init__(self, offset=0.0):
        self.offset = offset

    def fit(self, matrix, target):
        self.mean_ = float(np.mean(target))
        self.n_features_in_ = matrix.shape[1]
        return self

    def predict(self, matrix):
        return np.full(
            matrix.shape[0],
            self.mean_ + self.offset,
        )


def synthetic_factory(configuration):
    return SyntheticRegressor(
        offset=(
            0.0
            if configuration == "A_squared_error"
            else 1.0
        )
    )


def constant_reader(manifest, value=5):
    ids = manifest["id"].tolist()

    def read(first, last):
        assert 1 <= first <= last <= 1885
        return {
            series_id: [value] * (last - first + 1)
            for series_id in ids
        }

    return read


def test_documented_feature_interface():
    experiment.check_feature_interface()


@pytest.mark.parametrize(
    "cutoff",
    [1689, 1745, 1801, 1857],
)
def test_historical_origin_generation(cutoff):
    origins = experiment.generate_origins(cutoff)
    first = experiment.feature_module.MIN_ORIGIN
    upper = cutoff - experiment.HORIZON
    count = (upper - first) // 7 + 1

    assert origins == experiment.generate_origins(cutoff)
    assert len(origins) == count
    assert len(set(origins)) == count
    assert all(
        b > a
        for a, b in zip(origins, origins[1:])
    )
    assert origins[0] == first
    assert origins[-1] == upper
    assert {
        (origin - 1) % 7 for origin in origins
    } == set(range(7))

    expected = tuple(
        first
        + (upper - first) * index // (count - 1)
        + (
            0
            if index in (0, count - 1)
            else index % 7 - 3
        )
        for index in range(count)
    )
    assert origins == expected

    for index, origin in enumerate(origins):
        anchor = first + (upper - first) * index // (count - 1)
        assert abs(origin - anchor) <= 3
        assert origin - 56 >= 1
        targets = [
            origin + horizon for horizon in range(1, 29)
        ]
        assert len(targets) == 28
        assert min(targets) > origin
        assert max(targets) <= cutoff

    # Anchor spacing is approximately weekly. Actual origin gaps can be
    # shorter because of the documented deterministic weekday offsets.
    gaps = [
        b - a
        for a, b in zip(origins, origins[1:])
    ]
    assert min(gaps) >= 1
    assert any(gap < 7 for gap in gaps)


def test_singleton_origin_schedule():
    assert experiment.generate_origins(85) == (57,)


@pytest.mark.parametrize(
    "cutoff",
    [84, 1858, True],
)
def test_invalid_cutoff_rejected(cutoff):
    with pytest.raises(ValueError):
        experiment.generate_origins(cutoff)


def test_training_labels_respect_cutoff():
    manifest = manifest_for()
    histories = {
        series_id: [5] * 113
        for series_id in manifest["id"]
    }
    predictors, targets, diagnostics = experiment.build_fit_examples(
        histories,
        manifest,
        calendar_mapping(),
        {},
        113,
    )
    assert len(predictors) == len(targets)
    assert diagnostics["maximum_training_target_day"] <= 113
    assert diagnostics["training_example_count"] == (
        2 * len(experiment.generate_origins(113)) * 28
    )
    assert not {
        "origin_day",
        "target_day",
        "target_sales",
    }.intersection(predictors.columns)
    assert tuple(predictors.columns) == experiment.PREDICTOR_COLUMNS


def test_fitting_rejects_history_beyond_cutoff():
    manifest = manifest_for()
    histories = {
        series_id: [5] * 114
        for series_id in manifest["id"]
    }
    with pytest.raises(ValueError, match="exactly"):
        experiment.build_fit_examples(
            histories,
            manifest,
            calendar_mapping(),
            {},
            113,
        )


@pytest.fixture
def frozen_forecasts():
    manifest = manifest_for()
    calendar = calendar_mapping()
    forecasts, diagnostics = experiment.fit_and_forecast(
        113,
        "A_squared_error",
        manifest,
        calendar,
        {},
        constant_reader(manifest),
        synthetic_factory,
    )
    return manifest, calendar, forecasts, diagnostics


def test_fixed_origin_all_horizons(frozen_forecasts):
    manifest, _, forecasts, diagnostics = frozen_forecasts
    assert len(forecasts) == len(manifest) * 28
    assert "actual_sales" not in forecasts
    assert forecasts["forecast_origin_day"].eq("d_113").all()
    assert forecasts.groupby("id").size().eq(28).all()
    assert not forecasts.duplicated(
        ["id", "model", "horizon"]
    ).any()
    assert set(forecasts["d"]) == {
        f"d_{day}" for day in range(114, 142)
    }
    assert diagnostics["maximum_training_target_day"] <= 113


def test_future_actuals_do_not_change_predictions(frozen_forecasts):
    manifest, _, forecasts, _ = frozen_forecasts
    first = experiment.attach_actuals(
        forecasts,
        {
            series_id: [1] * 28
            for series_id in manifest["id"]
        },
        113,
    )
    second = experiment.attach_actuals(
        forecasts,
        {
            series_id: [999] * 28
            for series_id in manifest["id"]
        },
        113,
    )
    assert (
        first["actual_sales"].tolist()
        != second["actual_sales"].tolist()
    )
    np.testing.assert_array_equal(
        first["predicted_sales"],
        second["predicted_sales"],
    )
    np.testing.assert_array_equal(
        first["raw_predicted_sales"],
        second["raw_predicted_sales"],
    )


def test_backtest_load_order_and_independent_fits():
    manifest = manifest_for()
    calendar = calendar_mapping()
    events = []
    estimators = []

    class EventRegressor(SyntheticRegressor):
        def fit(self, matrix, target):
            events.append(("fit", len(target)))
            return super().fit(matrix, target)

        def predict(self, matrix):
            events.append(("predict", matrix.shape[0]))
            return super().predict(matrix)

    def factory(configuration):
        estimator = EventRegressor(
            offset=(
                0
                if configuration == "A_squared_error"
                else 1
            )
        )
        estimators.append(estimator)
        return estimator

    def reader(first, last):
        if first == 1:
            events.append(("history", last))
        else:
            since_previous_actual = []
            for event in reversed(events):
                if event[0] == "actual":
                    break
                since_previous_actual.append(event)
            assert sum(
                event[0] == "fit"
                for event in since_previous_actual
            ) == 2
            assert sum(
                event[0] == "predict"
                for event in since_previous_actual
            ) == 2
            events.append(("actual", first, last))
        return {
            series_id: [5] * (last - first + 1)
            for series_id in manifest["id"]
        }

    result = experiment.run_backtests(
        manifest,
        calendar,
        {},
        reader,
        origins=(113, 169, 225),
        estimator_factory=factory,
    )
    assert len(estimators) == 6
    assert len({
        id(estimator) for estimator in estimators
    }) == 6
    assert result["selected_configuration"] == "A_squared_error"
    assert len(result["metrics"]) == 14
    for diagnostic in result["diagnostics"]:
        assert (
            diagnostic["maximum_training_target_day"]
            <= diagnostic["fitting_cutoff"]
        )
    assert result["combined_metrics"][
        "A_squared_error"
    ]["observations"] == 168


def test_backtests_prepare_each_partition_only_once(monkeypatch):
    manifest = manifest_for()
    calendar = calendar_mapping()
    read_calls = []
    build_calls = []
    original_build = experiment.build_fit_examples

    def reader(first, last):
        read_calls.append((first, last))
        return {
            series_id: [5] * (last - first + 1)
            for series_id in manifest["id"]
        }

    def build_spy(histories, selected, dates, prices, cutoff):
        build_calls.append(cutoff)
        return original_build(
            histories,
            selected,
            dates,
            prices,
            cutoff,
        )

    monkeypatch.setattr(
        experiment,
        "build_fit_examples",
        build_spy,
    )
    experiment.run_backtests(
        manifest,
        calendar,
        {},
        reader,
        origins=(113, 169, 225),
        estimator_factory=synthetic_factory,
    )

    assert build_calls == [113, 169, 225]
    assert read_calls == [
        (1, 113),
        (114, 141),
        (1, 169),
        (170, 197),
        (1, 225),
        (226, 253),
    ]


def test_prepared_inputs_unchanged_by_independent_configuration_fits():
    manifest = manifest_for()
    partition = experiment.prepare_partition(
        113,
        manifest,
        calendar_mapping(),
        {},
        constant_reader(manifest),
    )
    training_before = partition["features"].copy(deep=True)
    prediction_before = partition[
        "prediction_features"
    ].copy(deep=True)
    targets_before = partition["targets"].copy()

    for configuration in experiment.CONFIGURATION_ORDER:
        experiment.fit_prepared_partition(
            partition,
            configuration,
            synthetic_factory,
        )

    pd.testing.assert_frame_equal(
        partition["features"],
        training_before,
    )
    pd.testing.assert_frame_equal(
        partition["prediction_features"],
        prediction_before,
    )
    np.testing.assert_array_equal(
        partition["targets"],
        targets_before,
    )


def test_missing_prices_remain_nan_and_zero_remains_zero():
    manifest = manifest_for()
    histories = {
        series_id: [0] * 113
        for series_id in manifest["id"]
    }
    predictors, _, _ = experiment.build_fit_examples(
        histories,
        manifest,
        calendar_mapping(),
        {},
        113,
    )
    pipeline = experiment.make_pipeline(
        "A_squared_error",
        tuple(sorted(manifest["id"])),
        synthetic_factory,
    )
    pipeline.fit(predictors, np.ones(len(predictors)))
    matrix = pipeline.named_steps[
        "preprocess"
    ].transform(predictors)
    assert isinstance(matrix, np.ndarray)
    assert np.isnan(matrix).any()
    assert (matrix == 0).any()

    numeric = matrix[
        :,
        -len(experiment.NUMERIC_PREDICTORS):,
    ]
    for name in experiment.MISSING_ALLOWED_COLUMNS:
        index = experiment.NUMERIC_PREDICTORS.index(name)
        assert np.isnan(numeric[:, index]).all()
    index = experiment.NUMERIC_PREDICTORS.index("origin_sales")
    assert (numeric[:, index] == 0).all()


def test_price_features_respect_origin_availability():
    manifest = manifest_for(1)
    calendar = calendar_mapping()
    weeks = experiment.feature_module.training_weeks(calendar)
    histories = {"SERIES_00": [5] * 113}
    prices = {
        ("CA_1", "ITEM_00", week["wm_yr_wk"]): 2.0
        for week in weeks
    }
    first, targets, _ = experiment.build_fit_examples(
        histories,
        manifest,
        calendar,
        prices,
        113,
    )
    changed = dict(prices)
    for week in weeks:
        if week["first_day"] > 113:
            changed[
                ("CA_1", "ITEM_00", week["wm_yr_wk"])
            ] = 999.0
    second, second_targets, _ = experiment.build_fit_examples(
        histories,
        manifest,
        calendar,
        changed,
        113,
    )
    pd.testing.assert_frame_equal(first, second)
    np.testing.assert_array_equal(targets, second_targets)


def test_categorical_encoding_consistency_and_unknown_product():
    manifest = manifest_for()
    histories = {
        series_id: [5] * 113
        for series_id in manifest["id"]
    }
    predictors, targets, _ = experiment.build_fit_examples(
        histories,
        manifest,
        calendar_mapping(),
        {},
        113,
    )
    pipeline = experiment.make_pipeline(
        "A_squared_error",
        tuple(sorted(manifest["id"])),
        synthetic_factory,
    )
    pipeline.fit(predictors, targets)
    encoder = (
        pipeline.named_steps["preprocess"]
        .named_transformers_["categories"]
    )
    categories_before = [
        category.copy()
        for category in encoder.categories_
    ]

    unseen = predictors.iloc[:1].copy()
    unseen.loc[
        :,
        "target_event_type_1",
    ] = "NEW_SCHEDULED_EVENT"
    result = pipeline.predict(unseen)
    assert np.isfinite(result).all()
    for before, after in zip(
        categories_before,
        encoder.categories_,
    ):
        np.testing.assert_array_equal(before, after)

    unknown = predictors.iloc[:1].copy()
    unknown.loc[:, "id"] = "UNKNOWN"
    with pytest.raises(ValueError, match="unknown"):
        pipeline.predict(unknown)


def test_exact_predeclared_configurations():
    a = experiment.make_estimator(
        "A_squared_error"
    ).get_params()
    b = experiment.make_estimator(
        "B_tweedie"
    ).get_params()
    for name, value in experiment.COMMON_PARAMETERS.items():
        assert a[name] == value
        assert b[name] == value
    assert a["objective"] == "reg:squarederror"
    assert b["objective"] == "reg:tweedie"
    assert b["tweedie_variance_power"] == pytest.approx(1.3)


def test_negative_prediction_clipping():
    clipped, diagnostics = experiment.clip_predictions(
        [-2.5, 0, 1.25]
    )
    np.testing.assert_array_equal(
        clipped,
        [0, 0, 1.25],
    )
    assert diagnostics["negative_raw_prediction_count"] == 1
    assert diagnostics["total_upward_adjustment"] == pytest.approx(2.5)
    assert diagnostics["percentage_clipped"] == pytest.approx(100 / 3)


def metric_example():
    records = []
    start = date(2020, 1, 1)
    for series_id, forecast in (("A", 12), ("B", 8)):
        for horizon in range(1, 29):
            records.append({
                "id": series_id,
                "item_id": f"ITEM_{series_id}",
                "model": experiment.MODEL,
                "forecast_origin_day": "d_1857",
                "horizon": horizon,
                "d": f"d_{1857 + horizon}",
                "date": (
                    start + timedelta(days=horizon - 1)
                ).isoformat(),
                "actual_sales": 10,
                "raw_predicted_sales": float(forecast),
                "predicted_sales": float(forecast),
            })
    return pd.DataFrame(
        records,
        columns=experiment.PREDICTION_COLUMNS,
    )


def test_per_product_observations_are_explicitly_28():
    products, summary = experiment.metric_tables(
        metric_example(),
        1857,
    )
    assert products["observations"].tolist() == [28, 28]
    assert summary.loc[
        summary["evaluation_level"].eq("pooled_product_day"),
        "observations",
    ].iloc[0] == 56


@pytest.mark.parametrize(
    "problem",
    ["missing", "duplicate", "wrong_day"],
)
def test_metric_tables_rejects_incomplete_or_misaligned_product_horizons(
    problem,
):
    predictions = metric_example()
    if problem == "missing":
        predictions = predictions.iloc[1:].copy()
    elif problem == "duplicate":
        predictions = pd.concat(
            [predictions, predictions.iloc[:1]],
            ignore_index=True,
        )
    else:
        predictions.loc[0, "d"] = "d_1859"
    with pytest.raises(ValueError):
        experiment.metric_tables(predictions, 1857)


def test_pooled_and_aggregate_metrics_reuse_existing_functions(
    monkeypatch,
):
    predictions = metric_example()
    calls = []

    def metric_spy(actual, predicted):
        calls.append(("product", len(actual)))
        return calculate_metrics(actual, predicted)

    def aggregate_spy(records, level):
        calls.append((level, len(records)))
        return evaluate_records(records, level)

    monkeypatch.setattr(
        experiment,
        "calculate_metrics",
        metric_spy,
    )
    monkeypatch.setattr(
        experiment,
        "evaluate_records",
        aggregate_spy,
    )
    products, summary = experiment.metric_tables(
        predictions,
        1857,
    )
    assert calls.count(("product", 28)) == 2
    assert ("pooled_product_day", 56) in calls
    assert ("aggregate_daily", 56) in calls

    pooled = summary.loc[
        summary["evaluation_level"].eq("pooled_product_day")
    ].iloc[0]
    aggregate = summary.loc[
        summary["evaluation_level"].eq("aggregate_daily")
    ].iloc[0]
    assert len(products) == 2
    assert pooled["mae"] == pytest.approx(2)
    assert pooled["rmse"] == pytest.approx(2)
    assert pooled["wape_pct"] == pytest.approx(20)
    assert aggregate["mae"] == pytest.approx(0)
    assert aggregate["wape_pct"] == pytest.approx(0)


def selection_records():
    rows = []
    for configuration in experiment.CONFIGURATION_ORDER:
        for origin, actual in (
            (100, 1),
            (200, 100),
            (300, 100),
        ):
            if configuration == "A_squared_error":
                predicted = 0 if origin == 100 else 100
            else:
                predicted = 1 if origin == 100 else 120
            rows.append({
                "configuration": configuration,
                "origin_day": origin,
                "id": "A",
                "horizon": 1,
                "actual_sales": actual,
                "predicted_sales": predicted,
            })
    return pd.DataFrame(rows)


def test_selection_uses_combined_errors_not_mean_fold_wape():
    selected, metrics = experiment.select_configuration(
        selection_records()
    )
    assert selected == "A_squared_error"
    assert metrics["A_squared_error"]["observations"] == 3
    assert metrics["A_squared_error"]["wape_pct"] == pytest.approx(
        100 / 201
    )
    assert metrics["B_tweedie"]["wape_pct"] == pytest.approx(
        4000 / 201
    )
    # Averaging fold WAPEs would incorrectly prefer configuration B.
    assert (
        metrics["A_squared_error"]["wape_pct"]
        < metrics["B_tweedie"]["wape_pct"]
    )


def test_configuration_tie_prefers_squared_error():
    rows = selection_records()
    rows["predicted_sales"] = rows["actual_sales"]
    selected, _ = experiment.select_configuration(rows)
    assert selected == "A_squared_error"


def test_zero_actual_combined_wape_is_undefined_and_tie_prefers_a():
    rows = selection_records()
    rows["actual_sales"] = 0
    rows["predicted_sales"] = 0
    selected, metrics = experiment.select_configuration(rows)
    assert selected == "A_squared_error"
    assert all(
        value["wape_pct"] is None
        for value in metrics.values()
    )


def test_duplicate_backtest_prediction_keys_rejected():
    rows = selection_records()
    rows = pd.concat(
        [rows, rows.iloc[:1]],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="Duplicate"):
        experiment.select_configuration(rows)


def write_table(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        na_rep="",
        lineterminator="\n",
    )


def write_benchmark(root, predictions, model):
    benchmark = predictions[
        list(experiment.BASELINE_PREDICTION_COLUMNS)
    ].copy()
    benchmark["model"] = model
    _, summary = experiment.metric_tables(
        benchmark,
        1857,
    )
    predictions_path = root / f"{model}_predictions.csv"
    summary_path = root / f"{model}_summary.csv"
    write_table(predictions_path, benchmark)
    write_table(summary_path, summary)
    return predictions_path, summary_path


@pytest.mark.parametrize(
    "field",
    ["id", "horizon", "d", "date", "actual_sales"],
)
def test_benchmark_alignment_mismatch_rejected(
    tmp_path,
    field,
):
    predictions = metric_example()
    prediction_path, summary_path = write_benchmark(
        tmp_path,
        predictions,
        experiment.RIDGE_MODEL,
    )
    benchmark = pd.read_csv(prediction_path)
    if field in ("horizon", "actual_sales"):
        benchmark.loc[0, field] = 999
    else:
        benchmark.loc[0, field] = "WRONG"
    write_table(prediction_path, benchmark)
    with pytest.raises(ValueError):
        experiment.load_benchmark(
            prediction_path,
            summary_path,
            experiment.RIDGE_MODEL,
            predictions,
        )


def test_benchmark_metric_reconciliation(tmp_path):
    predictions = metric_example()
    prediction_path, summary_path = write_benchmark(
        tmp_path,
        predictions,
        experiment.RIDGE_MODEL,
    )
    summary, products = experiment.load_benchmark(
        prediction_path,
        summary_path,
        experiment.RIDGE_MODEL,
        predictions,
    )
    assert len(summary) == 2
    assert len(products) == 2

    modified = pd.read_csv(summary_path)
    modified.loc[0, "mae"] = 999.0
    write_table(summary_path, modified)
    with pytest.raises(ValueError, match="disagrees"):
        experiment.load_benchmark(
            prediction_path,
            summary_path,
            experiment.RIDGE_MODEL,
            predictions,
        )


def test_benchmark_extra_ridge_prediction_column_is_permitted(tmp_path):
    predictions = metric_example()
    prediction_path, summary_path = write_benchmark(
        tmp_path,
        predictions,
        experiment.RIDGE_MODEL,
    )
    benchmark = pd.read_csv(prediction_path)
    benchmark["raw_predicted_sales"] = benchmark["predicted_sales"]
    write_table(prediction_path, benchmark)

    summary, products = experiment.load_benchmark(
        prediction_path,
        summary_path,
        experiment.RIDGE_MODEL,
        predictions,
    )
    assert len(summary) == 2
    assert len(products) == 2


def test_output_schemas_metadata_publication_and_input_immutability(
    tmp_path,
):
    manifest = manifest_for()
    calendar = calendar_mapping()
    result = experiment.run_backtests(
        manifest,
        calendar,
        {},
        constant_reader(manifest),
        origins=(113, 169, 225),
        estimator_factory=synthetic_factory,
    )
    predictions = metric_example()
    products, summary = experiment.metric_tables(
        predictions,
        1857,
    )
    seasonal = summary.copy()
    seasonal["model"] = experiment.SEASONAL_MODEL
    ridge = summary.copy()
    ridge["model"] = experiment.RIDGE_MODEL
    comparison = experiment.comparison_table(
        summary,
        seasonal,
        ridge,
    )

    source = tmp_path / "source.txt"
    source.write_text(
        "unchanged synthetic input",
        encoding="utf-8",
    )
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = {
        "selected_configuration": result["selected_configuration"],
        "combined_metrics": result["combined_metrics"],
        "feature_names": list(experiment.PREDICTOR_COLUMNS),
        "test_sales_accessed": False,
    }
    paths = experiment.publish_outputs(
        tmp_path,
        result["metrics"],
        predictions,
        products,
        summary,
        comparison,
        metadata,
    )
    assert len(paths) == 6
    assert all(
        path.is_file() and path.stat().st_size > 0
        for path in paths
    )
    assert tuple(pd.read_csv(paths[0]).columns) == (
        experiment.BACKTEST_METRIC_COLUMNS
    )
    assert tuple(pd.read_csv(paths[1]).columns) == (
        experiment.PREDICTION_COLUMNS
    )
    assert tuple(pd.read_csv(paths[2]).columns) == (
        experiment.PRODUCT_COLUMNS
    )
    assert tuple(pd.read_csv(paths[3]).columns) == (
        experiment.SUMMARY_COLUMNS
    )
    assert tuple(pd.read_csv(paths[4]).columns) == (
        experiment.COMPARISON_COLUMNS
    )
    assert len(pd.read_csv(paths[0])) == 14
    assert len(pd.read_csv(paths[4])) == 6
    assert json.loads(
        paths[5].read_text(encoding="utf-8")
    ) == metadata
    assert hashlib.sha256(
        source.read_bytes()
    ).hexdigest() == before


def test_csv_round_trip_accepts_numeric_format_variation_and_nullable_wape(
    tmp_path,
):
    expected = pd.DataFrame(
        [{
            "model": experiment.MODEL,
            "evaluation_level": "pooled_product_day",
            "observations": 560,
            "mae": 1.0,
            "rmse": 2.5,
            "wape_pct": None,
            "bias_units": -0.25,
        }],
        columns=experiment.SUMMARY_COLUMNS,
        dtype=object,
    )
    path = tmp_path / "metrics.csv"
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(experiment.SUMMARY_COLUMNS)
        writer.writerow([
            experiment.MODEL,
            "pooled_product_day",
            "5.60e2",
            "1",
            "2.5000000000000004",
            "",
            "-2.5e-1",
        ])

    experiment.verify_staged_csv(
        expected,
        path,
        ("model", "evaluation_level"),
    )


@pytest.mark.parametrize(
    "missing",
    [None, np.nan, pd.NA],
)
def test_csv_round_trip_accepts_explicit_nullable_wape_variants(
    tmp_path,
    missing,
):
    frame = pd.DataFrame(
        [{
            "model": experiment.MODEL,
            "evaluation_level": "aggregate_daily",
            "observations": 28,
            "mae": 1.0,
            "rmse": 1.0,
            "wape_pct": missing,
            "bias_units": 1.0,
        }],
        columns=experiment.SUMMARY_COLUMNS,
        dtype=object,
    )
    path = tmp_path / "summary.csv"
    frame.to_csv(path, index=False, na_rep="")
    experiment.verify_staged_csv(
        frame,
        path,
        ("model", "evaluation_level"),
    )


@pytest.mark.parametrize(
    "column,replacement",
    [
        ("id", "A_CHANGED"),
        ("horizon", "2"),
        ("actual_sales", "10.5"),
        ("predicted_sales", "999"),
    ],
)
def test_csv_round_trip_rejects_changed_identifiers_keys_or_values(
    tmp_path,
    column,
    replacement,
):
    expected = metric_example().iloc[:1].copy()
    path = tmp_path / "predictions.csv"
    expected.to_csv(path, index=False)
    actual = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
    )
    actual.loc[0, column] = replacement
    actual.to_csv(path, index=False)

    with pytest.raises(ValueError):
        experiment.verify_staged_csv(
            expected,
            path,
            ("id", "model", "horizon"),
        )


def test_csv_round_trip_rejects_missing_required_numeric_value(tmp_path):
    expected = metric_example().iloc[:1].copy()
    path = tmp_path / "predictions.csv"
    expected.to_csv(path, index=False)
    actual = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
    )
    actual.loc[0, "predicted_sales"] = ""
    actual.to_csv(path, index=False)
    with pytest.raises(ValueError, match="empty"):
        experiment.verify_staged_csv(
            expected,
            path,
            ("id", "model", "horizon"),
        )


def twenty_product_integrity_inputs():
    manifest = manifest_for(20)
    calendar = calendar_mapping()
    actuals = {
        series_id: [10] * 28
        for series_id in manifest["id"]
    }
    records = []
    for product in manifest.itertuples(index=False):
        for horizon in range(1, 29):
            day = 1857 + horizon
            records.append({
                "id": product.id,
                "item_id": product.item_id,
                "model": experiment.MODEL,
                "forecast_origin_day": "d_1857",
                "horizon": horizon,
                "d": f"d_{day}",
                "date": calendar[day]["date"].isoformat(),
                "actual_sales": 10,
                "raw_predicted_sales": 10.25,
                "predicted_sales": 10.25,
            })
    predictions = pd.DataFrame(
        records,
        columns=experiment.PREDICTION_COLUMNS,
    )
    products, summary = experiment.metric_tables(
        predictions,
        1857,
    )

    seasonal = summary.copy()
    seasonal["model"] = experiment.SEASONAL_MODEL
    ridge = summary.copy()
    ridge["model"] = experiment.RIDGE_MODEL
    comparison = experiment.comparison_table(
        summary,
        seasonal,
        ridge,
    )

    backtest_rows = []
    combined_metrics = calculate_metrics(
        [10] * (20 * 28 * 3),
        [10.25] * (20 * 28 * 3),
    )
    for configuration in experiment.CONFIGURATION_ORDER:
        for origin in experiment.BACKTEST_ORIGINS:
            for record in summary.to_dict("records"):
                backtest_rows.append({
                    "configuration": configuration,
                    "objective": experiment.CONFIGURATIONS[
                        configuration
                    ]["objective"],
                    "scope": "fold",
                    "forecast_origin_day": f"d_{origin}",
                    "fitting_cutoff": origin,
                    "evaluation_level": record["evaluation_level"],
                    "observations": record["observations"],
                    **{
                        metric: record[metric]
                        for metric in experiment.METRICS
                    },
                })
        backtest_rows.append({
            "configuration": configuration,
            "objective": experiment.CONFIGURATIONS[
                configuration
            ]["objective"],
            "scope": "combined",
            "forecast_origin_day": "combined",
            "fitting_cutoff": None,
            "evaluation_level": "pooled_product_day",
            **combined_metrics,
        })
    backtests = pd.DataFrame(
        backtest_rows,
        columns=experiment.BACKTEST_METRIC_COLUMNS,
        dtype=object,
    )
    return (
        backtests,
        predictions,
        products,
        summary,
        comparison,
        manifest,
        actuals,
        calendar,
    )


def test_complete_output_integrity_with_20_products_and_28_horizons(
    tmp_path,
):
    inputs = twenty_product_integrity_inputs()
    checks = experiment.verify_outputs(*inputs)

    assert checks["status"] == "PASS"
    assert checks["prediction_rows"] == 560
    assert checks["per_product_metric_rows"] == 20
    assert checks["summary_metric_rows"] == 2
    assert checks["comparison_rows"] == 6
    assert checks["backtest_metric_rows"] == 14
    assert checks["test_sales_accessed"] is False

    backtests, predictions, products, summary, comparison = inputs[:5]
    paths = experiment.publish_outputs(
        tmp_path,
        backtests,
        predictions,
        products,
        summary,
        comparison,
        {"integrity_checks": checks},
    )
    assert len(paths) == 6
    assert len(pd.read_csv(paths[0])) == 14
    assert len(pd.read_csv(paths[1])) == 560
    assert len(pd.read_csv(paths[2])) == 20
    assert len(pd.read_csv(paths[3])) == 2
    assert len(pd.read_csv(paths[4])) == 6


@pytest.mark.parametrize(
    "problem",
    [
        "product_observations",
        "combined_observations",
        "actual",
        "comparison",
    ],
)
def test_complete_integrity_rejects_corrupted_outputs(problem):
    inputs = list(twenty_product_integrity_inputs())
    if problem == "product_observations":
        inputs[2].loc[0, "observations"] = 27
    elif problem == "combined_observations":
        index = inputs[0].index[
            inputs[0]["scope"].eq("combined")
        ][0]
        inputs[0].loc[index, "observations"] = 1679
    elif problem == "actual":
        inputs[1].loc[0, "actual_sales"] = 999
    else:
        inputs[4] = inputs[4].iloc[:-1].copy()

    with pytest.raises(ValueError):
        experiment.verify_outputs(*inputs)


def write_selected_sales(path, manifest, calendar):
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            experiment.feature_module.SALES_COLUMNS
        )
        start = calendar[1]["date"]
        for product in manifest.itertuples(index=False):
            for day in range(1, 1914):
                split = (
                    "train"
                    if day <= 1857
                    else "validation"
                    if day <= 1885
                    else "test"
                )
                writer.writerow([
                    product.id,
                    product.item_id,
                    "CA_1",
                    "FOODS_3",
                    f"d_{day}",
                    day,
                    (
                        start + timedelta(days=day - 1)
                    ).isoformat(),
                    (
                        5
                        if day <= 1857
                        else "BLOCKED_HELD_OUT_DEMAND"
                    ),
                    split,
                ])


class GuardedRow(list):
    def __init__(self, values, sales_index):
        super().__init__(values)
        self.sales_index = sales_index

    def __getitem__(self, key):
        if key == self.sales_index:
            raise AssertionError(
                "Disallowed sales field accessed."
            )
        return super().__getitem__(key)


def test_loader_does_not_access_demand_after_fitting_cutoff(
    tmp_path,
    monkeypatch,
):
    manifest = manifest_for(1)
    calendar = calendar_mapping()
    path = tmp_path / "selected_sales.csv"
    write_selected_sales(path, manifest, calendar)
    before = path.read_bytes()
    original_reader = csv.reader

    def guarded_reader(handle, **kwargs):
        reader = original_reader(handle, **kwargs)
        names = next(reader)
        yield names
        day_index = names.index("day_number")
        sales_index = names.index("sales")
        for row in reader:
            if int(row[day_index]) > 113:
                yield GuardedRow(row, sales_index)
            else:
                yield row

    monkeypatch.setattr(
        experiment.csv,
        "reader",
        guarded_reader,
    )
    loaded = experiment.load_demand_window(
        path,
        manifest,
        calendar,
        1,
        113,
    )
    assert loaded == {"SERIES_00": [5] * 113}
    assert path.read_bytes() == before


def test_loader_refuses_final_test_interval(tmp_path):
    with pytest.raises(ValueError, match="1..1885"):
        experiment.load_demand_window(
            tmp_path / "not_opened.csv",
            manifest_for(),
            calendar_mapping(),
            1886,
            1913,
        )


def test_real_xgboost_reproducibility_with_fixed_random_state():
    manifest = manifest_for()
    histories = {
        series_id: [5] * 113
        for series_id in manifest["id"]
    }
    predictors, targets, _ = experiment.build_fit_examples(
        histories,
        manifest,
        calendar_mapping(),
        {},
        113,
    )
    ids = tuple(sorted(manifest["id"]))
    first = experiment.make_pipeline(
        "A_squared_error",
        ids,
    )
    second = experiment.make_pipeline(
        "A_squared_error",
        ids,
    )
    first.fit(predictors, targets)
    second.fit(predictors, targets)
    np.testing.assert_allclose(
        first.predict(predictors.iloc[:56]),
        second.predict(predictors.iloc[:56]),
        rtol=1e-7,
        atol=1e-7,
    )