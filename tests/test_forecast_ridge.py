"""Synthetic-only tests for the Stage 4 Ridge implementation."""

import csv
import hashlib
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import src.forecast_ridge as ridge
from src.evaluate import calculate_metrics, evaluate_records


def calendar_mapping(start=date(2011, 1, 3)):
    return {
        day: start + timedelta(days=day - 1)
        for day in range(1, 1886)
    }


def histories_for(count=20):
    return {
        f"SERIES_{index:02d}": [
            index + 1 + day % 7
            for day in range(1, 1858)
        ]
        for index in range(count)
    }


def manifest_for(histories):
    records = []
    for rank, (series_id, values) in enumerate(
        sorted(histories.items()),
        start=1,
    ):
        records.append(
            {
                "selection_rank": rank,
                "id": series_id,
                "item_id": f"ITEM_{series_id}",
                "store_id": "CA_1",
                "dept_id": "FOODS_3",
                "train_total_units": sum(values),
                "train_positive_days": sum(
                    value > 0 for value in values
                ),
                "train_zero_sales_pct": (
                    100
                    * sum(value == 0 for value in values)
                    / 1857
                ),
            }
        )
    return pd.DataFrame(
        records,
        columns=ridge.MANIFEST_COLUMNS,
    )


@pytest.fixture(scope="module")
def examples():
    histories = histories_for()
    calendar = calendar_mapping()
    features, targets, keys = ridge.build_examples(
        histories,
        calendar,
    )
    return histories, calendar, features, targets, keys


@pytest.fixture(scope="module")
def small_model(examples):
    histories, _, features, targets, keys = examples

    # Use two origins per product for focused estimator tests.
    mask = keys["origin_day"].isin(
        ridge.ORIGINS[:2]
    ).to_numpy()
    training_features = features.loc[mask].reset_index(drop=True)
    training_targets = targets[mask]
    ids = tuple(sorted(histories))
    model = ridge.fit_model(
        training_features,
        training_targets,
        ids,
    )
    return model, training_features, training_targets, ids


def test_origin_schedule_count_range_order_weekdays_and_determinism():
    origins = ridge.generate_origins()
    assert origins == ridge.ORIGINS
    assert origins == ridge.generate_origins()
    assert len(origins) == 258
    assert len(set(origins)) == 258
    assert origins == tuple(sorted(origins))
    assert origins[0] == 29
    assert origins[-1] == 1829
    assert all(29 <= origin <= 1829 for origin in origins)
    assert all(
        b > a
        for a, b in zip(origins, origins[1:])
    )

    expected = tuple(
        29
        + (1800 * index) // 257
        + (
            0
            if index in (0, 257)
            else (index % 7) - 3
        )
        for index in range(258)
    )
    assert origins == expected

    for index, origin in enumerate(origins):
        anchor = 29 + (1800 * index) // 257
        assert abs(origin - anchor) <= 3

    assert {
        (origin - 1) % 7 for origin in origins
    } == set(range(7))
    calendar = calendar_mapping()
    assert {
        calendar[origin].weekday() for origin in origins
    } == set(range(7))

    for origin in origins:
        targets = [
            origin + horizon for horizon in range(1, 29)
        ]
        assert len(targets) == 28
        assert min(targets) > origin
        assert max(targets) <= 1857


def test_example_count_keys_and_target_boundaries(examples):
    histories, _, features, targets, keys = examples
    assert len(features) == len(targets) == len(keys) == 144480
    assert tuple(features.columns) == ridge.FEATURES
    assert not keys.duplicated(
        ["id", "origin_day", "horizon"]
    ).any()
    assert keys["target_day"].max() == 1857
    assert keys["target_day"].le(1857).all()
    assert (
        keys["target_day"]
        == keys["origin_day"] + keys["horizon"]
    ).all()
    assert keys.groupby(
        ["id", "origin_day"]
    ).size().eq(28).all()
    assert set(keys["origin_day"]) == set(ridge.ORIGINS)

    for series_id in histories:
        product_keys = keys.loc[keys["id"].eq(series_id)]
        assert len(product_keys) == 258 * 28
        for origin, group in product_keys.groupby(
            "origin_day",
            sort=True,
        ):
            assert group["horizon"].tolist() == list(range(1, 29))
            assert group["target_day"].tolist() == list(
                range(origin + 1, origin + 29)
            )


def test_exact_lags_windows_and_population_standard_deviation():
    history = list(range(1, 1858))
    rows = ridge.feature_rows(
        "A",
        history,
        calendar_mapping(),
        29,
    )
    row = rows[0]
    assert row["origin_sales"] == 29
    assert row["origin_lag_7"] == 22
    assert row["origin_lag_14"] == 15
    assert row["origin_lag_28"] == 1

    for window in (7, 14, 28):
        expected = np.arange(
            30 - window,
            30,
            dtype=float,
        )
        assert row[f"mean_{window}"] == pytest.approx(
            expected.mean()
        )
        assert row[f"std_{window}"] == pytest.approx(
            expected.std(ddof=0)
        )
    assert row["zero_pct_28"] == pytest.approx(0)


def test_zero_percentage_uses_exactly_28_origin_relative_days():
    history = [9] * 1857
    history[1:29] = [0] * 7 + [4] * 21
    row = ridge.feature_rows(
        "A",
        history,
        calendar_mapping(),
        29,
    )[0]
    assert row["zero_pct_28"] == pytest.approx(25)


def test_same_weekday_mean_uses_actual_calendar():
    # Day 1 is Thursday, deliberately not a day-number-derived weekday.
    calendar = calendar_mapping(date(2011, 1, 6))
    history = list(range(1, 1858))
    rows = ridge.feature_rows(
        "A",
        history,
        calendar,
        29,
    )
    for horizon, row in enumerate(rows, start=1):
        weekday = calendar[29 + horizon].weekday()
        days = [
            day
            for day in range(2, 30)
            if calendar[day].weekday() == weekday
        ]
        assert len(days) == 4
        assert row["target_weekday"] == weekday
        assert row["same_weekday_mean_28"] == pytest.approx(
            sum(history[day - 1] for day in days) / 4
        )


def test_historical_features_ignore_future_sales():
    original = list(range(1, 1858))
    changed = original.copy()
    changed[29:] = [None] * (1857 - 29)
    assert ridge.feature_rows(
        "A",
        original,
        calendar_mapping(),
        29,
    ) == ridge.feature_rows(
        "A",
        changed,
        calendar_mapping(),
        29,
    )


@pytest.mark.parametrize(
    "bad",
    [None, -1, 0.5, np.nan, np.inf, True, "3"],
)
def test_invalid_history_rejected(bad):
    history = [1] * 29
    history[0] = bad
    with pytest.raises(ValueError):
        ridge.feature_rows(
            "A",
            history,
            calendar_mapping(),
            29,
        )


def test_missing_history_rejected():
    with pytest.raises(ValueError, match="Missing"):
        ridge.feature_rows(
            "A",
            [1] * 28,
            calendar_mapping(),
            29,
        )


def test_missing_or_nonconsecutive_calendar_rejected():
    calendar = calendar_mapping()
    del calendar[10]
    with pytest.raises(ValueError, match="calendar"):
        ridge.feature_rows(
            "A",
            [1] * 1857,
            calendar,
            29,
        )

    calendar = calendar_mapping()
    calendar[30] += timedelta(days=1)
    with pytest.raises(ValueError, match="consecutive"):
        ridge.feature_rows(
            "A",
            [1] * 1857,
            calendar,
            29,
        )


def test_all_validation_horizons_without_actuals(
    examples,
    small_model,
):
    histories, calendar, _, _, _ = examples
    model, _, _, _ = small_model
    forecasts, _ = ridge.forecast_validation(
        model,
        histories,
        calendar,
        manifest_for(histories),
    )
    assert len(forecasts) == 560
    assert "actual_sales" not in forecasts
    assert forecasts.groupby("id").size().eq(28).all()
    assert forecasts["forecast_origin_day"].eq("d_1857").all()
    assert set(forecasts["d"]) == {
        f"d_{day}" for day in range(1858, 1886)
    }


def test_validation_actual_changes_cannot_change_predictions(
    examples,
    small_model,
):
    histories, calendar, _, _, _ = examples
    model, _, _, _ = small_model
    forecasts, _ = ridge.forecast_validation(
        model,
        histories,
        calendar,
        manifest_for(histories),
    )
    first_actuals = {
        series_id: [1] * 28 for series_id in histories
    }
    changed_actuals = {
        series_id: [1000] * 28 for series_id in histories
    }
    first = ridge.attach_actuals(forecasts, first_actuals)
    changed = ridge.attach_actuals(forecasts, changed_actuals)

    assert (
        first["actual_sales"].tolist()
        != changed["actual_sales"].tolist()
    )
    assert (
        first["raw_predicted_sales"].tolist()
        == changed["raw_predicted_sales"].tolist()
    )
    assert (
        first["predicted_sales"].tolist()
        == changed["predicted_sales"].tolist()
    )


def test_preprocessing_fit_is_training_only(small_model):
    model, features, _, _ = small_model
    scaler = (
        model.named_steps["preprocess"]
        .named_transformers_["numeric"]
    )
    expected = features[list(ridge.NUMERIC)].to_numpy(dtype=float)
    np.testing.assert_allclose(
        scaler.mean_,
        expected.mean(axis=0),
    )
    np.testing.assert_allclose(
        scaler.var_,
        expected.var(axis=0),
    )

    before_mean = scaler.mean_.copy()
    before_scale = scaler.scale_.copy()
    changed = features.iloc[:28].copy()
    changed.loc[:, list(ridge.NUMERIC)] = (
        changed[list(ridge.NUMERIC)] + 10000
    )
    model.predict(changed)
    np.testing.assert_array_equal(
        scaler.mean_,
        before_mean,
    )
    np.testing.assert_array_equal(
        scaler.scale_,
        before_scale,
    )


def test_fixed_product_encoding_and_unknown_rejection(small_model):
    model, features, _, ids = small_model
    encoder = (
        model.named_steps["preprocess"]
        .named_transformers_["categories"]
    )
    assert encoder.categories_[0].tolist() == list(ids)
    assert encoder.categories_[1].tolist() == list(range(1, 29))
    assert encoder.categories_[2].tolist() == list(range(7))
    assert model.named_steps["preprocess"].sparse_output_

    bad = features.iloc[:1].copy()
    bad.loc[:, "id"] = "UNKNOWN"
    with pytest.raises(ValueError, match="unknown"):
        model.predict(bad)

    bad.loc[:, "id"] = None
    with pytest.raises(ValueError):
        model.predict(bad)


@pytest.mark.parametrize(
    "column,value",
    [
        ("horizon", 29),
        ("target_weekday", 7),
        ("horizon", True),
    ],
)
def test_unexpected_categories_rejected(
    small_model,
    column,
    value,
):
    model, features, _, _ = small_model
    bad = features.iloc[:1].copy()
    bad[column] = pd.Series(
        [value],
        index=bad.index,
        dtype=object,
    )
    with pytest.raises(ValueError, match="category"):
        model.predict(bad)


def test_repeated_fits_and_predictions_are_deterministic(
    small_model,
):
    model, features, targets, ids = small_model
    another = ridge.fit_model(features, targets, ids)
    first = model.predict(features.iloc[:56])
    second = another.predict(features.iloc[:56])
    np.testing.assert_allclose(
        first,
        second,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_array_equal(
        first,
        model.predict(features.iloc[:56]),
    )


@pytest.mark.parametrize("availability", ["none", "absent"])
def test_fit_does_not_require_iteration_diagnostics(
    monkeypatch,
    small_model,
    availability,
):
    _, features, targets, ids = small_model
    original_fit = ridge.Ridge.fit

    def fit_without_iterations(self, *args, **kwargs):
        result = original_fit(self, *args, **kwargs)
        if availability == "none":
            self.n_iter_ = None
        else:
            del self.n_iter_
        return result

    monkeypatch.setattr(
        ridge.Ridge,
        "fit",
        fit_without_iterations,
    )
    model = ridge.fit_model(features, targets, ids)
    estimator = model.named_steps["ridge"]

    assert ridge.solver_iterations(estimator) is None
    assert np.isfinite(estimator.coef_).all()
    assert np.isfinite(
        model.predict(features.iloc[:28])
    ).all()


def test_iteration_metadata_records_available_values():
    from types import SimpleNamespace

    assert ridge.solver_iterations(
        SimpleNamespace(n_iter_=np.array([17]))
    ) == [17]
    assert ridge.solver_iterations(
        SimpleNamespace(n_iter_=None)
    ) is None
    assert ridge.solver_iterations(SimpleNamespace()) is None


@pytest.mark.parametrize("field", ["coef_", "intercept_"])
def test_nonfinite_fitted_parameters_rejected(
    monkeypatch,
    small_model,
    field,
):
    _, features, targets, ids = small_model
    original_fit = ridge.Ridge.fit

    def fit_with_invalid_parameter(self, *args, **kwargs):
        result = original_fit(self, *args, **kwargs)
        if field == "coef_":
            self.coef_ = np.full_like(self.coef_, np.nan)
        else:
            self.intercept_ = float("inf")
        return result

    monkeypatch.setattr(
        ridge.Ridge,
        "fit",
        fit_with_invalid_parameter,
    )
    with pytest.raises(
        ValueError,
        match="coefficients or intercept",
    ):
        ridge.fit_model(features, targets, ids)


def test_nonfinite_fitted_predictions_rejected(
    monkeypatch,
    small_model,
):
    _, features, targets, ids = small_model

    def invalid_predict(self, features):
        return np.full(features.shape[0], np.nan)

    monkeypatch.setattr(
        ridge.Ridge,
        "predict",
        invalid_predict,
    )
    with pytest.raises(ValueError, match="fitted predictions"):
        ridge.fit_model(features, targets, ids)


def test_convergence_warnings_still_raise(
    monkeypatch,
    small_model,
):
    import warnings

    _, features, targets, ids = small_model

    def nonconvergent_fit(self, *args, **kwargs):
        warnings.warn(
            "Synthetic convergence failure",
            ridge.ConvergenceWarning,
        )
        return self

    monkeypatch.setattr(
        ridge.Ridge,
        "fit",
        nonconvergent_fit,
    )
    with pytest.raises(ridge.ConvergenceWarning):
        ridge.fit_model(features, targets, ids)


def test_negative_clipping_and_diagnostics():
    clipped, diagnostics = ridge.clip_predictions(
        [-2.5, 0, 3.25, -0.5]
    )
    np.testing.assert_array_equal(
        clipped,
        [0, 0, 3.25, 0],
    )
    assert diagnostics["negative_raw_prediction_count"] == 2
    assert diagnostics["percentage_clipped"] == pytest.approx(50)
    assert diagnostics["minimum_raw_prediction"] == pytest.approx(-2.5)
    assert diagnostics["total_upward_adjustment"] == pytest.approx(3)
    assert clipped[2] == pytest.approx(3.25)


def write_table(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        na_rep="",
        lineterminator="\n",
    )


def baseline_artifacts(root, predictions):
    baseline = predictions[list(ridge.BASELINE_COLUMNS)].copy()
    baseline["model"] = ridge.BASELINE
    baseline["predicted_sales"] = 8
    _, summary = ridge.metric_tables(baseline)
    directory = root / "outputs/evaluation"
    write_table(
        directory / "baseline_predictions.csv",
        baseline,
    )
    write_table(
        directory / "baseline_metrics_summary.csv",
        summary,
    )
    return baseline, summary


@pytest.fixture
def synthetic_predictions(examples, small_model):
    histories, calendar, _, _, _ = examples
    model, _, _, _ = small_model
    manifest = manifest_for(histories)
    forecasts, _ = ridge.forecast_validation(
        model,
        histories,
        calendar,
        manifest,
    )
    actuals = {
        series_id: [10] * 28 for series_id in histories
    }
    predictions = ridge.attach_actuals(forecasts, actuals)
    return predictions, actuals, calendar, manifest


@pytest.mark.parametrize(
    "field",
    ["id", "horizon", "d", "date", "actual_sales"],
)
def test_exact_baseline_alignment_rejects_mismatch(
    tmp_path,
    synthetic_predictions,
    field,
):
    predictions, _, _, _ = synthetic_predictions
    baseline, _ = baseline_artifacts(
        tmp_path,
        predictions,
    )
    if field == "horizon":
        baseline.loc[0, field] = 28
    elif field == "actual_sales":
        baseline.loc[0, field] = 999
    elif field == "date":
        baseline.loc[0, field] = "1900-01-01"
    else:
        baseline.loc[0, field] = "WRONG"

    write_table(
        tmp_path / "outputs/evaluation/baseline_predictions.csv",
        baseline,
    )
    with pytest.raises(ValueError):
        ridge.load_and_align_baseline(
            tmp_path,
            predictions,
        )


def test_baseline_stored_metric_mismatch_rejected(
    tmp_path,
    synthetic_predictions,
):
    predictions, _, _, _ = synthetic_predictions
    _, summary = baseline_artifacts(
        tmp_path,
        predictions,
    )
    summary.loc[0, "mae"] = 999
    write_table(
        tmp_path / "outputs/evaluation/baseline_metrics_summary.csv",
        summary,
    )
    with pytest.raises(ValueError, match="disagrees"):
        ridge.load_and_align_baseline(
            tmp_path,
            predictions,
        )


def test_metric_functions_are_reused(
    monkeypatch,
    synthetic_predictions,
):
    predictions, _, _, _ = synthetic_predictions
    calls = []

    def metric_spy(actual, predicted):
        calls.append(("product", len(actual)))
        return calculate_metrics(actual, predicted)

    def aggregate_spy(records, level):
        calls.append((level, len(records)))
        return evaluate_records(records, level)

    monkeypatch.setattr(
        ridge,
        "calculate_metrics",
        metric_spy,
    )
    monkeypatch.setattr(
        ridge,
        "evaluate_records",
        aggregate_spy,
    )
    products, summary = ridge.metric_tables(predictions)
    assert calls.count(("product", 28)) == 20
    assert ("pooled_product_day", 560) in calls
    assert ("aggregate_daily", 560) in calls
    assert len(products) == 20
    assert len(summary) == 2
    assert summary.loc[
        summary["evaluation_level"].eq("pooled_product_day"),
        "observations",
    ].iloc[0] == 560
    assert summary.loc[
        summary["evaluation_level"].eq("aggregate_daily"),
        "observations",
    ].iloc[0] == 28


@pytest.mark.parametrize(
    "ridge_wape,baseline_wape,expected",
    [
        (10, 11, ridge.MODEL),
        (11, 11, ridge.BASELINE),
        (12, 11, ridge.BASELINE),
        (None, None, ridge.BASELINE),
    ],
)
def test_predeclared_model_selection_rule(
    ridge_wape,
    baseline_wape,
    expected,
):
    def summary(model, wape):
        return pd.DataFrame(
            [
                {
                    "model": model,
                    "evaluation_level": level,
                    "observations": (
                        560
                        if level == "pooled_product_day"
                        else 28
                    ),
                    "mae": 1.0,
                    "rmse": 2.0,
                    "wape_pct": wape,
                    "bias_units": -0.2,
                }
                for level in ridge.LEVELS
            ],
            columns=ridge.SUMMARY_COLUMNS,
            dtype=object,
        )

    comparison, selected = ridge.compare_models(
        summary(ridge.MODEL, ridge_wape),
        summary(ridge.BASELINE, baseline_wape),
    )
    assert selected == expected
    assert len(comparison) == 2
    if ridge_wape is not None:
        assert comparison[
            "difference_wape_pct"
        ].iloc[0] == pytest.approx(
            ridge_wape - baseline_wape
        )


def test_output_schemas_counts_and_publication(
    tmp_path,
    synthetic_predictions,
):
    predictions, actuals, calendar, manifest = synthetic_predictions
    products, summary = ridge.metric_tables(predictions)
    baseline_artifacts(tmp_path, predictions)
    baseline_paths = [
        tmp_path / "outputs/evaluation/baseline_predictions.csv",
        tmp_path / "outputs/evaluation/baseline_metrics_summary.csv",
    ]
    before = {
        path: path.read_bytes() for path in baseline_paths
    }
    baseline_summary = ridge.load_and_align_baseline(
        tmp_path,
        predictions,
    )
    comparison, selected = ridge.compare_models(
        summary,
        baseline_summary,
    )
    checks = ridge.verify_outputs(
        predictions,
        products,
        summary,
        comparison,
        actuals,
        calendar,
        manifest,
    )
    assert tuple(predictions.columns) == ridge.PREDICTION_COLUMNS
    assert len(predictions) == 560
    assert len(products) == 20
    assert len(summary) == 2
    assert len(comparison) == 2
    assert checks["status"] == "PASS"

    paths = ridge.publish_outputs(
        tmp_path,
        predictions,
        products,
        summary,
        comparison,
        {
            "status": "PASS",
            "selected_model": selected,
            "checks": checks,
        },
    )
    assert len(paths) == 5
    assert all(
        path.is_file() and path.stat().st_size > 0
        for path in paths
    )
    assert all(
        path.read_bytes() == before[path]
        for path in baseline_paths
    )


def write_sales(path, histories, test_token):
    path.parent.mkdir(parents=True, exist_ok=True)
    calendar = calendar_mapping()
    start = calendar[1]
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(ridge.SALES_COLUMNS)
        for series_id, values in sorted(histories.items()):
            for day in range(1, 1914):
                split = (
                    "train"
                    if day <= 1857
                    else "validation"
                    if day <= 1885
                    else "test"
                )
                sales = (
                    values[day - 1]
                    if day <= 1857
                    else 10
                    if day <= 1885
                    else test_token
                )
                writer.writerow(
                    [
                        series_id,
                        f"ITEM_{series_id}",
                        "CA_1",
                        "FOODS_3",
                        f"d_{day}",
                        day,
                        (
                            start + timedelta(days=day - 1)
                        ).isoformat(),
                        sales,
                        split,
                    ]
                )


def test_test_sales_are_not_parsed_or_fingerprinted(
    tmp_path,
    monkeypatch,
):
    histories = histories_for(1)
    manifest = manifest_for(histories)
    path = tmp_path / "selected_sales.csv"
    write_sales(
        path,
        histories,
        "UNTOUCHED_TEST_SALES",
    )
    original = ridge.units
    calls = []

    def spy(value, context, text=False):
        assert value != "UNTOUCHED_TEST_SALES"
        assert value != "DIFFERENT_TEST_SALES"
        calls.append(context)
        return original(value, context, text=text)

    monkeypatch.setattr(ridge, "units", spy)
    train, calendar, first_hash = ridge.load_sales(
        path,
        manifest,
        "train",
    )
    assert len(calls) == 1857
    validation, _, validation_hash = ridge.load_sales(
        path,
        manifest,
        "validation",
    )
    assert len(calls) == 1857 + 28
    assert len(train["SERIES_00"]) == 1857
    assert len(validation["SERIES_00"]) == 28
    assert len(calendar) == 1885

    write_sales(
        path,
        histories,
        "DIFFERENT_TEST_SALES",
    )
    _, _, changed_hash = ridge.load_sales(
        path,
        manifest,
        "train",
    )
    _, _, changed_validation_hash = ridge.load_sales(
        path,
        manifest,
        "validation",
    )
    assert first_hash == changed_hash
    assert validation_hash == changed_validation_hash


class GuardedRow(list):
    def __init__(self, values, sales_index):
        super().__init__(values)
        self.sales_index = sales_index

    def __getitem__(self, key):
        if key == self.sales_index:
            raise AssertionError(
                "Test sales field was accessed."
            )
        return super().__getitem__(key)


def test_test_row_sales_field_is_never_indexed(
    tmp_path,
    monkeypatch,
):
    histories = histories_for(1)
    manifest = manifest_for(histories)
    path = tmp_path / "sales.csv"
    write_sales(path, histories, "BLOCKED")
    original_reader = csv.reader

    def guarded_reader(handle, **kwargs):
        rows = original_reader(handle, **kwargs)
        names = next(rows)
        yield names
        split_index = names.index("split")
        sales_index = names.index("sales")
        for row in rows:
            if row[split_index] == "test":
                yield GuardedRow(row, sales_index)
            else:
                yield row

    monkeypatch.setattr(
        ridge.csv,
        "reader",
        guarded_reader,
    )
    ridge.load_sales(path, manifest, "train")
    ridge.load_sales(path, manifest, "validation")


@pytest.mark.parametrize("problem", ["duplicate", "missing"])
def test_duplicate_or_missing_source_days_rejected(
    tmp_path,
    problem,
):
    histories = histories_for(1)
    path = tmp_path / "sales.csv"
    write_sales(path, histories, "UNUSED")

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.reader(handle))
    if problem == "duplicate":
        rows.insert(2, list(rows[1]))
    else:
        rows.pop(1)
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        csv.writer(handle).writerows(rows)

    with pytest.raises(ValueError):
        ridge.load_sales(
            path,
            manifest_for(histories),
            "train",
        )


def build_project(root):
    histories = histories_for()
    manifest = manifest_for(histories)
    processed = root / "data/processed"
    processed.mkdir(parents=True)
    write_table(
        processed / "selected_products.csv",
        manifest,
    )
    (processed / "selection_metadata.json").write_text(
        json.dumps(
            {
                "actual_selected_series_count": 20,
                "selected_series_ids_in_rank_order": (
                    manifest["id"].tolist()
                ),
                "store_id": "CA_1",
                "dept_id": "FOODS_3",
            }
        ),
        encoding="utf-8",
    )
    write_sales(
        processed / "selected_sales.csv",
        histories,
        "NEVER_PARSE_TEST",
    )
    calendar = calendar_mapping()
    records = []
    for series_id, values in sorted(histories.items()):
        for horizon in range(1, 29):
            day = 1857 + horizon
            records.append(
                {
                    "id": series_id,
                    "item_id": f"ITEM_{series_id}",
                    "model": ridge.BASELINE,
                    "forecast_origin_day": "d_1857",
                    "horizon": horizon,
                    "d": f"d_{day}",
                    "date": calendar[day].isoformat(),
                    "actual_sales": 10,
                    "predicted_sales": (
                        values[-7 + (horizon - 1) % 7]
                    ),
                }
            )
    baseline = pd.DataFrame(
        records,
        columns=ridge.BASELINE_COLUMNS,
    )
    _, summary = ridge.metric_tables(baseline)
    write_table(
        root / "outputs/evaluation/baseline_predictions.csv",
        baseline,
    )
    write_table(
        root / "outputs/evaluation/baseline_metrics_summary.csv",
        summary,
    )


def test_full_synthetic_pipeline_order_outputs_and_input_immutability(
    tmp_path,
    monkeypatch,
):
    build_project(tmp_path)
    input_paths = list(
        (tmp_path / "data/processed").iterdir()
    ) + [
        tmp_path / "outputs/evaluation/baseline_predictions.csv",
        tmp_path / "outputs/evaluation/baseline_metrics_summary.csv",
    ]
    fingerprints = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in input_paths
    }
    original_load = ridge.load_sales
    original_forecast = ridge.forecast_validation
    events = []

    def load_spy(path, manifest, period):
        if period == "validation":
            assert events == [
                "train_loaded",
                "forecasts_frozen",
            ]
        result = original_load(path, manifest, period)
        events.append(f"{period}_loaded")
        return result

    def forecast_spy(*args, **kwargs):
        assert events == ["train_loaded"]
        result = original_forecast(*args, **kwargs)
        events.append("forecasts_frozen")
        return result

    monkeypatch.setattr(ridge, "load_sales", load_spy)
    monkeypatch.setattr(
        ridge,
        "forecast_validation",
        forecast_spy,
    )
    result = ridge.run_pipeline(tmp_path)
    metadata = result["metadata"]

    assert events == [
        "train_loaded",
        "forecasts_frozen",
        "validation_loaded",
    ]
    assert metadata["training_example_count"] == 144480
    assert metadata["historical_origins"]["last_actual"] == 1829
    assert metadata["historical_origins"]["count_per_product"] == 258
    assert metadata["historical_origins"]["step_days"] is None
    assert metadata["historical_origins"][
        "origin_weekday_offsets"
    ] == list(range(7))
    assert metadata["historical_origins"][
        "actual_calendar_origin_weekdays"
    ] == list(range(7))
    assert metadata["training_targets"]["last_day"] == 1857
    assert metadata["integrity_checks"]["prediction_rows"] == 560
    assert metadata["integrity_checks"][
        "test_sales_loaded_or_evaluated"
    ] is False
    assert metadata["input_fingerprints"]["sales"][
        "full_file_hashed"
    ] is False
    assert len(result["output_paths"]) == 5

    for path, expected in fingerprints.items():
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == expected
        )