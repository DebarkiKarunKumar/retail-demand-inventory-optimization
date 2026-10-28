"""Synthetic Stage 5A tests; no real M5 holdout or expensive model fitting."""

import csv
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.evaluate_holdout as holdout
import src.forecast_xgboost as experiment
import src.prepare_xgb_features as features


def make_calendar():
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
        for day in range(1, 1914)
    }


def make_manifest():
    return pd.DataFrame([
        {
            "selection_rank": index + 1,
            "id": f"SERIES_{index:02d}",
            "item_id": f"ITEM_{index:02d}",
            "store_id": "CA_1",
            "dept_id": "FOODS_3",
        }
        for index in range(20)
    ], columns=features.MANIFEST_COLUMNS)


def metadata_for(manifest):
    return {
        "model": "xgboost",
        "selected_configuration": "A_squared_error",
        "selected_objective": "reg:squarederror",
        "model_configurations": experiment.CONFIGURATIONS,
        "feature_names": list(holdout.FEATURE_NAMES),
        "selected_product_count": 20,
        "selected_product_ids_in_rank_order": manifest["id"].tolist(),
        "scope": {"store_id": "CA_1", "dept_id": "FOODS_3"},
        "chronological_windows": {
            "training": [1, 1857],
            "validation": [1858, 1885],
            "excluded_final_test": [1886, 1913],
            "forecast_horizon": 28,
        },
        "categorical_encoding": dict(holdout.EXPECTED_ENCODING),
    }


def build_project(root):
    processed = root / "data/processed"
    raw = root / "data/raw"
    evaluation = root / "outputs/evaluation"
    source = root / "src"
    for directory in (processed, raw, evaluation, source):
        directory.mkdir(parents=True, exist_ok=True)
    manifest = make_manifest()
    calendar = make_calendar()

    manifest.to_csv(processed / "selected_products.csv", index=False)
    (processed / "selection_metadata.json").write_text(
        json.dumps({
            "actual_selected_series_count": 20,
            "selected_series_ids_in_rank_order": manifest["id"].tolist(),
            "store_id": "CA_1",
            "dept_id": "FOODS_3",
        }),
        encoding="utf-8",
    )
    with (processed / "selected_sales.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(features.SALES_COLUMNS)
        for index, product in enumerate(manifest.itertuples(index=False)):
            for day in range(1, 1914):
                writer.writerow([
                    product.id, product.item_id, "CA_1", "FOODS_3",
                    f"d_{day}", day, calendar[day]["date"].isoformat(),
                    index + 1 if day <= 1885 else index + 4,
                    "train" if day <= 1857
                    else "validation" if day <= 1885 else "test",
                ])

    with (raw / "calendar.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(features.CALENDAR_COLUMNS)
        for day, record in calendar.items():
            writer.writerow([
                f"d_{day}", record["date"].isoformat(),
                record["wm_yr_wk"], record["snap_CA"], "", "", "", "",
            ])

    weeks = features.training_weeks(calendar, cutoff=1885)
    with (raw / "sell_prices.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(features.PRICE_COLUMNS)
        for product in manifest.itertuples(index=False):
            for week in weeks:
                writer.writerow([
                    "CA_1", product.item_id, week["wm_yr_wk"], 2.5,
                ])
            writer.writerow([
                "CA_1", product.item_id,
                calendar[1913]["wm_yr_wk"], "FUTURE_PRICE_NOT_PARSED",
            ])

    (evaluation / "xgb_run_metadata.json").write_text(
        json.dumps(metadata_for(manifest)), encoding="utf-8"
    )
    modules = {
        "prepare_xgb_features.py": features.__file__,
        "forecast_xgboost.py": experiment.__file__,
        "evaluate.py": holdout.evaluation_module.__file__,
        "evaluate_holdout.py": holdout.__file__,
    }
    for name, original in modules.items():
        (source / name).write_bytes(Path(original).read_bytes())
    return manifest, calendar


@pytest.fixture
def project(tmp_path):
    manifest, calendar = build_project(tmp_path)
    return tmp_path, manifest, calendar


@pytest.fixture
def fast_forecaster(monkeypatch):
    calls = []

    def fake_forecast(
        origin, configuration, manifest, calendar, prices, read_demand,
        estimator_factory=None, *,
        max_fitting_cutoff=1857, max_forecast_day=1885,
    ):
        assert origin == 1885
        assert configuration == "A_squared_error"
        assert max_fitting_cutoff == 1885
        assert max_forecast_day == 1913
        histories = read_demand(1, 1885)
        calls.append((origin, configuration))
        items = manifest.set_index("id")["item_id"].to_dict()
        weeks = features.training_weeks(calendar, cutoff=1885)
        records = []
        raw_values = []

        for series_id in sorted(histories):
            rows = features.build_feature_rows(
                series_id, items[series_id], histories[series_id],
                origin, calendar, prices, weeks,
                max_origin=1885, max_forecast_day=1913,
            )
            assert len(rows) == 28
            for horizon in range(1, 29):
                raw = float(histories[series_id][-1] - 2)
                raw_values.append(raw)
                records.append({
                    "id": series_id,
                    "item_id": items[series_id],
                    "model": "xgboost",
                    "forecast_origin_day": "d_1885",
                    "horizon": horizon,
                    "d": f"d_{1885 + horizon}",
                    "date": calendar[1885 + horizon]["date"].isoformat(),
                    "raw_predicted_sales": raw,
                    "predicted_sales": max(0.0, raw),
                })
        _, clipping = experiment.clip_predictions(raw_values)
        origins = experiment.generate_origins(1885, max_cutoff=1885)
        return pd.DataFrame(records), {
            "training_example_count": 20 * len(origins) * 28,
            "maximum_training_target_day": 1885,
            "forecast_origin_day": "d_1885",
            "clipping": clipping,
        }

    monkeypatch.setattr(experiment, "fit_and_forecast", fake_forecast)
    return calls


def test_compatibility_and_stage4_defaults(project):
    root, _, calendar = project
    holdout.check_compatibility()
    path = root / "data/raw/calendar.csv"
    assert max(features.load_calendar(path)) == 1885
    assert max(features.load_calendar(path, max_day=1913)) == 1913
    assert len(features.training_weeks(calendar, cutoff=1885)) > len(
        features.training_weeks(calendar)
    )
    with pytest.raises(ValueError):
        features.checked_history_prefix([1] * 1885, 1885)
    assert len(features.checked_history_prefix(
        [1] * 1885, 1885, max_origin=1885
    )) == 1885
    with pytest.raises(ValueError):
        experiment.generate_origins(1885)
    assert experiment.generate_origins(1885, max_cutoff=1885)[-1] == 1857
    with pytest.raises(ValueError):
        experiment.load_demand_window(
            root / "not_opened.csv", make_manifest(),
            calendar, 1886, 1913,
        )


def test_adapter_check_fails_clearly(monkeypatch):
    def old_loader(path):
        return {}

    monkeypatch.setattr(features, "load_calendar", old_loader)
    with pytest.raises(ValueError, match="Compatibility adapter"):
        holdout.check_compatibility()


def test_extended_training_labels_stop_at_cutoff():
    calendar = make_calendar()
    with pytest.raises(ValueError):
        features.build_training_examples(
            {"A": [1] * 1885}, {"A": "ITEM_A"},
            calendar, {}, [1857],
        )
    examples = features.build_training_examples(
        {"A": [1] * 1885}, {"A": "ITEM_A"},
        calendar, {}, [1857],
        fitting_cutoff=1885, max_forecast_day=1913,
    )
    assert len(examples) == 28
    assert examples["target_day"].max() == 1885
    with pytest.raises(ValueError):
        features.build_training_examples(
            {"A": [1] * 1913}, {"A": "ITEM_A"},
            calendar, {}, [1858],
            fitting_cutoff=1885, max_forecast_day=1913,
        )


def test_extended_fit_examples():
    manifest = make_manifest().iloc[:1].copy()
    predictors, targets, diagnostics = experiment.build_fit_examples(
        {"SERIES_00": [2] * 1885},
        manifest, make_calendar(), {}, 1885,
        max_fitting_cutoff=1885, max_forecast_day=1913,
    )
    assert len(predictors) == len(targets)
    assert diagnostics["maximum_training_target_day"] == 1885
    assert diagnostics["historical_origins"][-1] == 1857
    assert "target_sales" not in predictors


def test_future_sales_and_prices_do_not_change_origin_features():
    calendar = make_calendar()
    weeks = features.training_weeks(calendar, cutoff=1885)
    prices = {
        ("CA_1", "ITEM_A", week["wm_yr_wk"]): 2.5
        for week in weeks
    }
    future_prices = dict(prices)
    future_prices[
        ("CA_1", "ITEM_A", calendar[1913]["wm_yr_wk"])
    ] = "DO_NOT_READ"

    first = features.build_feature_rows(
        "A", "ITEM_A", [3] * 1885, 1885,
        calendar, prices, weeks,
        max_origin=1885, max_forecast_day=1913,
    )
    changed = features.build_feature_rows(
        "A", "ITEM_A", [3] * 1885 + [None] * 28, 1885,
        calendar, future_prices, weeks,
        max_origin=1885, max_forecast_day=1913,
    )

    assert len(first) == len(changed) == 28

    for original_row, changed_row in zip(first, changed):
        assert tuple(original_row) == tuple(changed_row)

        for name, original_value in original_row.items():
            changed_value = changed_row[name]
            original_nan = (
                isinstance(original_value, (float, np.floating))
                and np.isnan(original_value)
            )
            changed_nan = (
                isinstance(changed_value, (float, np.floating))
                and np.isnan(changed_value)
            )

            if original_nan or changed_nan:
                assert name in features.PRICE_FEATURES[:5]
                assert original_nan and changed_nan, (
                    f"{name}: missingness changed after future-data perturbation"
                )
            elif isinstance(original_value, (float, np.floating)):
                assert isinstance(
                    changed_value, (int, float, np.integer, np.floating)
                )
                assert changed_value == pytest.approx(
                    original_value,
                    rel=1e-12,
                    abs=1e-12,
                ), name
            else:
                assert changed_value == original_value, name

    assert first[0]["target_day"] == 1886
    assert first[-1]["target_day"] == 1913


def test_frozen_metadata_validates(project):
    root, manifest, _ = project
    metadata = holdout.load_frozen_configuration(root, manifest)
    assert metadata["selected_objective"] == "reg:squarederror"
    assert len(metadata["feature_names"]) == 29


@pytest.mark.parametrize(
    "problem",
    ["missing", "objective", "parameters", "features", "encoding", "products"],
)
def test_invalid_frozen_metadata_rejected(project, problem):
    root, manifest, _ = project
    path = root / "outputs/evaluation/xgb_run_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if problem == "missing":
        del metadata["selected_configuration"]
    elif problem == "objective":
        metadata["selected_objective"] = "reg:tweedie"
    elif problem == "parameters":
        metadata["model_configurations"]["A_squared_error"]["max_depth"] = 9
    elif problem == "features":
        metadata["feature_names"] = list(reversed(metadata["feature_names"]))
    elif problem == "encoding":
        metadata["categorical_encoding"]["matrix_dtype"] = "float64"
    else:
        metadata["selected_product_ids_in_rank_order"] = ["WRONG"] * 20
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        holdout.load_frozen_configuration(root, manifest)


def test_forecast_does_not_access_or_hash_holdout_sales(
    project, fast_forecaster, monkeypatch
):
    root, _, _ = project
    sales_path = root / "data/processed/selected_sales.csv"
    original_hash = holdout.sha256_file
    original_reader = csv.reader

    def hash_spy(path):
        assert Path(path) != sales_path
        return original_hash(path)

    class GuardedRow(list):
        def __init__(self, values, sales_index):
            super().__init__(values)
            self.sales_index = sales_index

        def __getitem__(self, key):
            if key == self.sales_index:
                raise AssertionError("Holdout sales accessed during forecast.")
            return super().__getitem__(key)

    def guarded_reader(handle, **kwargs):
        reader = original_reader(handle, **kwargs)
        names = next(reader)
        yield names
        if "sales" not in names or "day_number" not in names:
            yield from reader
            return
        day_index = names.index("day_number")
        sales_index = names.index("sales")
        for row in reader:
            yield (
                GuardedRow(row, sales_index)
                if int(row[day_index]) > 1885 else row
            )

    monkeypatch.setattr(holdout, "sha256_file", hash_spy)
    monkeypatch.setattr(experiment.csv, "reader", guarded_reader)
    result = holdout.forecast(root)
    assert result["status"] == "FORECAST_READY"
    assert result["selected_sales_fingerprint"]["full_file_sha256"] is None
    assert fast_forecaster == [(1885, "A_squared_error")]
    frame = pd.read_csv(root / "outputs/evaluation" / holdout.FROZEN_NAME)
    assert len(frame) == 560
    assert "actual_sales" not in frame
    assert not frame.duplicated(["id", "model", "horizon"]).any()


def test_scoring_is_independent_and_preserves_inputs_and_predictions(
    project, fast_forecaster, monkeypatch
):
    root, _, _ = project
    inputs = [path for path in root.rglob("*") if path.is_file()]
    hashes = {path: holdout.sha256_file(path) for path in inputs}
    ready = holdout.forecast(root)
    directory = root / "outputs/evaluation"
    frozen_path = directory / holdout.FROZEN_NAME
    frozen_bytes = frozen_path.read_bytes()
    assert not (directory / holdout.SCORED_NAME).exists()

    def no_fit(*args, **kwargs):
        raise AssertionError("Scoring attempted refitting.")

    monkeypatch.setattr(experiment, "fit_and_forecast", no_fit)
    original_load = experiment.load_demand_window

    def guarded_load(path, manifest, calendar, first, last, **kwargs):
        if first == 1886:
            marker = holdout.read_json(directory / holdout.METADATA_NAME)
            assert marker["status"] == "SCORING_STARTED"
            assert frozen_path.read_bytes() == frozen_bytes
            assert holdout.sha256_file(frozen_path) == ready["frozen_prediction_sha256"]
        return original_load(path, manifest, calendar, first, last, **kwargs)

    monkeypatch.setattr(experiment, "load_demand_window", guarded_load)
    result = holdout.score(root)
    assert result["status"] == "EVALUATED"
    assert result["scoring_attempts_by_this_script"] == 1
    assert result["successful_scoring_publications_by_this_script"] == 1
    assert result["holdout_evaluated_at_utc"]
    assert frozen_path.read_bytes() == frozen_bytes
    assert result["original_prechange_script_sha256"] == holdout.ORIGINAL_SCRIPT_HASHES

    for path, digest in hashes.items():
        assert holdout.sha256_file(path) == digest
    assert result["selected_sales_fingerprint"]["full_file_sha256"] == (
        holdout.sha256_file(root / "data/processed/selected_sales.csv")
    )

    scored = pd.read_csv(directory / holdout.SCORED_NAME)
    frozen = pd.read_csv(frozen_path)
    pd.testing.assert_frame_equal(scored[list(holdout.FROZEN_COLUMNS)], frozen)
    products = pd.read_csv(directory / holdout.PRODUCT_NAME)
    summary = pd.read_csv(directory / holdout.SUMMARY_NAME)
    assert len(products) == 20
    assert products["observations"].eq(28).all()

    pooled = summary.loc[
        summary["evaluation_level"].eq("pooled_product_day")
    ].iloc[0]
    aggregate = summary.loc[
        summary["evaluation_level"].eq("aggregate_daily")
    ].iloc[0]
    expected_pooled = holdout.evaluation_module.calculate_metrics(
        scored["actual_sales"].tolist(), scored["predicted_sales"].tolist()
    )
    daily = scored.groupby("date")[["actual_sales", "predicted_sales"]].sum()
    expected_aggregate = holdout.evaluation_module.calculate_metrics(
        daily["actual_sales"].tolist(), daily["predicted_sales"].tolist()
    )
    assert pooled["observations"] == 560
    assert aggregate["observations"] == 28
    for name in holdout.METRIC_NAMES:
        assert pooled[name] == pytest.approx(expected_pooled[name])
        assert aggregate[name] == pytest.approx(expected_aggregate[name])
    for row in products.itertuples(index=False):
        group = scored.loc[scored["id"].eq(row.id)]
        expected = holdout.evaluation_module.calculate_metrics(
            group["actual_sales"].tolist(), group["predicted_sales"].tolist()
        )
        for name in holdout.METRIC_NAMES:
            assert getattr(row, name) == pytest.approx(expected[name])


def test_metric_functions_are_reused(project, fast_forecaster, monkeypatch):
    root, _, _ = project
    holdout.forecast(root)
    calls = []
    original_product = holdout.evaluation_module.calculate_metrics
    original_aggregate = holdout.evaluation_module.evaluate_records

    def product_spy(actual, predicted):
        calls.append(("product", len(actual)))
        return original_product(actual, predicted)

    def aggregate_spy(records, level):
        calls.append((level, len(records)))
        return original_aggregate(records, level)

    monkeypatch.setattr(
        holdout.evaluation_module, "calculate_metrics", product_spy
    )
    monkeypatch.setattr(
        holdout.evaluation_module, "evaluate_records", aggregate_spy
    )
    holdout.score(root)
    assert calls.count(("product", 28)) == 20
    assert ("pooled_product_day", 560) in calls
    assert ("aggregate_daily", 560) in calls


def test_repeated_scoring_and_forecast_regeneration_refused(
    project, fast_forecaster
):
    root, _, _ = project
    holdout.forecast(root)
    holdout.score(root)
    with pytest.raises(ValueError, match="only once"):
        holdout.score(root)
    with pytest.raises(ValueError, match="already exist"):
        holdout.forecast(root)


def test_prediction_hash_mismatch_blocks_actual_access(
    project, fast_forecaster, monkeypatch
):
    root, _, _ = project
    holdout.forecast(root)
    path = root / "outputs/evaluation" / holdout.FROZEN_NAME
    path.write_bytes(path.read_bytes() + b"\n")
    original_load = experiment.load_demand_window

    def loader(path, manifest, calendar, first, last, **kwargs):
        assert last <= 1885
        return original_load(path, manifest, calendar, first, last, **kwargs)

    monkeypatch.setattr(experiment, "load_demand_window", loader)
    with pytest.raises(ValueError, match="SHA256"):
        holdout.score(root)


def test_exact_prediction_keys_checked(project, fast_forecaster):
    root, manifest, calendar = project
    holdout.forecast(root)
    path = root / "outputs/evaluation" / holdout.FROZEN_NAME
    frame, _ = holdout.read_frozen_predictions(path)
    frame.loc[0, "horizon"] = 2
    with pytest.raises(ValueError):
        holdout.verify_frozen_predictions(frame, manifest, calendar)


def test_frozen_csv_round_trip_and_hash(project, fast_forecaster):
    root, _, _ = project
    metadata = holdout.forecast(root)
    path = root / "outputs/evaluation" / holdout.FROZEN_NAME
    frame, digest = holdout.read_frozen_predictions(path)
    assert holdout.verify_frozen_csv(frame, path) == digest
    assert digest == metadata["frozen_prediction_sha256"]
    assert "actual_sales" not in frame


@pytest.mark.parametrize(
    "problem",
    ["identifier", "numeric", "actual_column", "column_order", "duplicate"],
)
def test_frozen_csv_corruption_rejected(project, fast_forecaster, problem):
    root, _, _ = project
    holdout.forecast(root)
    path = root / "outputs/evaluation" / holdout.FROZEN_NAME
    expected, _ = holdout.read_frozen_predictions(path)
    changed = expected.copy(deep=True)
    if problem == "identifier":
        changed.loc[0, "id"] = "WRONG"
    elif problem == "numeric":
        changed.loc[0, "raw_predicted_sales"] = 999.0
    elif problem == "actual_column":
        changed["actual_sales"] = 1
    elif problem == "column_order":
        changed = changed[list(reversed(changed.columns))]
    else:
        changed = pd.concat(
            [changed.iloc[:1], changed.iloc[:1], changed.iloc[2:]],
            ignore_index=True,
        )
    changed.to_csv(path, index=False)
    with pytest.raises(ValueError):
        holdout.verify_frozen_csv(expected, path)


def test_changed_holdout_actuals_cannot_change_frozen_predictions(
    project, fast_forecaster
):
    root, _, _ = project
    holdout.forecast(root)
    directory = root / "outputs/evaluation"
    before = (directory / holdout.FROZEN_NAME).read_bytes()
    sales = root / "data/processed/selected_sales.csv"
    frame = pd.read_csv(sales)
    frame.loc[frame["day_number"].ge(1886), "sales"] = 999
    frame.to_csv(sales, index=False)
    assert holdout.score(root)["status"] == "EVALUATED"
    assert (directory / holdout.FROZEN_NAME).read_bytes() == before


def test_failed_scoring_attempt_cannot_retry(project, fast_forecaster, monkeypatch):
    root, _, _ = project
    holdout.forecast(root)
    original_load = experiment.load_demand_window

    def fail_actuals(path, manifest, calendar, first, last, **kwargs):
        if first == 1886:
            raise ValueError("Synthetic actual-read failure")
        return original_load(path, manifest, calendar, first, last, **kwargs)

    monkeypatch.setattr(experiment, "load_demand_window", fail_actuals)
    with pytest.raises(ValueError, match="Synthetic"):
        holdout.score(root)
    directory = root / "outputs/evaluation"
    metadata = holdout.read_json(directory / holdout.METADATA_NAME)
    assert metadata["status"] == "SCORING_FAILED"
    assert metadata["scoring_attempts_by_this_script"] == 1
    assert not any(
        (directory / name).exists() for name in holdout.SCORING_OUTPUT_NAMES
    )
    with pytest.raises(ValueError, match="only once"):
        holdout.score(root)


def test_interrupted_scoring_leaves_attempt_marker(
    project, fast_forecaster, monkeypatch
):
    root, _, _ = project
    holdout.forecast(root)
    original_load = experiment.load_demand_window
    directory = root / "outputs/evaluation"

    def interrupt(path, manifest, calendar, first, last, **kwargs):
        if first == 1886:
            marker = holdout.read_json(directory / holdout.METADATA_NAME)
            assert marker["status"] == "SCORING_STARTED"
            raise KeyboardInterrupt("Synthetic interruption")
        return original_load(path, manifest, calendar, first, last, **kwargs)

    monkeypatch.setattr(experiment, "load_demand_window", interrupt)
    with pytest.raises(KeyboardInterrupt):
        holdout.score(root)
    assert holdout.read_json(
        directory / holdout.METADATA_NAME
    )["status"] == "SCORING_STARTED"
    with pytest.raises(ValueError, match="only once"):
        holdout.score(root)


def test_publication_failure_preserves_attempt_and_frozen_artifact(
    project, fast_forecaster, monkeypatch
):
    root, _, _ = project
    holdout.forecast(root)
    directory = root / "outputs/evaluation"
    frozen_path = directory / holdout.FROZEN_NAME
    before = frozen_path.read_bytes()
    original_replace = Path.replace

    def fail_commit(self, target):
        target = Path(target)
        if self.name == holdout.METADATA_NAME and target.name == holdout.METADATA_NAME:
            candidate = json.loads(self.read_text(encoding="utf-8"))
            if candidate.get("status") == "EVALUATED":
                raise OSError("Synthetic metadata commit failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_commit)
    with pytest.raises(RuntimeError, match="publication failed"):
        holdout.score(root)
    metadata = holdout.read_json(directory / holdout.METADATA_NAME)
    assert metadata["status"] == "SCORING_FAILED"
    assert frozen_path.read_bytes() == before
    assert not any(
        (directory / name).exists() for name in holdout.SCORING_OUTPUT_NAMES
    )
    with pytest.raises(ValueError, match="only once"):
        holdout.score(root)


def test_scoring_publication_requires_started_state(project, fast_forecaster):
    root, _, _ = project
    holdout.forecast(root)
    directory = root / "outputs/evaluation"
    path = directory / holdout.METADATA_NAME
    original = path.read_bytes()
    metadata = holdout.read_json(path)
    metadata.update({
        "status": "EVALUATED",
        "scoring_attempts_by_this_script": 1,
        "successful_scoring_publications_by_this_script": 1,
    })
    with pytest.raises(ValueError, match="state transition"):
        holdout.publish_phase(
            root,
            {
                holdout.SCORED_NAME: pd.DataFrame(),
                holdout.PRODUCT_NAME: pd.DataFrame(),
                holdout.SUMMARY_NAME: pd.DataFrame(),
            },
            metadata,
            holdout.sha256_file(path),
        )
    assert path.read_bytes() == original


def test_durable_marker_precedes_actual_access(project, fast_forecaster, monkeypatch):
    root, _, _ = project
    holdout.forecast(root)
    sync_calls = []
    original_sync = holdout.os.fsync
    original_load = experiment.load_demand_window
    directory = root / "outputs/evaluation"

    def sync_spy(descriptor):
        sync_calls.append(descriptor)
        return original_sync(descriptor)

    def load_spy(path, manifest, calendar, first, last, **kwargs):
        if first == 1886:
            assert sync_calls
            marker = holdout.read_json(directory / holdout.METADATA_NAME)
            assert marker["status"] == "SCORING_STARTED"
        return original_load(path, manifest, calendar, first, last, **kwargs)

    monkeypatch.setattr(holdout.os, "fsync", sync_spy)
    monkeypatch.setattr(experiment, "load_demand_window", load_spy)
    assert holdout.score(root)["status"] == "EVALUATED"


def test_original_source_hashes_preserved():
    result = holdout.original_script_hashes()
    assert result == holdout.ORIGINAL_SCRIPT_HASHES
    assert result["src/forecast_xgboost.py"] == (
        "9D7934B723235628A6FCDAE180EDB80A8BEEECDFC8EE62FCA1A6A1528AC892AB"
    )
    assert result["src/prepare_xgb_features.py"] == (
        "DD2AAE1312C47661B90D81F42DAD245F3E0C29135EFD8D482599CF38F134040D"
    )