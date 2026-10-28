"""Synthetic tests only; no real M5 files are required."""

import csv
import hashlib
import json
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import src.prepare_xgb_features as features


def synthetic_calendar():
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


def synthetic_manifest(count=1):
    return pd.DataFrame([
        {
            "selection_rank": index + 1,
            "id": f"SERIES_{index:02d}",
            "item_id": f"ITEM_{index:02d}",
            "store_id": "CA_1",
            "dept_id": "FOODS_3",
        }
        for index in range(count)
    ], columns=features.MANIFEST_COLUMNS)


def make_row(history=None, origin=57, horizon=1, calendar=None, prices=None):
    return features.build_feature_row(
        "SERIES_00", "ITEM_00",
        list(range(1, 1858)) if history is None else history,
        origin, horizon,
        synthetic_calendar() if calendar is None else calendar,
        {} if prices is None else prices,
    )


def test_exact_lag_indexing():
    row = make_row()
    assert row["origin_sales"] == 57
    assert row["lag_7"] == 50
    assert row["lag_14"] == 43
    assert row["lag_28"] == 29
    assert row["lag_56"] == 1


def test_exact_rolling_windows_and_population_sd():
    row = make_row()
    for window in (7, 28, 56):
        expected = np.arange(58 - window, 58, dtype=float)
        assert row[f"mean_{window}"] == pytest.approx(expected.mean())
        assert row[f"std_{window}"] == pytest.approx(expected.std(ddof=0))


def test_zero_sales_denominators():
    history = [9] * 1857
    history[1:57] = [0] * 28 + [4] * 28
    row = make_row(history=history)
    assert row["zero_pct_56"] == pytest.approx(50)
    assert row["zero_pct_28"] == pytest.approx(0)
    assert row["lag_56"] == 9


def test_same_weekday_mean_from_actual_dates():
    calendar = synthetic_calendar()
    # Move the complete mapping to a Thursday start.
    shift = timedelta(days=3)
    for record in calendar.values():
        record["date"] += shift
    history = list(range(1, 1858))
    for horizon in range(1, 29):
        row = make_row(history=history, horizon=horizon, calendar=calendar)
        weekday = calendar[57 + horizon]["date"].weekday()
        days = [
            day for day in range(30, 58)
            if calendar[day]["date"].weekday() == weekday
        ]
        assert len(days) == 4
        assert row["target_weekday"] == weekday
        assert row["same_weekday_mean_28"] == pytest.approx(
            sum(history[day - 1] for day in days) / 4
        )


def test_calendar_integrity_and_mapping():
    calendar = synthetic_calendar()
    features.validate_calendar(calendar, 1885)
    assert make_row(calendar=calendar)["target_month"] == calendar[58]["date"].month
    calendar[58]["date"] = calendar[57]["date"]
    with pytest.raises(ValueError, match="duplicated"):
        features.validate_calendar(calendar, 1885)


def test_missing_target_calendar_fails():
    calendar = synthetic_calendar()
    del calendar[85]
    with pytest.raises(ValueError, match="calendar"):
        make_row(calendar=calendar)


def test_snap_and_event_encoding():
    calendar = synthetic_calendar()
    calendar[58].update({
        "snap_CA": 1,
        "event_type_1": "Cultural",
        "event_type_2": "None",
        "has_event": 1,
    })
    row = make_row(calendar=calendar)
    assert row["target_snap_CA"] == 1
    assert row["target_event_type_1"] == "Cultural"
    assert row["target_event_type_2"] == "None"
    assert row["target_has_event"] == 1


def test_future_sales_are_not_accessed_and_features_are_invariant():
    history = list(range(1, 1858))
    changed = history.copy()
    changed[57:] = [None] * (1857 - 57)
    first = features.build_feature_rows(
        "SERIES_00", "ITEM_00", history, 57, synthetic_calendar(), {}
    )
    second = features.build_feature_rows(
        "SERIES_00", "ITEM_00", changed, 57, synthetic_calendar(), {}
    )
    for left, right in zip(first, second):
        for name in features.DEMAND_FEATURES + features.CALENDAR_FEATURES:
            assert left[name] == right[name]


def test_future_prices_do_not_change_features():
    calendar = synthetic_calendar()
    weeks = features.training_weeks(calendar)
    prices = {
        ("CA_1", "ITEM_00", week["wm_yr_wk"]): 2.0
        for week in weeks
    }
    original = features.historical_price_features(
        "ITEM_00", 57, calendar, prices, weeks
    )
    current_index = next(
        week["index"] for week in weeks
        if week["wm_yr_wk"] == calendar[57]["wm_yr_wk"]
    )
    changed = dict(prices)
    for week in weeks:
        if week["index"] > current_index:
            changed[("CA_1", "ITEM_00", week["wm_yr_wk"])] = "DO_NOT_READ"
    actual = features.historical_price_features(
    "ITEM_00", 57, calendar, changed, weeks
    )
    assert actual.keys() == original.keys()
    for key in original:
        expected_value = original[key]
        actual_value = actual[key]

        if isinstance(expected_value, float) and math.isnan(expected_value):
            assert isinstance(actual_value, float)
            assert math.isnan(actual_value), f"Mismatch in {key}"
        else:
            assert actual_value == pytest.approx(expected_value), (
                f"Mismatch in {key}"
            )

def test_missing_price_history_is_nan_not_zero():
    result = features.historical_price_features(
        "ITEM_00", 57, synthetic_calendar(), {}
    )
    for name in features.PRICE_FEATURES[:5]:
        assert math.isnan(result[name])
    assert result["recorded_price_weeks_to_origin"] == 0
    assert result["price_history_missing_indicator"] == 1


def test_no_backward_price_fill():
    calendar = synthetic_calendar()
    weeks = features.training_weeks(calendar)
    prices = {("CA_1", "ITEM_00", weeks[10]["wm_yr_wk"]): 5.0}
    result = features.historical_price_features(
        "ITEM_00", 57, calendar, prices, weeks
    )
    assert math.isnan(result["last_known_price"])
    assert result["recorded_price_weeks_to_origin"] == 0


def test_unconfirmed_current_first_week_is_excluded():
    calendar = synthetic_calendar()
    weeks = features.training_weeks(calendar)
    prices = {("CA_1", "ITEM_00", weeks[0]["wm_yr_wk"]): 2.0}
    first_week = features.historical_price_features(
        "ITEM_00", 3, calendar, prices, weeks
    )
    assert math.isnan(first_week["last_known_price"])
    later = features.historical_price_features(
        "ITEM_00", 8, calendar, prices, weeks
    )
    assert later["last_known_price"] == pytest.approx(2)
    assert later["recorded_price_weeks_to_origin"] == 1


def test_historical_price_changes_and_elapsed_calendar_weeks():
    calendar = synthetic_calendar()
    weeks = features.training_weeks(calendar)
    # Observed weeks 2, 3, 5, 6. Origin is in week 9, which is missing.
    prices = {
        ("CA_1", "ITEM_00", weeks[1]["wm_yr_wk"]): 2.0,
        ("CA_1", "ITEM_00", weeks[2]["wm_yr_wk"]): 2.0,
        ("CA_1", "ITEM_00", weeks[4]["wm_yr_wk"]): 3.0,
        ("CA_1", "ITEM_00", weeks[5]["wm_yr_wk"]): 3.0,
    }
    result = features.historical_price_features(
        "ITEM_00", 57, calendar, prices, weeks
    )
    assert result["last_known_price"] == pytest.approx(3)
    assert result["price_change_pct"] == pytest.approx(0)
    assert result["trailing_4_observed_week_mean_price"] == pytest.approx(2.5)
    assert result["price_relative_to_trailing_mean"] == pytest.approx(1.2)
    assert result["weeks_since_last_observed_price_change"] == 4
    assert result["recorded_price_weeks_to_origin"] == 4
    assert result["price_history_missing_indicator"] == 1


def test_latest_price_change_percentage():
    calendar = synthetic_calendar()
    weeks = features.training_weeks(calendar)
    prices = {
        ("CA_1", "ITEM_00", weeks[1]["wm_yr_wk"]): 2.0,
        ("CA_1", "ITEM_00", weeks[8]["wm_yr_wk"]): 3.0,
    }
    result = features.historical_price_features(
        "ITEM_00", 57, calendar, prices, weeks
    )
    assert result["price_change_pct"] == pytest.approx(50)
    assert result["weeks_since_last_observed_price_change"] == 0
    assert result["price_history_missing_indicator"] == 0


@pytest.mark.parametrize("bad", [None, -1, 1.5, np.nan, np.inf, True, "4"])
def test_invalid_sales_fail(bad):
    history = [1] * 85
    history[0] = bad
    with pytest.raises(ValueError):
        make_row(history=history)


def test_missing_history_fails():
    with pytest.raises(ValueError, match="Missing"):
        make_row(history=[1] * 56)


def test_supervised_examples_keys_labels_and_determinism():
    histories = {"B": list(range(1, 186)), "A": [2] * 185}
    items = {"A": "ITEM_A", "B": "ITEM_B"}
    calendar = synthetic_calendar()
    first = features.build_training_examples(
        histories, items, calendar, {}, [57, 64]
    )
    second = features.build_training_examples(
        histories, items, calendar, {}, [57, 64]
    )
    pd.testing.assert_frame_equal(first, second)
    assert tuple(first.columns) == features.EXAMPLE_COLUMNS
    assert len(first) == 2 * 2 * 28
    assert not first.duplicated(["id", "origin_day", "horizon"]).any()
    assert first["target_day"].le(1857).all()
    for row in first.itertuples(index=False):
        assert row.target_day == row.origin_day + row.horizon
        assert row.target_sales == histories[row.id][row.target_day - 1]


def test_training_last_label_can_be_1857():
    frame = features.build_training_examples(
        {"A": [1] * 1857}, {"A": "ITEM_A"},
        synthetic_calendar(), {}, [1829],
    )
    assert frame["target_day"].max() == 1857
    assert len(frame) == 28


@pytest.mark.parametrize("origins", [[56], [1830], [57, 57], [64, 57], []])
def test_invalid_training_origins_fail(origins):
    with pytest.raises(ValueError):
        features.build_training_examples(
            {"A": [1] * 1857}, {"A": "ITEM_A"},
            synthetic_calendar(), {}, origins,
        )


def test_invalid_or_missing_training_labels_fail():
    history = [1] * 85
    history[84] = None
    with pytest.raises(ValueError, match="training label"):
        features.build_training_examples(
            {"A": history}, {"A": "ITEM_A"},
            synthetic_calendar(), {}, [57],
        )
    with pytest.raises(ValueError, match="missing"):
        features.build_training_examples(
            {"A": [1] * 84}, {"A": "ITEM_A"},
            synthetic_calendar(), {}, [57],
        )


def write_prices(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(features.PRICE_COLUMNS)
        writer.writerows(rows)


def test_duplicate_product_week_keys_rejected(tmp_path):
    calendar = synthetic_calendar()
    weeks = features.training_weeks(calendar)
    week = weeks[0]["wm_yr_wk"]
    path = tmp_path / "prices.csv"
    write_prices(path, [
        ["CA_1", "ITEM_00", week, 2.0],
        ["CA_1", "ITEM_00", week, 3.0],
    ])
    with pytest.raises(ValueError, match="Duplicate"):
        features.load_prices(path, synthetic_manifest(), weeks)


@pytest.mark.parametrize("bad", [0, -1, "NaN", "inf", "bad", ""])
def test_invalid_observed_prices_rejected(tmp_path, bad):
    weeks = features.training_weeks(synthetic_calendar())
    path = tmp_path / "prices.csv"
    write_prices(path, [["CA_1", "ITEM_00", weeks[1]["wm_yr_wk"], bad]])
    with pytest.raises(ValueError):
        features.load_prices(path, synthetic_manifest(), weeks)


def test_future_training_external_price_rows_are_not_parsed(tmp_path):
    weeks = features.training_weeks(synthetic_calendar())
    path = tmp_path / "prices.csv"
    write_prices(path, [
        ["CA_1", "ITEM_00", weeks[1]["wm_yr_wk"], 2.0],
        ["CA_1", "ITEM_00", 999999, "DO_NOT_PARSE"],
        ["TX_1", "ITEM_00", weeks[1]["wm_yr_wk"], "DO_NOT_PARSE"],
    ])
    loaded = features.load_prices(path, synthetic_manifest(), weeks)
    assert len(loaded) == 1


def test_price_gaps_before_and_after_first_recorded_price():
    manifest = synthetic_manifest()
    weeks = features.training_weeks(synthetic_calendar())
    prices = {
        ("CA_1", "ITEM_00", weeks[2]["wm_yr_wk"]): 2.0,
        ("CA_1", "ITEM_00", weeks[4]["wm_yr_wk"]): 3.0,
    }
    report = features.price_coverage_report(manifest, weeks, prices)
    product = report["products"][0]
    assert product["missing_weeks_before_first_recorded_price"] == 2
    assert product["missing_weeks_after_first_recorded_price"] == len(weeks) - 4
    assert report["recorded_product_week_combinations"] == 2
    assert report["missing_product_week_combinations"] == len(weeks) - 2


def test_no_recorded_price_history_is_reported_without_launch_inference():
    weeks = features.training_weeks(synthetic_calendar())
    report = features.price_coverage_report(synthetic_manifest(), weeks, {})
    product = report["products"][0]
    assert product["first_recorded_training_week"] is None
    assert product["missing_weeks_before_first_recorded_price"] is None
    assert product["missing_weeks_after_first_recorded_price"] is None
    assert product["missing_weeks_with_no_recorded_history"] == len(weeks)


def test_audit_schema_and_json_publication(tmp_path):
    manifest = synthetic_manifest(2)
    histories = {"SERIES_00": [1] * 1857, "SERIES_01": [0] * 1857}
    calendar = synthetic_calendar()
    report = features.create_audit_report(
        manifest, histories, calendar, {},
        {"selected_sales": "synthetic/selected_sales.csv"},
    )
    assert report["status"] == "PASS"
    assert report["selected_product_count"] == 2
    assert report["training_observation_count"] == 3714
    assert report["calendar_integrity"]["status"] == "PASS"
    assert report["feature_names"] == list(features.FEATURE_COLUMNS)
    assert set(features.FEATURE_COLUMNS) == set(report["feature_definitions"])
    assert report["leakage_safeguard_checks"]["validation_sales_accessed"] is False
    assert report["leakage_safeguard_checks"]["test_sales_accessed"] is False
    assert report["leakage_safeguard_checks"]["model_trained"] is False
    path = tmp_path / "outputs/evaluation/xgb_feature_audit.json"
    features.write_audit(report, path)

    assert json.loads(path.read_text(encoding="utf-8")) == report

    def reject_invalid_constant(value):
        raise ValueError(f"Invalid JSON numeric constant: {value}")

    parsed = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_invalid_constant
    )

    assert parsed == report


def write_calendar(path, calendar):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(features.CALENDAR_COLUMNS)
        for day, record in sorted(calendar.items()):
            writer.writerow([
                f"d_{day}", record["date"].isoformat(), record["wm_yr_wk"],
                record["snap_CA"], "", "", "", "",
            ])


def test_calendar_loader_missing_event_encoding(tmp_path):
    path = tmp_path / "calendar.csv"
    calendar = synthetic_calendar()
    write_calendar(path, calendar)
    loaded = features.load_calendar(path)
    assert loaded[58]["event_type_1"] == "None"
    assert loaded[58]["event_type_2"] == "None"
    assert loaded[58]["has_event"] == 0


def test_calendar_loader_duplicate_days_rejected(tmp_path):
    path = tmp_path / "calendar.csv"
    write_calendar(path, synthetic_calendar())
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    rows.append(list(rows[1]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    with pytest.raises(ValueError, match="Duplicate"):
        features.load_calendar(path)


def write_selected_sales(path, manifest, calendar):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(features.SALES_COLUMNS)
        start = calendar[1]["date"]
        for product in manifest.itertuples(index=False):
            for day in range(1, 1914):
                split = (
                    "train" if day <= 1857
                    else "validation" if day <= 1885 else "test"
                )
                sales = 2 if day <= 1857 else "DO_NOT_ACCESS_HELD_OUT"
                writer.writerow([
                    product.id, product.item_id, "CA_1", "FOODS_3",
                    f"d_{day}", day,
                    (start + timedelta(days=day - 1)).isoformat(),
                    sales, split,
                ])


class GuardedRow(list):
    def __init__(self, values, sales_index):
        super().__init__(values)
        self.sales_index = sales_index

    def __getitem__(self, key):
        if key == self.sales_index:
            raise AssertionError("Held-out sales field was accessed.")
        return super().__getitem__(key)


def test_training_loader_does_not_access_held_out_sales(tmp_path, monkeypatch):
    path = tmp_path / "sales.csv"
    calendar = synthetic_calendar()
    manifest = synthetic_manifest()
    write_selected_sales(path, manifest, calendar)
    original_reader = csv.reader

    def guarded_reader(handle, **kwargs):
        reader = original_reader(handle, **kwargs)
        names = next(reader)
        yield names
        split_index = names.index("split")
        sales_index = names.index("sales")
        for row in reader:
            if row[split_index] != "train":
                yield GuardedRow(row, sales_index)
            else:
                yield row

    monkeypatch.setattr(features.csv, "reader", guarded_reader)
    histories = features.load_training_sales(path, manifest, calendar)
    assert histories == {"SERIES_00": [2] * 1857}


def test_full_synthetic_audit_preserves_inputs(tmp_path):
    processed = tmp_path / "data/processed"
    raw = tmp_path / "data/raw"
    processed.mkdir(parents=True)
    raw.mkdir(parents=True)
    manifest = synthetic_manifest(20)
    calendar = synthetic_calendar()
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
    write_selected_sales(processed / "selected_sales.csv", manifest, calendar)
    write_calendar(raw / "calendar.csv", calendar)
    weeks = features.training_weeks(calendar)
    price_rows = [
        ["CA_1", product.item_id, week["wm_yr_wk"], 2.5]
        for product in manifest.itertuples(index=False)
        for week in weeks
    ]
    write_prices(raw / "sell_prices.csv", price_rows)
    inputs = list(processed.iterdir()) + list(raw.iterdir())
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in inputs
    }
    result = features.run_audit(tmp_path)
    report = result["report"]
    assert report["status"] == "PASS"
    assert report["selected_product_count"] == 20
    assert report["training_observation_count"] == 20 * 1857
    assert report["price_coverage"]["missing_product_week_combinations"] == 0
    assert result["output_path"].is_file()
    assert list((tmp_path / "outputs/evaluation").iterdir()) == [
        result["output_path"]
    ]
    for path, expected in before.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected