"""Synthetic inventory tests only; no real M5 data or forecasting required."""

import csv
import hashlib
import json
from datetime import date, timedelta
from decimal import Decimal

import pandas as pd
import pytest

import src.simulate_inventory as inventory


def simulate(
    demand,
    estimates,
    initial=0,
    sigma=0.0,
    lead=1,
    holding="0.01",
    shortage="0.50",
):
    return inventory.simulate_product(
        demand, estimates, initial, sigma, lead, holding, shortage
    )


def test_morning_review_requires_l_plus_one_protection():
    rows = simulate([3, 3, 3], [3, 3, 3], initial=0, lead=1)
    assert rows[0]["protection_days"] == 2
    assert rows[0]["order_up_to"] == 6
    assert rows[0]["order_quantity"] == 6
    assert rows[0]["arrivals"] == 0
    assert rows[0]["fulfilled_units"] == 0
    assert rows[0]["unmet_units"] == 3
    assert rows[1]["arrivals"] == 6
    assert rows[1]["fulfilled_units"] == 3


@pytest.mark.parametrize("lead", [1, 2, 3])
def test_orders_arrive_exactly_at_t_plus_l(lead):
    rows = simulate([0] * 8, [4] * 8, lead=lead)
    quantity = rows[0]["order_quantity"]
    assert quantity == 4 * (lead + 1)
    assert rows[0]["order_arrival_day"] == 1 + lead
    assert all(row["arrivals"] == 0 for row in rows[:lead])
    assert rows[lead]["arrivals"] == quantity


def test_outstanding_orders_prevent_duplicate_replenishment():
    rows = simulate([0] * 8, [4] * 8, lead=2)
    assert rows[0]["order_quantity"] == 12
    assert rows[1]["outstanding_before_order"] == 12
    assert rows[1]["inventory_position"] == 12
    assert rows[1]["order_quantity"] == 0
    assert rows[2]["arrivals"] == 12
    assert rows[2]["outstanding_before_order"] == 0
    assert sum(row["arrivals"] for row in rows) == 12


def test_exact_depletion_is_not_a_stockout():
    rows = simulate([5, 0], [0, 0], initial=5)
    assert rows[0]["end_on_hand"] == 0
    assert rows[0]["unmet_units"] == 0
    assert sum(row["unmet_units"] > 0 for row in rows) == 0


def test_lost_sales_are_not_backordered():
    rows = simulate([10, 0, 0], [0, 0, 0], initial=2)
    assert rows[0]["fulfilled_units"] == 2
    assert rows[0]["unmet_units"] == 8
    assert all(row["order_quantity"] == 0 for row in rows)
    assert rows[-1]["end_on_hand"] == 0
    assert rows[-1]["outstanding_end"] == 0


@pytest.mark.parametrize("lead", [1, 2, 3])
def test_inventory_and_pipeline_conservation(lead):
    rows = simulate(
        [3, 8, 0, 12, 1, 5, 9, 0],
        [5] * 8,
        initial=7,
        sigma=1.2,
        lead=lead,
    )
    received = fulfilled = 0
    for row in rows:
        received += row["arrivals"]
        fulfilled += row["fulfilled_units"]
        assert row["demand_proxy_units"] == (
            row["fulfilled_units"] + row["unmet_units"]
        )
        assert 7 + received == row["end_on_hand"] + fulfilled
        assert (
            row["outstanding_start"] - row["arrivals"] + row["order_quantity"]
            == row["outstanding_end"]
        )
        for field in (
            "start_on_hand", "arrivals", "order_quantity",
            "fulfilled_units", "unmet_units", "end_on_hand", "outstanding_end",
        ):
            assert isinstance(row[field], int)
            assert row[field] >= 0
    assert rows[-1]["outstanding_end"] == 0


def test_no_orders_arrive_after_terminal_day():
    rows = simulate([10] * 5, [10] * 5, initial=0, lead=3)
    assert all(
        row["order_arrival_day"] == 0
        or row["order_arrival_day"] <= 5
        for row in rows
    )
    assert all(row["order_quantity"] == 0 for row in rows[2:])
    assert rows[-1]["outstanding_end"] == 0


def test_identical_plans_produce_identical_outcomes():
    demand = [1, 2, 7, 0, 4]
    estimates = [3.25] * 5
    first = simulate(demand, estimates, initial=4, sigma=1.0)
    second = simulate(demand, estimates, initial=4, sigma=1.0)
    assert first == second


def test_future_demand_does_not_change_earlier_orders():
    estimates = [3.0] * 6
    original = simulate([1, 1, 1, 1, 1, 1], estimates, initial=6)
    changed = simulate([1, 1, 99, 99, 99, 99], estimates, initial=6)
    assert original[:2] == changed[:2]
    # Day 3 is ordered before day 3's demand is observed.
    for field in (
        "order_quantity", "inventory_position", "order_up_to",
        "safety_stock_units", "outstanding_before_order",
    ):
        assert original[2][field] == changed[2][field]


def test_forecast_and_demand_inputs_are_not_mutated():
    demand = [1, 2, 3, 4]
    estimates = [2.5] * 4
    demand_before = demand.copy()
    estimates_before = estimates.copy()
    simulate(demand, estimates, initial=5)
    assert demand == demand_before
    assert estimates == estimates_before


def test_safety_stock_and_initial_inventory_formulas():
    estimates = tuple([2.5] * 28)
    expected_buffer = inventory.math.ceil(1.645 * 2.0 * inventory.math.sqrt(2))
    assert inventory.safety_stock(2.0, 2) == expected_buffer
    assert inventory.initial_inventory(estimates, 2.0, 1) == (
        5 + expected_buffer
    )


def test_costs_reconcile_with_integer_state():
    rows = simulate([10, 1, 0], [2] * 3, initial=5)
    expected_holding = Decimal("0.01") * sum(row["end_on_hand"] for row in rows)
    expected_shortage = Decimal("0.50") * sum(row["unmet_units"] for row in rows)
    actual = sum(Decimal(str(row["total_cost"])) for row in rows)
    assert actual == expected_holding + expected_shortage


def test_nine_scenarios_are_fixed():
    scenarios = inventory.scenarios()
    assert len(scenarios) == 9
    assert len({row["scenario_id"] for row in scenarios}) == 9
    assert {
        (row["lead_time"], row["shortage_cost"]) for row in scenarios
    } == {
        (lead, penalty)
        for lead in (1, 2, 3)
        for penalty in ("0.10", "0.50", "1.00")
    }
    assert {row["holding_cost"] for row in scenarios} == {"0.01"}


def test_cost_scenarios_do_not_tune_orders():
    first = simulate([3] * 8, [4] * 8, initial=6, shortage="0.10")
    changed = simulate([3] * 8, [4] * 8, initial=6, shortage="1.00")
    for left, right in zip(first, changed):
        for field in (
            "order_quantity", "arrivals", "end_on_hand",
            "fulfilled_units", "unmet_units",
        ):
            assert left[field] == right[field]


def historical_fixture():
    first = date(2020, 1, 1)
    dates = {
        day: first + timedelta(days=day - 1830)
        for day in range(1830, 1886)
    }
    forecast_dates = tuple(
        dates[1885] + timedelta(days=day) for day in range(1, 29)
    )
    history = [dates[day].weekday() + 1 for day in range(1830, 1886)]
    return dates, forecast_dates, {"A": history}


def test_initialization_is_historical_only():
    dates, forecast_dates, histories = historical_fixture()
    first_plans, first_stats = inventory.prepare_plans(
        histories, dates, {"A": tuple([0.0] * 28)}, forecast_dates
    )
    changed_plans, changed_stats = inventory.prepare_plans(
        histories, dates, {"A": tuple([999.0] * 28)}, forecast_dates
    )
    assert first_stats == changed_stats
    assert first_plans[inventory.POLICY_A] == changed_plans[inventory.POLICY_A]
    assert first_plans[inventory.POLICY_B] != changed_plans[inventory.POLICY_B]
    assert first_plans[inventory.POLICY_A]["A"] == tuple(
        target.weekday() + 1 for target in forecast_dates
    )


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def build_project(root):
    evaluation = root / "outputs/evaluation"
    processed = root / "data/processed"
    evaluation.mkdir(parents=True)
    processed.mkdir(parents=True)
    ids = [f"SERIES_{index:02d}" for index in range(20)]
    start = date(2016, 1, 1)
    frozen_rows = []
    scored_rows = []
    history_rows = []

    for index, series_id in enumerate(ids):
        item = f"ITEM_{index:02d}"
        for horizon in range(1, 29):
            target_date = start + timedelta(days=horizon - 1)
            frozen = [
                series_id, item, "xgboost", "d_1885",
                horizon, f"d_{1885 + horizon}", target_date.isoformat(),
                float(index + 2), float(index + 2),
            ]
            frozen_rows.append(frozen)
            scored_rows.append(
                frozen[:7] + [index + 3] + frozen[7:]
            )
        for day in range(1830, 1914):
            current = start + timedelta(days=day - 1886)
            history_rows.append([
                series_id, item, "CA_1", "FOODS_3",
                f"d_{day}", day, current.isoformat(),
                index + 1 if day <= 1885 else "DO_NOT_READ_HISTORY_HOLDOUT",
                "train" if day <= 1857
                else "validation" if day <= 1885 else "test",
            ])

    frozen_path = evaluation / "holdout_predictions_frozen.csv"
    scored_path = evaluation / "holdout_predictions_scored.csv"
    write_csv(frozen_path, inventory.FROZEN_COLUMNS, frozen_rows)
    write_csv(scored_path, inventory.SCORED_COLUMNS, scored_rows)
    write_csv(
        processed / "selected_sales.csv",
        inventory.HISTORICAL_COLUMNS,
        history_rows,
    )
    metadata = {
        "status": "EVALUATED",
        "model": "xgboost",
        "selected_product_count": 20,
        "selected_product_ids_in_rank_order": ids,
        "training_cutoff": 1885,
        "forecast_origin": 1885,
        "holdout_boundaries": [1886, 1913],
        "forecast_horizon": 28,
        "frozen_prediction_sha256": inventory.sha256_file(frozen_path),
        "scored_prediction_sha256": inventory.sha256_file(scored_path),
    }
    (evaluation / "holdout_run_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )


def test_configuration_is_saved_before_outcomes(tmp_path, monkeypatch):
    build_project(tmp_path)
    original = inventory.load_saved_actuals

    def actual_spy(inputs):
        path = tmp_path / "outputs/evaluation/inventory_run_metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        assert metadata["status"] == "CONFIG_FROZEN"
        assert metadata["configuration_frozen_before_outcomes"] is True
        assert len(metadata["configuration"]["scenarios"]) == 9
        assert metadata["configuration_sha256"] == inventory.sha256_bytes(
            inventory.canonical_bytes(metadata["configuration"])
        )
        return original(inputs)

    monkeypatch.setattr(inventory, "load_saved_actuals", actual_spy)
    result = inventory.run_simulation(tmp_path)
    assert result["status"] == "PASS"


def test_frozen_sha_mismatch_rejected_before_simulation(tmp_path):
    build_project(tmp_path)
    path = tmp_path / "outputs/evaluation/holdout_predictions_frozen.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA256"):
        inventory.run_simulation(tmp_path)


def test_saved_actual_prediction_mismatch_rejected(tmp_path):
    build_project(tmp_path)
    evaluation = tmp_path / "outputs/evaluation"
    path = evaluation / "holdout_predictions_scored.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "predicted_sales"] = 999
    frame.to_csv(path, index=False)
    metadata_path = evaluation / "holdout_run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["scored_prediction_sha256"] = inventory.sha256_file(path)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="preserve"):
        inventory.run_simulation(tmp_path)


def test_history_loader_does_not_access_holdout_sales(tmp_path, monkeypatch):
    build_project(tmp_path)
    inputs = inventory.load_frozen_inputs(tmp_path)
    original_reader = csv.reader

    class GuardedRow(list):
        def __init__(self, values, sales_index):
            super().__init__(values)
            self.sales_index = sales_index

        def __getitem__(self, key):
            if key == self.sales_index:
                raise AssertionError("History loader accessed holdout sales.")
            return super().__getitem__(key)

    def reader_spy(handle, **kwargs):
        reader = original_reader(handle, **kwargs)
        names = next(reader)
        yield names
        day_index = names.index("day_number")
        sales_index = names.index("sales")
        for row in reader:
            yield (
                GuardedRow(row, sales_index)
                if int(row[day_index]) > 1885 else row
            )

    monkeypatch.setattr(inventory.csv, "reader", reader_spy)
    histories, dates, digest = inventory.load_history(
        inputs["history_path"], inputs["selected_ids"],
        inputs["items"], inputs["forecast_dates"],
    )
    assert len(histories) == 20
    assert all(len(values) == 56 for values in histories.values())
    assert len(dates) == 56
    assert len(digest) == 64


def test_complete_synthetic_outputs_and_input_immutability(tmp_path):
    build_project(tmp_path)
    existing = [path for path in tmp_path.rglob("*") if path.is_file()]
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in existing
    }
    result = inventory.run_simulation(tmp_path)
    evaluation = tmp_path / "outputs/evaluation"
    assert result["status"] == "PASS"
    assert result["row_counts"] == {
        "daily_ledger": 10080,
        "metrics_by_product": 360,
        "policy_comparison": 9,
    }
    ledger = pd.read_csv(evaluation / inventory.OUTPUT_NAMES[0])
    products = pd.read_csv(evaluation / inventory.OUTPUT_NAMES[1])
    comparison = pd.read_csv(evaluation / inventory.OUTPUT_NAMES[2])
    assert tuple(ledger.columns) == inventory.LEDGER_COLUMNS
    assert tuple(products.columns) == inventory.PRODUCT_COLUMNS
    assert tuple(comparison.columns) == inventory.COMPARISON_COLUMNS
    assert ledger.groupby(["scenario_id", "policy", "id"]).size().eq(28).all()
    assert products["observations"].eq(28).all()
    assert not ledger.duplicated(["scenario_id", "policy", "id", "day_index"]).any()

    paired = products.pivot(
        index=["scenario_id", "id"], columns="policy", values="initial_inventory"
    )
    assert paired[inventory.POLICY_A].eq(paired[inventory.POLICY_B]).all()
    assert comparison["product_days"].eq(560).all()
    assert comparison["days"].eq(28).all()

    for path, digest in before.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    for name, digest in result["output_sha256"].items():
        assert inventory.sha256_file(evaluation / name) == digest


def test_zero_demand_fill_rate_is_undefined():
    rows = simulate([0] * 28, [0] * 28, initial=0)
    assert sum(row["fulfilled_units"] for row in rows) == 0
    assert sum(row["unmet_units"] for row in rows) == 0


@pytest.mark.parametrize("bad", [-1, 0.5, None, float("nan"), True])
def test_invalid_demand_rejected(bad):
    with pytest.raises(ValueError):
        simulate([bad], [1.0])


def test_zero_and_fractional_forecasts_remain_valid():
    rows = simulate([1, 0, 1], [0.0, 1.25, 2.5], initial=0)
    assert rows[0]["planned_units_over_protection"] == pytest.approx(1.25)
    assert rows[0]["order_up_to"] == 2
    assert rows[0]["order_quantity"] == 2