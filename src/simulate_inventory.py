"""Paired deterministic inventory simulation using saved holdout artifacts.

Run:
    python -m src.simulate_inventory

Policies:
    A: Frozen historical weekday means from d_1830..d_1885.
    B: Existing frozen XGBoost forecasts for d_1886..d_1913.

No forecasting, model fitting, protected scoring, or policy tuning occurs.
Recorded sales are a demand proxy, not verified unconstrained demand.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import statistics
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STORE = "CA_1"
DEPARTMENT = "FOODS_3"
PRODUCT_COUNT = 20
TRAINING_CUTOFF = 1885
HISTORY_FIRST = 1830
HISTORY_DAYS = 56
HORIZON = 28
SAFETY_MULTIPLIER = 1.645

POLICY_A = "historical_weekday_56"
POLICY_B = "frozen_xgboost"
POLICIES = (POLICY_A, POLICY_B)

HOLDING_COST = "0.01"
SHORTAGE_COSTS = ("0.10", "0.50", "1.00")
LEAD_TIMES = (1, 2, 3)

FROZEN_COLUMNS = (
    "id", "item_id", "model", "forecast_origin_day", "horizon",
    "d", "date", "raw_predicted_sales", "predicted_sales",
)
SCORED_COLUMNS = (
    "id", "item_id", "model", "forecast_origin_day", "horizon",
    "d", "date", "actual_sales",
    "raw_predicted_sales", "predicted_sales",
)
HISTORICAL_COLUMNS = (
    "id", "item_id", "store_id", "dept_id", "d",
    "day_number", "date", "sales", "split",
)

LEDGER_COLUMNS = (
    "scenario_id", "policy", "id", "item_id", "day_index", "d", "date",
    "lead_time", "holding_cost_per_unit_day", "shortage_cost_per_unit",
    "initial_inventory", "start_on_hand", "outstanding_start",
    "arrivals", "available_inventory", "outstanding_before_order",
    "inventory_position", "protection_days", "planned_units_over_protection",
    "safety_stock_units", "order_up_to", "order_quantity",
    "order_arrival_day", "demand_proxy_units", "fulfilled_units",
    "unmet_units", "end_on_hand", "outstanding_end",
    "holding_cost", "shortage_cost", "total_cost",
)

PRODUCT_COLUMNS = (
    "scenario_id", "policy", "id", "item_id", "lead_time",
    "holding_cost_per_unit_day", "shortage_cost_per_unit",
    "observations", "initial_inventory", "demand_proxy_units",
    "fulfilled_units", "unmet_units", "fill_rate_pct", "stockout_days",
    "inventory_unit_days", "average_end_on_hand",
    "ordered_units", "received_units", "final_inventory",
    "holding_cost", "shortage_cost", "total_simulated_cost",
)

PORTFOLIO_FIELDS = (
    "fulfilled_units", "unmet_units", "fill_rate_pct",
    "stockout_product_days", "days_with_any_stockout",
    "average_daily_total_inventory", "average_inventory_per_product_day",
    "initial_inventory", "final_inventory",
    "ordered_units", "received_units",
    "holding_cost", "shortage_cost", "total_simulated_cost",
)

COMPARISON_COLUMNS = (
    "scenario_id", "lead_time", "holding_cost_per_unit_day",
    "shortage_cost_per_unit", "products", "days", "product_days",
    "demand_proxy_units",
    *(f"policy_a_{name}" for name in PORTFOLIO_FIELDS),
    *(f"policy_b_{name}" for name in PORTFOLIO_FIELDS),
    "difference_fill_rate_percentage_points",
    "difference_unmet_units",
    "difference_average_daily_total_inventory",
    "difference_total_cost",
    "cost_reduction_pct_vs_policy_a",
)

OUTPUT_NAMES = (
    "inventory_daily_ledger.csv",
    "inventory_metrics_by_product.csv",
    "inventory_policy_comparison.csv",
    "inventory_run_metadata.json",
)

NUMBER_PATTERN = re.compile(
    r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def reject_json_constant(token: str) -> None:
    raise ValueError(f"Invalid JSON constant: {token}")


def units(value: Any, context: str, text: bool = False) -> int:
    require(
        not isinstance(value, bool),
        f"{context}: booleans are invalid.",
    )
    if isinstance(value, str):
        require(
            text and NUMBER_PATTERN.fullmatch(value) is not None,
            f"{context}: malformed unit value.",
        )
    else:
        require(
            isinstance(value, (Real, Decimal)),
            f"{context}: expected numeric units.",
        )
    try:
        exact = (
            Decimal(value)
            if isinstance(value, (str, Decimal))
            else Decimal(str(value))
        )
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"{context}: invalid units.") from exc
    require(
        exact.is_finite() and exact >= 0
        and exact == exact.to_integral_value(),
        f"{context}: units must be finite nonnegative integers.",
    )
    result = int(exact)
    require(
        result <= 2**53,
        f"{context}: units exceed exact float-based feature precision.",
    )
    return result


def nonnegative_number(value: Any, context: str, text: bool = False) -> float:
    require(not isinstance(value, bool), f"{context}: boolean is invalid.")
    if isinstance(value, str):
        require(
            text and NUMBER_PATTERN.fullmatch(value) is not None,
            f"{context}: malformed number.",
        )
    else:
        require(isinstance(value, (Real, Decimal)), f"{context}: expected a number.")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{context}: invalid number.") from exc
    require(
        math.isfinite(result) and result >= 0,
        f"{context}: expected a finite nonnegative number.",
    )
    return result


def signed_number(value: str, context: str) -> float:
    require(
        isinstance(value, str) and NUMBER_PATTERN.fullmatch(value) is not None,
        f"{context}: malformed number.",
    )
    result = float(value)
    require(math.isfinite(result), f"{context}: non-finite number.")
    return result


def parsed_date(value: str, context: str) -> date:
    require(
        re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value) is not None,
        f"{context}: expected YYYY-MM-DD.",
    )
    return date.fromisoformat(value)


def csv_header(
    reader: Any, required: tuple[str, ...], source: str, exact: bool
) -> list[str]:
    names = next(reader, [])
    require(names and all(name.strip() for name in names), f"{source}: blank header.")
    require(len(names) == len(set(names)), f"{source}: duplicate headers.")
    require(
        set(required).issubset(names),
        f"{source}: missing required columns.",
    )
    if exact:
        require(tuple(names) == required, f"{source}: unexpected schema/order.")
    return names


def read_csv_payload(
    payload: bytes, required: tuple[str, ...], source: str
) -> list[dict[str, str]]:
    reader = csv.reader(
        io.StringIO(payload.decode("utf-8-sig"), newline=""),
        strict=True,
    )
    names = csv_header(reader, required, source, exact=True)
    records = []
    for number, row in enumerate(reader, start=1):
        require(
            len(row) == len(names),
            f"{source}: malformed record {number}.",
        )
        records.append(dict(zip(names, row)))
    return records


def scenarios() -> tuple[dict[str, Any], ...]:
    """The complete predeclared grid; no runtime tuning parameters."""
    return tuple(
        {
            "scenario_id": f"L{lead}_P{int(Decimal(penalty) * 100):03d}",
            "lead_time": lead,
            "holding_cost": HOLDING_COST,
            "shortage_cost": penalty,
        }
        for lead in LEAD_TIMES
        for penalty in SHORTAGE_COSTS
    )


def safety_stock(sigma: float, protection_days: int) -> int:
    sigma = nonnegative_number(sigma, "Historical standard deviation")
    require(
        isinstance(protection_days, Integral)
        and not isinstance(protection_days, bool)
        and protection_days >= 1,
        "Protection days must be positive integers.",
    )
    return math.ceil(
        SAFETY_MULTIPLIER * sigma * math.sqrt(protection_days)
    )


def initial_inventory(
    historical_estimates: tuple[float, ...], sigma: float, lead_time: int
) -> int:
    require(
        isinstance(lead_time, Integral)
        and not isinstance(lead_time, bool)
        and lead_time >= 1,
        "Lead time must be a positive integer.",
    )
    require(
        len(historical_estimates) >= lead_time + 1,
        "Insufficient planning estimates for initialization.",
    )
    return (
        math.ceil(math.fsum(historical_estimates[:lead_time + 1]))
        + safety_stock(sigma, lead_time + 1)
    )


def prepare_plans(
    histories: dict[str, list[int]],
    historical_dates: dict[int, date],
    forecasts: dict[str, tuple[float, ...]],
    forecast_dates: tuple[date, ...],
) -> tuple[dict[str, dict[str, tuple[float, ...]]], dict[str, dict[str, Any]]]:
    require(
        set(histories) == set(forecasts),
        "Historical/forecast population mismatch.",
    )
    require(len(forecast_dates) == HORIZON, "Expected 28 forecast dates.")
    require(
        sorted(historical_dates) == list(range(HISTORY_FIRST, TRAINING_CUTOFF + 1)),
        "Historical dates must cover d_1830..d_1885.",
    )
    require(
        historical_dates[TRAINING_CUTOFF] + timedelta(days=1) == forecast_dates[0],
        "Historical and holdout dates are not adjacent.",
    )
    plans = {POLICY_A: {}, POLICY_B: {}}
    statistics_by_product = {}
    for series_id in sorted(histories):
        values = [
            units(value, f"{series_id} history")
            for value in histories[series_id]
        ]
        require(len(values) == HISTORY_DAYS, f"{series_id}: expected 56 days.")
        weekday_means = []
        for weekday in range(7):
            observations = [
                values[day - HISTORY_FIRST]
                for day in range(HISTORY_FIRST, TRAINING_CUTOFF + 1)
                if historical_dates[day].weekday() == weekday
            ]
            require(len(observations) == 8, "Expected eight historical weekday values.")
            weekday_means.append(math.fsum(observations) / 8)

        historical_plan = tuple(
            weekday_means[target.weekday()] for target in forecast_dates
        )
        frozen_plan = tuple(
            nonnegative_number(value, f"{series_id} frozen prediction")
            for value in forecasts[series_id]
        )
        require(len(frozen_plan) == HORIZON, "Expected 28 frozen forecasts.")
        sigma = statistics.stdev(values)
        plans[POLICY_A][series_id] = historical_plan
        plans[POLICY_B][series_id] = frozen_plan
        statistics_by_product[series_id] = {
            "weekday_means_monday_to_sunday": weekday_means,
            "sample_daily_standard_deviation": sigma,
            "historical_observation_count": HISTORY_DAYS,
            "initial_inventory_by_lead_time": {
                str(lead): initial_inventory(historical_plan, sigma, lead)
                for lead in LEAD_TIMES
            },
        }
    return plans, statistics_by_product


def simulate_product(
    demand: list[int] | tuple[int, ...],
    estimates: list[float] | tuple[float, ...],
    starting_inventory: int,
    sigma: float,
    lead_time: int,
    holding_cost: str,
    shortage_cost: str,
) -> list[dict[str, Any]]:
    """Simulate morning review, receipts before demand, and lost sales.

    At morning t, today's order arrives at morning t+L. The order placed
    at tomorrow's review arrives at t+L+1. Protection therefore includes
    today's demand through day t+L: L+1 days, truncated at the horizon.

    The position is current available stock plus prior outstanding orders.
    Today's order cannot remedy a current-day shortage.
    """
    actual = tuple(units(value, "Demand proxy") for value in demand)
    planned = tuple(
        nonnegative_number(value, "Planning estimate")
        for value in estimates
    )
    require(actual and len(actual) == len(planned), "Demand/estimate length mismatch.")
    require(
        isinstance(lead_time, Integral)
        and not isinstance(lead_time, bool)
        and lead_time >= 1,
        "Lead time must be a positive integer.",
    )
    on_hand = units(starting_inventory, "Initial inventory")
    initial = on_hand
    sigma = nonnegative_number(sigma, "Historical standard deviation")
    h = Decimal(holding_cost)
    p = Decimal(shortage_cost)
    require(h.is_finite() and h >= 0 and p.is_finite() and p >= 0, "Invalid costs.")
    horizon = len(actual)
    pipeline: dict[int, int] = {}
    received_total = 0
    fulfilled_total = 0
    rows = []

    for day in range(1, horizon + 1):
        start_on_hand = on_hand
        outstanding_start = sum(pipeline.values())
        arrivals = pipeline.pop(day, 0)
        received_total += arrivals
        available = start_on_hand + arrivals
        outstanding_before_order = sum(pipeline.values())
        position = available + outstanding_before_order

        protection = min(lead_time + 1, horizon - day + 1)
        planned_units = math.fsum(
            planned[day - 1:day - 1 + protection]
        )
        buffer = safety_stock(sigma, protection)
        target = math.ceil(planned_units) + buffer
        order = (
            max(0, target - position)
            if day + lead_time <= horizon
            else 0
        )
        arrival_day = day + lead_time if order else 0
        if order:
            pipeline[arrival_day] = pipeline.get(arrival_day, 0) + order

        fulfilled = min(actual[day - 1], available)
        unmet = actual[day - 1] - fulfilled
        on_hand = available - fulfilled
        fulfilled_total += fulfilled
        outstanding_end = sum(pipeline.values())

        holding = h * on_hand
        shortage = p * unmet
        require(
            actual[day - 1] == fulfilled + unmet
            and initial + received_total == on_hand + fulfilled_total
            and outstanding_start - arrivals + order == outstanding_end
            and on_hand >= 0
            and order >= 0
            and all(day < due <= horizon for due in pipeline),
            "Inventory conservation or arrival-boundary failure.",
        )
        rows.append({
            "day_index": day,
            "initial_inventory": initial,
            "start_on_hand": start_on_hand,
            "outstanding_start": outstanding_start,
            "arrivals": arrivals,
            "available_inventory": available,
            "outstanding_before_order": outstanding_before_order,
            "inventory_position": position,
            "protection_days": protection,
            "planned_units_over_protection": planned_units,
            "safety_stock_units": buffer,
            "order_up_to": target,
            "order_quantity": order,
            # Zero means no order was placed on this day.
            "order_arrival_day": arrival_day,
            "demand_proxy_units": actual[day - 1],
            "fulfilled_units": fulfilled,
            "unmet_units": unmet,
            "end_on_hand": on_hand,
            "outstanding_end": outstanding_end,
            "holding_cost": float(holding),
            "shortage_cost": float(shortage),
            "total_cost": float(holding + shortage),
        })
    require(not pipeline, "Orders remain beyond the finite horizon.")
    return rows


def load_frozen_inputs(root: Path) -> dict[str, Any]:
    directory = root / "outputs/evaluation"
    metadata_path = directory / "holdout_run_metadata.json"
    frozen_path = directory / "holdout_predictions_frozen.csv"
    scored_path = directory / "holdout_predictions_scored.csv"
    history_path = root / "data/processed/selected_sales.csv"
    for path in (metadata_path, frozen_path, scored_path, history_path):
        require(path.is_file(), f"Required input missing: {path}")

    metadata_payload = metadata_path.read_bytes()
    metadata = json.loads(
        metadata_payload.decode("utf-8-sig"),
        parse_constant=reject_json_constant,
    )
    require(
        isinstance(metadata, dict)
        and metadata.get("status") == "EVALUATED"
        and metadata.get("model") == "xgboost"
        and metadata.get("selected_product_count") == PRODUCT_COUNT
        and metadata.get("training_cutoff") == TRAINING_CUTOFF
        and metadata.get("forecast_origin") == TRAINING_CUTOFF
        and metadata.get("holdout_boundaries") == [1886, 1913]
        and metadata.get("forecast_horizon") == HORIZON,
        "Holdout metadata is incompatible or scoring is incomplete.",
    )
    selected_ids = metadata.get("selected_product_ids_in_rank_order")
    require(
        isinstance(selected_ids, list)
        and len(selected_ids) == PRODUCT_COUNT
        and len(set(selected_ids)) == PRODUCT_COUNT
        and all(isinstance(value, str) and value.strip() for value in selected_ids),
        "Invalid selected population in holdout metadata.",
    )
    frozen_payload = frozen_path.read_bytes()
    frozen_hash = metadata.get("frozen_prediction_sha256")
    require(
        isinstance(frozen_hash, str)
        and sha256_bytes(frozen_payload) == frozen_hash.lower(),
        "Frozen prediction SHA256 mismatch.",
    )
    scored_hash = metadata.get("scored_prediction_sha256")
    require(
        isinstance(scored_hash, str)
        and re.fullmatch(r"[0-9a-fA-F]{64}", scored_hash) is not None,
        "Saved scored-artifact SHA256 is missing or invalid.",
    )
    records = read_csv_payload(
        frozen_payload, FROZEN_COLUMNS, frozen_path.name
    )
    require(len(records) == PRODUCT_COUNT * HORIZON, "Expected 560 frozen rows.")
    by_key = {}
    items = {}
    dates = {}
    forecasts = {series_id: {} for series_id in selected_ids}
    for record in records:
        series_id = record["id"]
        require(series_id in forecasts, "Unexpected frozen product.")
        horizon = units(record["horizon"], "Frozen horizon", text=True)
        require(1 <= horizon <= HORIZON, "Frozen horizon outside 1..28.")
        key = (series_id, horizon)
        require(key not in by_key, "Duplicate frozen prediction key.")
        target = TRAINING_CUTOFF + horizon
        target_date = parsed_date(record["date"], "Frozen date")
        require(
            record["model"] == "xgboost"
            and record["forecast_origin_day"] == "d_1885"
            and record["d"] == f"d_{target}"
            and bool(record["item_id"].strip()),
            "Frozen identity or chronological alignment failure.",
        )
        raw = signed_number(record["raw_predicted_sales"], "Raw prediction")
        prediction = nonnegative_number(
            record["predicted_sales"], "Prediction", text=True
        )
        require(prediction == max(0.0, raw), "Frozen clipping mismatch.")
        require(
            series_id not in items or items[series_id] == record["item_id"],
            "Inconsistent item identifier.",
        )
        require(
            horizon not in dates or dates[horizon] == target_date,
            "Inconsistent frozen calendar mapping.",
        )
        items[series_id] = record["item_id"]
        dates[horizon] = target_date
        forecasts[series_id][horizon] = prediction
        by_key[key] = {
            **record,
            "horizon": horizon,
            "raw_predicted_sales": raw,
            "predicted_sales": prediction,
        }
    require(
        len(set(items.values())) == PRODUCT_COUNT,
        "Selected product-store items must be unique.",
    )
    for series_id in selected_ids:
        require(
            sorted(forecasts[series_id]) == list(range(1, HORIZON + 1)),
            f"{series_id}: incomplete forecast coverage.",
        )
    forecast_dates = tuple(dates[day] for day in range(1, HORIZON + 1))
    require(
        all(b - a == timedelta(days=1) for a, b in zip(forecast_dates, forecast_dates[1:])),
        "Frozen dates are not consecutive.",
    )
    return {
        "selected_ids": selected_ids,
        "items": items,
        "forecast_dates": forecast_dates,
        "forecasts": {
            series_id: tuple(forecasts[series_id][day] for day in range(1, 29))
            for series_id in selected_ids
        },
        "frozen_records": by_key,
        "history_path": history_path,
        "scored_path": scored_path,
        "scored_hash": scored_hash.lower(),
        "artifact_hashes": {
            metadata_path.name: sha256_bytes(metadata_payload),
            frozen_path.name: sha256_bytes(frozen_payload),
            scored_path.name: scored_hash.lower(),
        },
    }


def load_history(
    path: Path,
    selected_ids: list[str],
    items: dict[str, str],
    forecast_dates: tuple[date, ...],
) -> tuple[dict[str, list[int]], dict[int, date], str]:
    observed = {series_id: {} for series_id in selected_ids}
    dates = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        names = csv_header(reader, HISTORICAL_COLUMNS, path.name, exact=False)
        positions = {name: names.index(name) for name in HISTORICAL_COLUMNS}
        for number, row in enumerate(reader, start=1):
            require(len(row) == len(names), f"History record {number}: malformed.")
            day = units(row[positions["day_number"]], "History day", text=True)
            # No sales field outside the approved historical window is accessed.
            if not HISTORY_FIRST <= day <= TRAINING_CUTOFF:
                continue
            series_id = row[positions["id"]]
            require(series_id in observed, "Unexpected historical product.")
            require(day not in observed[series_id], "Duplicate historical product-day.")
            expected_split = "train" if day <= 1857 else "validation"
            require(
                row[positions["item_id"]] == items[series_id]
                and row[positions["store_id"]] == STORE
                and row[positions["dept_id"]] == DEPARTMENT
                and row[positions["d"]] == f"d_{day}"
                and row[positions["split"]] == expected_split,
                "Historical identifiers or boundaries differ.",
            )
            current_date = parsed_date(row[positions["date"]], "Historical date")
            require(
                day not in dates or dates[day] == current_date,
                "Historical dates differ across products.",
            )
            dates[day] = current_date
            observed[series_id][day] = units(
                row[positions["sales"]], "Historical sales", text=True
            )
    expected_days = list(range(HISTORY_FIRST, TRAINING_CUTOFF + 1))
    histories = {}
    hasher = hashlib.sha256()
    for series_id in sorted(selected_ids):
        require(
            sorted(observed[series_id]) == expected_days,
            f"{series_id}: missing historical days.",
        )
        histories[series_id] = [observed[series_id][day] for day in expected_days]
        for day in expected_days:
            hasher.update(canonical_bytes([
                series_id, day, dates[day].isoformat(), observed[series_id][day]
            ]) + b"\n")
    require(
        sorted(dates) == expected_days
        and all(
            dates[day] - dates[day - 1] == timedelta(days=1)
            for day in expected_days[1:]
        )
        and dates[TRAINING_CUTOFF] + timedelta(days=1) == forecast_dates[0],
        "Historical and holdout calendar mapping differs.",
    )
    return histories, dates, hasher.hexdigest()


def load_saved_actuals(inputs: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    path = inputs["scored_path"]
    payload = path.read_bytes()
    require(
        sha256_bytes(payload) == inputs["scored_hash"],
        "Saved scored-artifact SHA256 mismatch.",
    )
    records = read_csv_payload(payload, SCORED_COLUMNS, path.name)
    require(len(records) == PRODUCT_COUNT * HORIZON, "Expected 560 saved actual rows.")
    actuals = {series_id: {} for series_id in inputs["selected_ids"]}
    seen = set()
    for record in records:
        series_id = record["id"]
        horizon = units(record["horizon"], "Scored horizon", text=True)
        key = (series_id, horizon)
        require(
            key in inputs["frozen_records"] and key not in seen,
            "Saved actual prediction keys differ or are duplicated.",
        )
        seen.add(key)
        frozen = inputs["frozen_records"][key]
        for field in ("id", "item_id", "model", "forecast_origin_day", "d", "date"):
            require(record[field] == frozen[field], f"Saved actual mismatch: {field}.")
        require(
            signed_number(record["raw_predicted_sales"], "Saved raw prediction")
            == frozen["raw_predicted_sales"]
            and nonnegative_number(
                record["predicted_sales"], "Saved prediction", text=True
            ) == frozen["predicted_sales"],
            "Scored artifact does not preserve frozen predictions.",
        )
        actuals[series_id][horizon] = units(
            record["actual_sales"], "Recorded demand proxy", text=True
        )
    return {
        series_id: tuple(actuals[series_id][day] for day in range(1, 29))
        for series_id in inputs["selected_ids"]
    }


def build_configuration(
    inputs: dict[str, Any],
    plans: dict[str, dict[str, tuple[float, ...]]],
    historical_statistics: dict[str, dict[str, Any]],
    history_hash: str,
) -> dict[str, Any]:
    return {
        "dataset": "Walmart M5 Forecasting Accuracy",
        "scope": {"store_id": STORE, "dept_id": DEPARTMENT},
        "selected_ids_in_rank_order": inputs["selected_ids"],
        "selected_product_count": PRODUCT_COUNT,
        "historical_window": [HISTORY_FIRST, TRAINING_CUTOFF],
        "simulation_window": [1886, 1913],
        "simulation_days": HORIZON,
        "policy_definitions": {
            POLICY_A: "Frozen means of eight observations per weekday in the final 56 historical days.",
            POLICY_B: "Existing nonnegative, unrounded fixed-origin XGBoost predictions.",
        },
        "planning_vectors": {
            policy: {
                series_id: list(values)
                for series_id, values in sorted(product_plans.items())
            }
            for policy, product_plans in plans.items()
        },
        "historical_statistics": historical_statistics,
        "scenarios": list(scenarios()),
        "safety_stock": {
            "multiplier": SAFETY_MULTIPLIER,
            "standard_deviation": "Sample daily sales SD over 56 days; ddof=1",
            "formula": "ceil(1.645 * sigma * sqrt(protection_days))",
            "service_level_guarantee": False,
        },
        "initial_inventory": (
            "ceil(sum of Policy A estimates for the first L+1 days)) "
            "+ safety_stock(L+1); same for both policies; initial pipeline zero."
        ),
        "event_sequence": [
            "Receive today's scheduled orders.",
            "Review available inventory and remaining outstanding orders.",
            "Place a nonnegative integer order.",
            "Realize recorded sales as a demand proxy.",
            "Fulfil available units and lose unmet units.",
            "Record end-of-day stock and illustrative costs.",
        ],
        "protection_period": (
            "Morning review: today's order arrives before demand at t+L; "
            "tomorrow's order arrives at t+L+1. Protect t..t+L, L+1 days."
        ),
        "inventory_position": "Available inventory plus outstanding orders before today's order.",
        "order_up_to": "ceil(sum frozen estimates over protection window) + common safety stock.",
        "finite_horizon": (
            "Truncate protection at day 28; suppress orders arriving after day 28; "
            "no forecasts beyond day 28; no salvage credit or disposal charge."
        ),
        "operating_assumptions": [
            "Lost sales; no backorders.",
            "Unlimited replenishment supply and capacity.",
            "No minimum order size, fixed order cost, spoilage, or substitution.",
            "Integer stock and orders; fractional forecast means remain unrounded.",
        ],
        "cost_units": (
            "Illustrative cost units; holding per unit per day, "
            "shortage per unmet unit. Not observed Walmart economics."
        ),
        "source_artifacts": {
            "frozen": "outputs/evaluation/holdout_predictions_frozen.csv",
            "saved_actuals": "outputs/evaluation/holdout_predictions_scored.csv",
            "holdout_metadata": "outputs/evaluation/holdout_run_metadata.json",
            "history": "data/processed/selected_sales.csv, d_1830..d_1885 sales only",
        },
        "artifact_sha256": inputs["artifact_hashes"],
        "historical_logical_sha256": history_hash,
        "configuration_uses_simulation_outcomes": False,
        "policy_or_scenario_tuning_permitted": False,
    }


def simulate_all(
    inputs: dict[str, Any],
    plans: dict[str, dict[str, tuple[float, ...]]],
    historical_statistics: dict[str, dict[str, Any]],
    actuals: dict[str, tuple[int, ...]],
) -> pd.DataFrame:
    records = []
    for scenario in scenarios():
        for policy in POLICIES:
            for series_id in sorted(inputs["selected_ids"]):
                statistics_record = historical_statistics[series_id]
                starting = statistics_record["initial_inventory_by_lead_time"][
                    str(scenario["lead_time"])
                ]
                rows = simulate_product(
                    actuals[series_id],
                    plans[policy][series_id],
                    starting,
                    statistics_record["sample_daily_standard_deviation"],
                    scenario["lead_time"],
                    scenario["holding_cost"],
                    scenario["shortage_cost"],
                )
                for row in rows:
                    day = row["day_index"]
                    records.append({
                        "scenario_id": scenario["scenario_id"],
                        "policy": policy,
                        "id": series_id,
                        "item_id": inputs["items"][series_id],
                        "d": f"d_{TRAINING_CUTOFF + day}",
                        "date": inputs["forecast_dates"][day - 1].isoformat(),
                        "lead_time": scenario["lead_time"],
                        "holding_cost_per_unit_day": float(scenario["holding_cost"]),
                        "shortage_cost_per_unit": float(scenario["shortage_cost"]),
                        **row,
                    })
    return pd.DataFrame(records, columns=LEDGER_COLUMNS)


def verify_ledger(
    ledger: pd.DataFrame,
    inputs: dict[str, Any],
    plans: dict[str, dict[str, tuple[float, ...]]],
    historical_statistics: dict[str, dict[str, Any]],
    actuals: dict[str, tuple[int, ...]],
) -> None:
    expected_rows = len(scenarios()) * len(POLICIES) * PRODUCT_COUNT * HORIZON
    require(
        tuple(ledger.columns) == LEDGER_COLUMNS
        and len(ledger) == expected_rows
        and not ledger.isna().any().any()
        and not ledger.duplicated(
            ["scenario_id", "policy", "id", "day_index"]
        ).any(),
        "Ledger schema, count, completeness, or uniqueness failure.",
    )
    expected_groups = {
        (scenario["scenario_id"], policy, series_id)
        for scenario in scenarios()
        for policy in POLICIES
        for series_id in inputs["selected_ids"]
    }
    grouped = ledger.groupby(["scenario_id", "policy", "id"], sort=True)
    require(set(grouped.groups) == expected_groups, "Ledger group coverage differs.")
    scenario_map = {scenario["scenario_id"]: scenario for scenario in scenarios()}

    for (scenario_id, policy, series_id), group in grouped:
        group = group.sort_values("day_index")
        require(group["day_index"].tolist() == list(range(1, 29)), "Ledger days differ.")
        scenario = scenario_map[scenario_id]
        lead = scenario["lead_time"]
        h, p = Decimal(scenario["holding_cost"]), Decimal(scenario["shortage_cost"])
        sigma = historical_statistics[series_id]["sample_daily_standard_deviation"]
        initial = historical_statistics[series_id]["initial_inventory_by_lead_time"][str(lead)]
        pipeline = {}
        previous_end = initial
        received = fulfilled_total = 0

        for row in group.itertuples(index=False):
            day = row.day_index
            arrivals = pipeline.pop(day, 0)
            remaining = sum(pipeline.values())
            protection = min(lead + 1, HORIZON - day + 1)
            planning = math.fsum(plans[policy][series_id][day - 1:day - 1 + protection])
            buffer = safety_stock(sigma, protection)
            position = previous_end + arrivals + remaining
            target = math.ceil(planning) + buffer
            order = max(0, target - position) if day + lead <= HORIZON else 0

            require(
                row.initial_inventory == initial
                and row.start_on_hand == previous_end
                and row.arrivals == arrivals
                and row.outstanding_before_order == remaining
                and row.available_inventory == previous_end + arrivals
                and row.inventory_position == position
                and row.protection_days == protection
                and row.planned_units_over_protection == planning
                and row.safety_stock_units == buffer
                and row.order_up_to == target
                and row.order_quantity == order
                and row.order_arrival_day == (day + lead if order else 0),
                "Ledger order timing/position/target reconciliation failed.",
            )
            require(
                row.demand_proxy_units == actuals[series_id][day - 1]
                and row.fulfilled_units == min(
                    row.available_inventory, row.demand_proxy_units
                )
                and row.unmet_units == row.demand_proxy_units - row.fulfilled_units
                and row.end_on_hand == row.available_inventory - row.fulfilled_units,
                "Ledger demand/fulfilment accounting failed.",
            )
            if order:
                pipeline[day + lead] = pipeline.get(day + lead, 0) + order
            require(
                row.outstanding_start - arrivals + order == row.outstanding_end
                and row.outstanding_end == sum(pipeline.values()),
                "Pipeline conservation failed.",
            )
            received += arrivals
            fulfilled_total += row.fulfilled_units
            require(
                initial + received == row.end_on_hand + fulfilled_total,
                "On-hand conservation failed.",
            )
            require(
                row.holding_cost == float(h * row.end_on_hand)
                and row.shortage_cost == float(p * row.unmet_units)
                and row.total_cost == float(
                    h * row.end_on_hand + p * row.unmet_units
                ),
                "Ledger cost reconciliation failed.",
            )
            previous_end = row.end_on_hand
        require(not pipeline, "Terminal pipeline must be empty.")


def summarize(ledger: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    product_records = []
    for (scenario_id, policy, series_id), group in ledger.groupby(
        ["scenario_id", "policy", "id"], sort=True
    ):
        group = group.sort_values("day_index")
        first, last = group.iloc[0], group.iloc[-1]
        demand = sum(group["demand_proxy_units"].tolist())
        fulfilled = sum(group["fulfilled_units"].tolist())
        unmet = sum(group["unmet_units"].tolist())
        inventory_days = sum(group["end_on_hand"].tolist())
        h = Decimal(str(first["holding_cost_per_unit_day"]))
        p = Decimal(str(first["shortage_cost_per_unit"]))
        holding = h * inventory_days
        shortage = p * unmet
        product_records.append({
            "scenario_id": scenario_id,
            "policy": policy,
            "id": series_id,
            "item_id": first["item_id"],
            "lead_time": int(first["lead_time"]),
            "holding_cost_per_unit_day": float(h),
            "shortage_cost_per_unit": float(p),
            "observations": len(group),
            "initial_inventory": int(first["initial_inventory"]),
            "demand_proxy_units": demand,
            "fulfilled_units": fulfilled,
            "unmet_units": unmet,
            "fill_rate_pct": 100.0 * fulfilled / demand if demand else None,
            "stockout_days": int(group["unmet_units"].gt(0).sum()),
            "inventory_unit_days": inventory_days,
            "average_end_on_hand": inventory_days / HORIZON,
            "ordered_units": sum(group["order_quantity"].tolist()),
            "received_units": sum(group["arrivals"].tolist()),
            "final_inventory": int(last["end_on_hand"]),
            "holding_cost": float(holding),
            "shortage_cost": float(shortage),
            "total_simulated_cost": float(holding + shortage),
        })
    products = pd.DataFrame(product_records, columns=PRODUCT_COLUMNS, dtype=object)
    comparison_records = []
    for scenario in scenarios():
        scenario_id = scenario["scenario_id"]
        portfolio = {}
        for policy in POLICIES:
            records = [
                row for row in product_records
                if row["scenario_id"] == scenario_id and row["policy"] == policy
            ]
            require(len(records) == PRODUCT_COUNT, "Product metric coverage differs.")
            demand = sum(row["demand_proxy_units"] for row in records)
            fulfilled = sum(row["fulfilled_units"] for row in records)
            unmet = sum(row["unmet_units"] for row in records)
            inventory_days = sum(row["inventory_unit_days"] for row in records)
            daily = ledger.loc[
                ledger["scenario_id"].eq(scenario_id) & ledger["policy"].eq(policy)
            ].groupby("day_index")["unmet_units"].sum()
            h, p = Decimal(scenario["holding_cost"]), Decimal(scenario["shortage_cost"])
            portfolio[policy] = {
                "demand_proxy_units": demand,
                "fulfilled_units": fulfilled,
                "unmet_units": unmet,
                "fill_rate_pct": 100.0 * fulfilled / demand if demand else None,
                "stockout_product_days": sum(row["stockout_days"] for row in records),
                "days_with_any_stockout": int(daily.gt(0).sum()),
                "average_daily_total_inventory": inventory_days / HORIZON,
                "average_inventory_per_product_day": inventory_days / (PRODUCT_COUNT * HORIZON),
                "initial_inventory": sum(row["initial_inventory"] for row in records),
                "final_inventory": sum(row["final_inventory"] for row in records),
                "ordered_units": sum(row["ordered_units"] for row in records),
                "received_units": sum(row["received_units"] for row in records),
                "holding_cost": float(h * inventory_days),
                "shortage_cost": float(p * unmet),
                "total_simulated_cost": float(h * inventory_days + p * unmet),
            }
        a, b = portfolio[POLICY_A], portfolio[POLICY_B]
        require(
            a["demand_proxy_units"] == b["demand_proxy_units"]
            and a["initial_inventory"] == b["initial_inventory"],
            "Paired demand or initial inventory differs.",
        )
        comparison_records.append({
            "scenario_id": scenario_id,
            "lead_time": scenario["lead_time"],
            "holding_cost_per_unit_day": float(scenario["holding_cost"]),
            "shortage_cost_per_unit": float(scenario["shortage_cost"]),
            "products": PRODUCT_COUNT,
            "days": HORIZON,
            "product_days": PRODUCT_COUNT * HORIZON,
            "demand_proxy_units": a["demand_proxy_units"],
            **{f"policy_a_{name}": a[name] for name in PORTFOLIO_FIELDS},
            **{f"policy_b_{name}": b[name] for name in PORTFOLIO_FIELDS},
            "difference_fill_rate_percentage_points": (
                None if a["fill_rate_pct"] is None
                else b["fill_rate_pct"] - a["fill_rate_pct"]
            ),
            "difference_unmet_units": b["unmet_units"] - a["unmet_units"],
            "difference_average_daily_total_inventory": (
                b["average_daily_total_inventory"] - a["average_daily_total_inventory"]
            ),
            "difference_total_cost": b["total_simulated_cost"] - a["total_simulated_cost"],
            "cost_reduction_pct_vs_policy_a": (
                100.0 * (a["total_simulated_cost"] - b["total_simulated_cost"])
                / a["total_simulated_cost"]
                if a["total_simulated_cost"] else None
            ),
        })
    comparison = pd.DataFrame(
        comparison_records, columns=COMPARISON_COLUMNS, dtype=object
    )
    require(
        len(products) == len(scenarios()) * len(POLICIES) * PRODUCT_COUNT
        and len(comparison) == len(scenarios()),
        "Summary row counts differ.",
    )
    return products, comparison


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(
        value, indent=2, ensure_ascii=False, allow_nan=False
    ).encode("utf-8") + b"\n"
    temporary = None
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".inventory-meta-", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def publish_results(
    directory: Path,
    ledger: pd.DataFrame,
    products: pd.DataFrame,
    comparison: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    frames = dict(zip(OUTPUT_NAMES[:3], (ledger, products, comparison)))
    previous = {
        directory / name: (
            (directory / name).read_bytes()
            if (directory / name).exists() else None
        )
        for name in OUTPUT_NAMES
    }
    replaced = []
    try:
        with tempfile.TemporaryDirectory(prefix=".inventory-stage-", dir=directory) as temp:
            stage = Path(temp)
            hashes = {}
            for name, frame in frames.items():
                path = stage / name
                frame.to_csv(path, index=False, na_rep="", lineterminator="\n")
                reread = pd.read_csv(path, dtype=str, keep_default_na=False)
                require(
                    tuple(reread.columns) == tuple(frame.columns)
                    and len(reread) == len(frame)
                    and path.stat().st_size > 0,
                    f"Staged output verification failed: {name}",
                )
                hashes[name] = sha256_file(path)
            completed = {
                **metadata,
                "status": "PASS",
                "completed_at_utc": timestamp(),
                "outputs_valid_for_this_run": True,
                "output_sha256": hashes,
                "row_counts": {
                    "daily_ledger": len(ledger),
                    "metrics_by_product": len(products),
                    "policy_comparison": len(comparison),
                },
                "integrity_checks": {
                    "inventory_conservation": True,
                    "pipeline_conservation": True,
                    "arrival_timing": True,
                    "paired_operating_constraints": True,
                    "cost_reconciliation": True,
                    "forecast_vectors_unchanged": True,
                    "all_nine_scenarios_reported": True,
                },
            }
            atomic_json(stage / OUTPUT_NAMES[3], completed)
            for name in OUTPUT_NAMES:
                (stage / name).replace(directory / name)
                replaced.append(directory / name)
            for name, digest in hashes.items():
                require(
                    sha256_file(directory / name) == digest,
                    f"Published output hash differs: {name}",
                )
            return completed
    except Exception as exc:
        rollback_errors = []
        for path in reversed(replaced):
            try:
                payload = previous[path]
                if payload is None:
                    path.unlink(missing_ok=True)
                else:
                    with tempfile.NamedTemporaryFile(
                        mode="wb", dir=directory, delete=False
                    ) as handle:
                        replacement = Path(handle.name)
                        handle.write(payload)
                    replacement.replace(path)
            except Exception as rollback_exc:
                rollback_errors.append(f"{path.name}: {rollback_exc}")
        suffix = (
            " Rollback issues: " + "; ".join(rollback_errors)
            if rollback_errors else ""
        )
        raise RuntimeError(f"Inventory publication failed: {exc}.{suffix}") from exc


def run_simulation(root: Path = ROOT) -> dict[str, Any]:
    root = Path(root)
    directory = root / "outputs/evaluation"
    inputs = load_frozen_inputs(root)
    histories, historical_dates, history_hash = load_history(
        inputs["history_path"], inputs["selected_ids"], inputs["items"],
        inputs["forecast_dates"],
    )
    plans, historical_statistics = prepare_plans(
        histories, historical_dates, inputs["forecasts"], inputs["forecast_dates"]
    )
    configuration = build_configuration(
        inputs, plans, historical_statistics, history_hash
    )
    configuration_hash = sha256_bytes(canonical_bytes(configuration))
    metadata = {
        "status": "CONFIG_FROZEN",
        "created_at_utc": timestamp(),
        "configuration": configuration,
        "configuration_sha256": configuration_hash,
        "configuration_frozen_before_outcomes": True,
        "outputs_valid_for_this_run": False,
        "limitations": [
            "Recorded sales are a common demand proxy, not verified unconstrained demand.",
            "Simulated unmet units are not measured historical lost sales.",
            "The selected high-volume products are not representative.",
            "Safety stock is a heuristic, not a calibrated service guarantee.",
            "Initial inventory and operational constraints are assumed.",
            "The finite horizon creates terminal effects.",
            "Costs are illustrative; results are not realized Walmart profit or savings.",
            "No model, forecast, policy parameter, or product selection is tuned.",
            "Cost scenarios reprice the same physical rule; shortage cost does not change orders.",
            "Average inventory means end-of-day inventory, not continuous-time inventory.",
            "No forecasting or protected holdout scoring workflow is invoked.",
            "Run one simulation process at a time; interrupted publication needs review.",
        ],
    }
    # Definitions, scenarios, initialization, and planning vectors are persisted
    # before saved holdout actuals are parsed or any outcomes are calculated.
    atomic_json(directory / OUTPUT_NAMES[3], metadata)

    try:
        actuals = load_saved_actuals(inputs)
        ledger = simulate_all(inputs, plans, historical_statistics, actuals)
        verify_ledger(ledger, inputs, plans, historical_statistics, actuals)
        require(
            sha256_bytes(canonical_bytes(configuration)) == configuration_hash,
            "Frozen configuration changed during simulation.",
        )
        require(
            configuration["planning_vectors"] == {
                policy: {
                    series_id: list(values)
                    for series_id, values in sorted(product_plans.items())
                }
                for policy, product_plans in plans.items()
            },
            "Planning vectors changed during simulation.",
        )
        products, comparison = summarize(ledger)
        for name, digest in inputs["artifact_hashes"].items():
            require(
                sha256_file(directory / name) == digest,
                f"Existing holdout artifact changed during simulation: {name}",
            )
        return publish_results(directory, ledger, products, comparison, metadata)
    except Exception as exc:
        failed = {
            **metadata,
            "status": "FAIL",
            "failed_at_utc": timestamp(),
            "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
            "outputs_valid_for_this_run": False,
        }
        atomic_json(directory / OUTPUT_NAMES[3], failed)
        raise


def main() -> int:
    try:
        result = run_simulation()
    except Exception as exc:
        print(
            f"Inventory simulation FAILED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print("Existing inventory CSVs may belong to an earlier run.", file=sys.stderr)
        return 1
    print("Inventory simulation: PASS")
    print("Policies: historical_weekday_56 and frozen_xgboost")
    print("Scenarios: 9; products: 20; days: 28")
    print(f"Ledger rows: {result['row_counts']['daily_ledger']}")
    print("Recorded sales were used as a demand proxy.")
    print("No forecasts, models, or policy parameters were updated.")
    for name in OUTPUT_NAMES:
        print(f"Output: {ROOT / 'outputs/evaluation' / name}")
    print("Exit code: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())