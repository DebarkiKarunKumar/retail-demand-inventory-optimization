"""Leakage-safe feature engineering and audit; no model training.

Run:
    python -m src.prepare_xgb_features

Only outputs/evaluation/xgb_feature_audit.json is published.

CSV readers transiently tokenize records. Validation/test sales fields are
never extracted, converted, retained, summarized, or used as labels/features.
Future selling-price rows outside historical training weeks are also excluded
before their price field is accessed.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
import tempfile
from collections.abc import Iterable, Mapping
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STORE = "CA_1"
DEPARTMENT = "FOODS_3"
PRODUCT_COUNT = 20
TRAIN_END = 1857
VALIDATION_END = 1885
HISTORY_END = 1913
HORIZON = 28
MIN_ORIGIN = 57
MISSING_EVENT = "None"

SALES_COLUMNS = (
    "id", "item_id", "store_id", "dept_id", "d",
    "day_number", "date", "sales", "split",
)
MANIFEST_COLUMNS = (
    "selection_rank", "id", "item_id", "store_id", "dept_id",
)
CALENDAR_COLUMNS = (
    "d", "date", "wm_yr_wk", "snap_CA",
    "event_name_1", "event_type_1", "event_name_2", "event_type_2",
)
PRICE_COLUMNS = ("store_id", "item_id", "wm_yr_wk", "sell_price")

DEMAND_FEATURES = (
    "origin_sales", "lag_7", "lag_14", "lag_28", "lag_56",
    "mean_7", "mean_28", "mean_56",
    "std_7", "std_28", "std_56",
    "zero_pct_28", "zero_pct_56", "same_weekday_mean_28",
)
CALENDAR_FEATURES = (
    "horizon", "target_weekday", "target_month", "target_snap_CA",
    "target_event_type_1", "target_event_type_2", "target_has_event",
)
PRICE_FEATURES = (
    "last_known_price", "price_change_pct",
    "trailing_4_observed_week_mean_price",
    "price_relative_to_trailing_mean",
    "weeks_since_last_observed_price_change",
    "recorded_price_weeks_to_origin",
    "price_history_missing_indicator",
)
FEATURE_COLUMNS = (
    "id", "origin_day", "target_day",
    *DEMAND_FEATURES, *CALENDAR_FEATURES, *PRICE_FEATURES,
)
EXAMPLE_COLUMNS = (*FEATURE_COLUMNS, "target_sales")

NUMBER_PATTERN = re.compile(
    r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$"
)
DAY_PATTERN = re.compile(r"^d_([1-9][0-9]*)$")

FEATURE_DEFINITIONS = {
    "id": "Previously selected series identifier; selection is unchanged.",
    "origin_day": "Historical day o through which demand is observed.",
    "target_day": "o+h.",
    "origin_sales": "y[o].",
    "lag_7": "y[o-7].",
    "lag_14": "y[o-14].",
    "lag_28": "y[o-28].",
    "lag_56": "y[o-56].",
    "mean_7": "Mean of y[o-6] through y[o], inclusive.",
    "mean_28": "Mean of y[o-27] through y[o], inclusive.",
    "mean_56": "Mean of y[o-55] through y[o], inclusive.",
    "std_7": "Population SD of y[o-6] through y[o], ddof=0.",
    "std_28": "Population SD of y[o-27] through y[o], ddof=0.",
    "std_56": "Population SD of y[o-55] through y[o], ddof=0.",
    "zero_pct_28": "100 * recorded zero days in o-27..o / 28.",
    "zero_pct_56": "100 * recorded zero days in o-55..o / 56.",
    "same_weekday_mean_28": (
        "Mean of four sales in o-27..o whose actual calendar weekday "
        "matches calendar[o+h]."
    ),
    "horizon": "Integer h in 1..28.",
    "target_weekday": "calendar[o+h].date.weekday(); Monday=0.",
    "target_month": "Month derived from calendar[o+h].date.",
    "target_snap_CA": "Validated binary SNAP California flag for o+h.",
    "target_event_type_1": "Calendar event type 1; missing encoded as 'None'.",
    "target_event_type_2": "Calendar event type 2; missing encoded as 'None'.",
    "target_has_event": (
        "1 if either event name or event type is present; otherwise 0."
    ),
    "last_known_price": (
        "Most recent recorded price among weeks eligible at origin; "
        "NaN before any eligible recorded price."
    ),
    "price_change_pct": (
        "100*(latest observed price / preceding observed price - 1); "
        "NaN with fewer than two eligible observed weeks."
    ),
    "trailing_4_observed_week_mean_price": (
        "Mean of the latest up to four eligible observed weekly prices; "
        "NaN when none exist. Missing weeks are not inserted or interpolated."
    ),
    "price_relative_to_trailing_mean": (
        "last_known_price / trailing_4_observed_week_mean_price; a ratio, "
        "not a percentage; NaN when no eligible price exists."
    ),
    "weeks_since_last_observed_price_change": (
        "Calendar-week index of latest eligible week minus index of the "
        "latest observed price transition with unequal consecutive observed "
        "prices. NaN until a price change is observed."
    ),
    "recorded_price_weeks_to_origin": (
        "Count of distinct eligible weeks with a recorded price."
    ),
    "price_history_missing_indicator": (
        "1 if no week is eligible or the latest eligible week lacks a "
        "recorded price; otherwise 0. Earlier gaps are separately audited."
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def integer_value(value: Any, context: str, text: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{context}: boolean values are invalid.")
    if isinstance(value, str):
        require(
            text and NUMBER_PATTERN.fullmatch(value) is not None,
            f"{context}: malformed numeric token.",
        )
    else:
        require(
            isinstance(value, (Real, Decimal)),
            f"{context}: expected a real numeric value.",
        )
    try:
        exact = (
            Decimal(value)
            if isinstance(value, (str, Decimal))
            else Decimal(str(value))
        )
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"{context}: invalid number.") from exc
    require(exact.is_finite(), f"{context}: value must be finite.")
    require(exact >= 0, f"{context}: value must be nonnegative.")
    require(
        exact == exact.to_integral_value(),
        f"{context}: fractional unit values are invalid.",
    )
    result = int(exact)
    try:
        represented = float(result)
    except OverflowError as exc:
        raise ValueError(f"{context}: numeric value is too large.") from exc
    require(
        math.isfinite(represented) and int(represented) == result,
        f"{context}: integer cannot be represented exactly for features.",
    )
    return result


def positive_price(value: Any, context: str, text: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{context}: boolean price is invalid.")
    if isinstance(value, str):
        require(
            text and NUMBER_PATTERN.fullmatch(value) is not None,
            f"{context}: malformed price.",
        )
    else:
        require(
            isinstance(value, (Real, Decimal)),
            f"{context}: expected a numeric price.",
        )
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{context}: invalid price.") from exc
    require(
        math.isfinite(result) and result > 0,
        f"{context}: observed prices must be positive and finite.",
    )
    return result


def csv_header(
    reader: Any, required: tuple[str, ...], source: str
) -> list[str]:
    names = next(reader, [])
    require(bool(names), f"{source}: empty CSV header.")
    require(all(name.strip() for name in names), f"{source}: blank header.")
    require(len(names) == len(set(names)), f"{source}: duplicate headers.")
    missing = [name for name in required if name not in names]
    require(not missing, f"{source}: missing columns {missing}.")
    return names


def reject_json_constant(token: str) -> None:
    raise ValueError(f"Invalid JSON constant: {token}")


def load_selection(root: Path) -> pd.DataFrame:
    path = root / "data/processed/selected_products.csv"
    records = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        names = csv_header(reader, MANIFEST_COLUMNS, path.name)
        positions = {name: names.index(name) for name in MANIFEST_COLUMNS}
        for number, row in enumerate(reader, start=1):
            require(
                len(row) == len(names),
                f"Manifest record {number}: incorrect field count.",
            )
            records.append({
                name: row[index] for name, index in positions.items()
            })
    manifest = pd.DataFrame(records, columns=MANIFEST_COLUMNS)
    require(
        len(manifest) == PRODUCT_COUNT,
        "Approved scope requires exactly 20 selected products.",
    )
    for field in ("id", "item_id", "store_id", "dept_id"):
        require(
            all(isinstance(value, str) and value.strip()
                for value in manifest[field]),
            f"Manifest contains missing {field}.",
        )
    require(not manifest["id"].duplicated().any(), "Duplicate manifest IDs.")
    require(
        not manifest["item_id"].duplicated().any(),
        "Selected product-store item identifiers must be unique.",
    )
    require(
        manifest["store_id"].eq(STORE).all()
        and manifest["dept_id"].eq(DEPARTMENT).all(),
        "Manifest scope must be CA_1 / FOODS_3.",
    )
    manifest["selection_rank"] = [
        integer_value(value, "Selection rank", text=True)
        for value in manifest["selection_rank"]
    ]
    require(
        manifest["selection_rank"].tolist()
        == list(range(1, PRODUCT_COUNT + 1)),
        "Manifest must preserve selection ranks 1..20.",
    )
    metadata_path = root / "data/processed/selection_metadata.json"
    with metadata_path.open("r", encoding="utf-8-sig") as handle:
        metadata = json.load(handle, parse_constant=reject_json_constant)
    require(isinstance(metadata, dict), "Selection metadata must be an object.")
    require(
        type(metadata.get("actual_selected_series_count")) is int
        and metadata["actual_selected_series_count"] == PRODUCT_COUNT,
        "Selection metadata count mismatch.",
    )
    require(
        metadata.get("selected_series_ids_in_rank_order")
        == manifest["id"].tolist(),
        "Selection metadata IDs disagree with manifest rank order.",
    )
    require(
        metadata.get("store_id") == STORE
        and metadata.get("dept_id") == DEPARTMENT,
        "Selection metadata scope mismatch.",
    )
    return manifest


def event_label(raw: str) -> str:
    return MISSING_EVENT if not raw.strip() else raw


def validate_calendar(
    calendar: Mapping[int, Mapping[str, Any]], last_day: int
) -> None:
    require(
        sorted(calendar) == list(range(1, last_day + 1)),
        f"Calendar must cover days 1..{last_day} exactly.",
    )
    dates = []
    previous_week = None
    closed_weeks = set()
    for day in range(1, last_day + 1):
        record = calendar[day]
        require(type(record["date"]) is date, f"d_{day}: invalid date type.")
        require(
            isinstance(record["wm_yr_wk"], Integral)
            and not isinstance(record["wm_yr_wk"], (bool, np.bool_)),
            f"d_{day}: invalid calendar week.",
        )
        require(
            type(record["snap_CA"]) is int and record["snap_CA"] in (0, 1),
            f"d_{day}: invalid SNAP flag.",
        )
        require(
            type(record["has_event"]) is int and record["has_event"] in (0, 1),
            f"d_{day}: invalid event indicator.",
        )
        for field in ("event_type_1", "event_type_2"):
            require(
                isinstance(record[field], str) and record[field].strip(),
                f"d_{day}: missing explicit event category.",
            )
        dates.append(record["date"])
        week = record["wm_yr_wk"]
        if previous_week is not None and week != previous_week:
            closed_weeks.add(previous_week)
            require(week not in closed_weeks, "Calendar week IDs reappear.")
        previous_week = week
    require(len(set(dates)) == len(dates), "Calendar dates are duplicated.")
    require(
        all(b - a == timedelta(days=1) for a, b in zip(dates, dates[1:])),
        "Calendar dates must be strictly increasing and consecutive.",
    )
    runs = []
    for day in range(1, last_day + 1):
        week = calendar[day]["wm_yr_wk"]
        if not runs or runs[-1][0] != week:
            runs.append([week, 1])
        else:
            runs[-1][1] += 1
    require(
        all(1 <= length <= 7 for _, length in runs),
        "Calendar weeks must contain at most seven consecutive days.",
    )
    require(
        all(length == 7 for _, length in runs[1:-1]),
        "Interior calendar weeks must contain exactly seven days.",
    )


def load_calendar(
    path: Path,
    *,
    max_day: int = VALIDATION_END,
) -> dict[int, dict[str, Any]]:
    require(
        isinstance(max_day, Integral)
        and not isinstance(max_day, (bool, np.bool_))
        and 1 <= max_day <= HISTORY_END,
        "Calendar maximum day must be an integer in 1..1913.",
    )
    calendar = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        names = csv_header(reader, CALENDAR_COLUMNS, path.name)
        positions = {
            name: names.index(name)
            for name in CALENDAR_COLUMNS
        }
        for number, row in enumerate(reader, start=1):
            require(
                len(row) == len(names),
                f"Calendar record {number}: incorrect field count.",
            )
            match = DAY_PATTERN.fullmatch(row[positions["d"]])
            require(
                match is not None,
                f"Calendar record {number}: invalid d ID.",
            )
            day = int(match.group(1))
            if day > max_day:
                continue
            require(
                day not in calendar,
                f"Duplicate calendar day d_{day}.",
            )
            raw_date = row[positions["date"]]
            require(
                re.fullmatch(
                    r"[0-9]{4}-[0-9]{2}-[0-9]{2}",
                    raw_date,
                ) is not None,
                f"d_{day}: date must be YYYY-MM-DD.",
            )
            parsed_date = date.fromisoformat(raw_date)
            week = integer_value(
                row[positions["wm_yr_wk"]],
                f"d_{day} week",
                text=True,
            )
            snap = integer_value(
                row[positions["snap_CA"]],
                f"d_{day} SNAP",
                text=True,
            )
            require(
                snap in (0, 1),
                f"d_{day}: SNAP must be 0 or 1.",
            )
            raw_events = [
                row[positions[field]]
                for field in (
                    "event_name_1",
                    "event_type_1",
                    "event_name_2",
                    "event_type_2",
                )
            ]
            calendar[day] = {
                "date": parsed_date,
                "wm_yr_wk": week,
                "snap_CA": snap,
                "event_type_1": event_label(
                    row[positions["event_type_1"]]
                ),
                "event_type_2": event_label(
                    row[positions["event_type_2"]]
                ),
                "has_event": int(any(
                    value.strip() and value != MISSING_EVENT
                    for value in raw_events
                )),
            }
    validate_calendar(calendar, max_day)
    return calendar


def training_weeks(
    calendar: Mapping[int, Mapping[str, Any]],
    *,
    cutoff: int = TRAIN_END,
) -> list[dict[str, Any]]:
    """Order historical weeks by dates, never by numeric week-code magnitude."""
    require(
        isinstance(cutoff, Integral)
        and not isinstance(cutoff, (bool, np.bool_))
        and 1 <= cutoff <= VALIDATION_END,
        "Historical price cutoff must be an integer in 1..1885.",
    )
    require(
        all(day in calendar for day in range(1, cutoff + 1)),
        "Missing historical calendar days for price-week mapping.",
    )
    weeks = []
    for day in range(1, cutoff + 1):
        week = calendar[day]["wm_yr_wk"]
        if not weeks or weeks[-1]["wm_yr_wk"] != week:
            weeks.append({
                "wm_yr_wk": week,
                "index": len(weeks),
                "first_day": day,
                "first_date": calendar[day]["date"],
                "confirmed_start": (
                    day > 1
                    and calendar[day - 1]["wm_yr_wk"] != week
                ),
            })
    require(
        len({week["wm_yr_wk"] for week in weeks}) == len(weeks),
        "Historical calendar week IDs are not contiguous.",
    )
    return weeks


def load_training_sales(
    path: Path,
    manifest: pd.DataFrame,
    calendar: Mapping[int, Mapping[str, Any]],
) -> dict[str, list[int]]:
    selected = manifest.set_index("id").to_dict("index")
    observed = {series_id: {} for series_id in selected}
    last_seen = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        names = csv_header(reader, SALES_COLUMNS, path.name)
        positions = {name: names.index(name) for name in SALES_COLUMNS}
        for number, row in enumerate(reader, start=1):
            require(
                len(row) == len(names),
                f"Sales record {number}: incorrect field count.",
            )
            day = integer_value(
                row[positions["day_number"]], "Sales day_number", text=True
            )
            require(1 <= day <= HISTORY_END, "Sales day_number out of range.")
            expected = (
                "train" if day <= TRAIN_END
                else "validation" if day <= VALIDATION_END else "test"
            )
            require(
                row[positions["split"]] == expected,
                f"Sales record {number}: incorrect split.",
            )
            # Held-out demand fields are never accessed.
            if day > TRAIN_END:
                continue
            series_id = row[positions["id"]]
            require(series_id in selected, f"Unexpected training ID {series_id!r}.")
            require(
                day not in observed[series_id],
                f"Duplicate training product-day ({series_id}, {day}).",
            )
            require(
                day > last_seen.get(series_id, 0),
                f"{series_id}: training days are not ordered.",
            )
            last_seen[series_id] = day
            for field in ("item_id", "store_id", "dept_id"):
                require(
                    row[positions[field]] == selected[series_id][field],
                    f"{series_id}: {field} differs from manifest.",
                )
            require(
                row[positions["d"]] == f"d_{day}",
                f"{series_id}: incorrect day ID.",
            )
            require(
                row[positions["date"]] == calendar[day]["date"].isoformat(),
                f"{series_id}, d_{day}: prepared/calendar date mismatch.",
            )
            observed[series_id][day] = integer_value(
                row[positions["sales"]], f"{series_id}, d_{day}", text=True
            )
    histories = {}
    for series_id in sorted(selected):
        require(
            sorted(observed[series_id]) == list(range(1, TRAIN_END + 1)),
            f"{series_id}: missing training observations.",
        )
        histories[series_id] = [
            observed[series_id][day] for day in range(1, TRAIN_END + 1)
        ]
    return histories


def load_prices(
    path: Path,
    manifest: pd.DataFrame,
    weeks: list[dict[str, Any]],
) -> dict[tuple[str, str, int], float]:
    items = set(manifest["item_id"])
    allowed_weeks = {record["wm_yr_wk"] for record in weeks}
    prices = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        names = csv_header(reader, PRICE_COLUMNS, path.name)
        positions = {name: names.index(name) for name in PRICE_COLUMNS}
        for number, row in enumerate(reader, start=1):
            require(
                len(row) == len(names),
                f"Price record {number}: incorrect field count.",
            )
            if (
                row[positions["store_id"]] != STORE
                or row[positions["item_id"]] not in items
            ):
                continue
            week = integer_value(
                row[positions["wm_yr_wk"]], "Price week", text=True
            )
            if week not in allowed_weeks:
                continue
            key = (STORE, row[positions["item_id"]], week)
            require(key not in prices, f"Duplicate product-week price key {key}.")
            prices[key] = positive_price(
                row[positions["sell_price"]], str(key), text=True
            )
    return prices


def eligible_price_weeks(
    origin: int,
    calendar: Mapping[int, Mapping[str, Any]],
    weeks: list[dict[str, Any]],
    *,
    max_origin: int = TRAIN_END,
) -> list[dict[str, Any]]:
    require(
        isinstance(max_origin, Integral)
        and not isinstance(max_origin, (bool, np.bool_))
        and 1 <= max_origin <= VALIDATION_END,
        "Maximum price-history origin must be an integer in 1..1885.",
    )
    require(
        isinstance(origin, Integral)
        and not isinstance(origin, (bool, np.bool_))
        and 1 <= origin <= max_origin,
        (
            "Price origin must be an integer within training."
            if max_origin == TRAIN_END
            else f"Price origin must be an integer in 1..{max_origin}."
        ),
    )
    require(origin in calendar, "Missing price-origin calendar date.")
    origin_date = calendar[origin]["date"]
    current_week = calendar[origin]["wm_yr_wk"]
    return [
        week for week in weeks
        if week["first_date"] <= origin_date
        and (
            week["wm_yr_wk"] != current_week
            or week["confirmed_start"]
        )
    ]


def historical_price_features(
    item_id: str,
    origin: int,
    calendar: Mapping[int, Mapping[str, Any]],
    prices: Mapping[tuple[str, str, int], Any],
    weeks: list[dict[str, Any]] | None = None,
    *,
    max_origin: int = TRAIN_END,
) -> dict[str, Any]:
    if weeks is None:
        weeks = (
            training_weeks(calendar)
            if max_origin == TRAIN_END
            else training_weeks(calendar, cutoff=max_origin)
        )

    price_boundary = {}
    if max_origin != TRAIN_END:
        price_boundary["max_origin"] = max_origin
    eligible = eligible_price_weeks(
        origin,
        calendar,
        weeks,
        **price_boundary,
    )

    observed = []
    for week in eligible:
        key = (STORE, item_id, week["wm_yr_wk"])
        if key in prices:
            observed.append((
                week["index"],
                positive_price(prices[key], str(key)),
            ))
    latest_missing = (
        not eligible
        or (STORE, item_id, eligible[-1]["wm_yr_wk"]) not in prices
    )
    result = {
        "last_known_price": float("nan"),
        "price_change_pct": float("nan"),
        "trailing_4_observed_week_mean_price": float("nan"),
        "price_relative_to_trailing_mean": float("nan"),
        "weeks_since_last_observed_price_change": float("nan"),
        "recorded_price_weeks_to_origin": len(observed),
        "price_history_missing_indicator": int(latest_missing),
    }
    if not observed:
        return result
    values = [value for _, value in observed]
    latest = values[-1]
    trailing_mean = math.fsum(values[-4:]) / len(values[-4:])
    result["last_known_price"] = latest
    result["trailing_4_observed_week_mean_price"] = trailing_mean
    result["price_relative_to_trailing_mean"] = latest / trailing_mean
    if len(values) >= 2:
        result["price_change_pct"] = (
            100.0 * (latest / values[-2] - 1.0)
        )
    changed_at = [
        current[0]
        for previous, current in zip(observed, observed[1:])
        if current[1] != previous[1]
    ]
    if changed_at:
        result["weeks_since_last_observed_price_change"] = (
            eligible[-1]["index"] - changed_at[-1]
        )
    for name, value in result.items():
        require(
            math.isnan(float(value))
            or math.isfinite(float(value)),
            f"Non-finite derived price feature {name}.",
        )
    return result


def checked_history_prefix(
    history: Any,
    origin: int,
    *,
    max_origin: int = TRAIN_END,
) -> np.ndarray:
    require(
        isinstance(max_origin, Integral)
        and not isinstance(max_origin, (bool, np.bool_))
        and MIN_ORIGIN <= max_origin <= VALIDATION_END,
        "Maximum demand origin must be an integer in 57..1885.",
    )
    require(
        isinstance(origin, Integral)
        and not isinstance(origin, (bool, np.bool_))
        and MIN_ORIGIN <= origin <= max_origin,
        f"Demand feature origin must be an integer in "
        f"{MIN_ORIGIN}..{max_origin}.",
    )
    require(
        isinstance(history, (list, tuple, np.ndarray)),
        "History must be an ordered one-dimensional sequence.",
    )
    if isinstance(history, np.ndarray):
        require(
            history.ndim == 1,
            "History must be one-dimensional.",
        )
    require(
        len(history) >= origin,
        "Missing historical feature window.",
    )
    # Slice first: values after origin are not inspected.
    return np.asarray([
        integer_value(value, f"History day {index + 1}")
        for index, value in enumerate(history[:origin])
    ], dtype=np.float64)


def build_feature_rows(
    series_id: str,
    item_id: str,
    history: Any,
    origin: int,
    calendar: Mapping[int, Mapping[str, Any]],
    prices: Mapping[tuple[str, str, int], Any],
    weeks: list[dict[str, Any]] | None = None,
    *,
    max_origin: int = TRAIN_END,
    max_forecast_day: int = VALIDATION_END,
) -> list[dict[str, Any]]:
    """Construct all 28 horizons from a single frozen historical origin."""
    require(
        isinstance(series_id, str) and series_id.strip()
        and isinstance(item_id, str) and item_id.strip(),
        "Product and item identifiers must be nonempty strings.",
    )
    require(
        isinstance(max_forecast_day, Integral)
        and not isinstance(max_forecast_day, (bool, np.bool_))
        and 1 <= max_forecast_day <= HISTORY_END,
        "Maximum forecast day must be an integer in 1..1913.",
    )

    history_boundary = {}
    if max_origin != TRAIN_END:
        history_boundary["max_origin"] = max_origin
    values = checked_history_prefix(
        history,
        origin,
        **history_boundary,
    )
    require(
        origin + HORIZON <= max_forecast_day,
        "Feature horizon exceeds the explicitly permitted calendar boundary.",
    )

    required_days = range(origin - 55, origin + HORIZON + 1)
    require(
        all(day in calendar for day in required_days),
        "Missing historical or target calendar days.",
    )
    require(
        all(
            type(calendar[day]["date"]) is date
            for day in required_days
        ),
        "Calendar values must contain valid datetime.date objects.",
    )
    require(
        all(
            calendar[day]["date"] - calendar[day - 1]["date"]
            == timedelta(days=1)
            for day in range(origin - 54, origin + HORIZON + 1)
        ),
        "Feature calendar dates must be consecutive.",
    )

    base = {"origin_sales": float(values[origin - 1])}
    for lag in (7, 14, 28, 56):
        base[f"lag_{lag}"] = float(values[origin - lag - 1])
    for window in (7, 28, 56):
        recent = values[origin - window:origin]
        base[f"mean_{window}"] = float(np.mean(recent))
        base[f"std_{window}"] = float(np.std(recent, ddof=0))
    for window in (28, 56):
        recent = values[origin - window:origin]
        base[f"zero_pct_{window}"] = (
            100.0 * int(np.count_nonzero(recent == 0)) / window
        )

    weekday_means = {}
    for weekday in range(7):
        positions = [
            day - 1
            for day in range(origin - 27, origin + 1)
            if calendar[day]["date"].weekday() == weekday
        ]
        require(
            len(positions) == 4,
            "Expected four historical weekday matches.",
        )
        weekday_means[weekday] = float(np.mean(values[positions]))

    price_boundary = {}
    if max_origin != TRAIN_END:
        price_boundary["max_origin"] = max_origin
    price_features = historical_price_features(
        item_id,
        origin,
        calendar,
        prices,
        weeks,
        **price_boundary,
    )

    rows = []
    for horizon in range(1, HORIZON + 1):
        target = origin + horizon
        record = calendar[target]
        weekday = record["date"].weekday()
        require(
            type(record["snap_CA"]) is int
            and record["snap_CA"] in (0, 1),
            f"d_{target}: invalid SNAP flag.",
        )
        require(
            type(record["has_event"]) is int
            and record["has_event"] in (0, 1),
            f"d_{target}: invalid event indicator.",
        )
        for field in ("event_type_1", "event_type_2"):
            require(
                isinstance(record[field], str)
                and record[field].strip(),
                f"d_{target}: invalid explicit event category.",
            )
        row = {
            "id": series_id,
            "origin_day": origin,
            "target_day": target,
            **base,
            "same_weekday_mean_28": weekday_means[weekday],
            "horizon": horizon,
            "target_weekday": weekday,
            "target_month": record["date"].month,
            "target_snap_CA": record["snap_CA"],
            "target_event_type_1": record["event_type_1"],
            "target_event_type_2": record["event_type_2"],
            "target_has_event": record["has_event"],
            **price_features,
        }
        require(
            all(math.isfinite(row[name]) for name in DEMAND_FEATURES),
            "Demand-derived features must be finite.",
        )
        rows.append({
            name: row[name]
            for name in FEATURE_COLUMNS
        })
    return rows


def build_feature_row(
    series_id: str,
    item_id: str,
    history: Any,
    origin: int,
    horizon: int,
    calendar: Mapping[int, Mapping[str, Any]],
    prices: Mapping[tuple[str, str, int], Any],
    weeks: list[dict[str, Any]] | None = None,
    *,
    max_origin: int = TRAIN_END,
    max_forecast_day: int = VALIDATION_END,
) -> dict[str, Any]:
    require(
        isinstance(horizon, Integral)
        and not isinstance(horizon, (bool, np.bool_))
        and 1 <= horizon <= HORIZON,
        "Horizon must be an integer in 1..28.",
    )
    boundaries = {}
    if max_origin != TRAIN_END:
        boundaries["max_origin"] = max_origin
    if max_forecast_day != VALIDATION_END:
        boundaries["max_forecast_day"] = max_forecast_day

    return build_feature_rows(
        series_id,
        item_id,
        history,
        origin,
        calendar,
        prices,
        weeks,
        **boundaries,
    )[horizon - 1]


def build_training_examples(
    histories: Mapping[str, Any],
    item_ids: Mapping[str, str],
    calendar: Mapping[int, Mapping[str, Any]],
    prices: Mapping[tuple[str, str, int], Any],
    origins: Iterable[int],
    *,
    fitting_cutoff: int = TRAIN_END,
    max_forecast_day: int = VALIDATION_END,
) -> pd.DataFrame:
    """Generate supervised rows for caller-supplied complete 28-day origins."""
    require(
        isinstance(fitting_cutoff, Integral)
        and not isinstance(fitting_cutoff, (bool, np.bool_))
        and MIN_ORIGIN + HORIZON
        <= fitting_cutoff <= VALIDATION_END,
        "Fitting cutoff must permit complete labels and cannot exceed 1885.",
    )
    require(
        isinstance(max_forecast_day, Integral)
        and not isinstance(max_forecast_day, (bool, np.bool_))
        and 1 <= max_forecast_day <= HISTORY_END,
        "Maximum forecast day must be an integer in 1..1913.",
    )
    require(bool(histories), "No product histories supplied.")
    require(
        set(histories) == set(item_ids),
        "History/item ID coverage mismatch.",
    )
    require(
        not isinstance(
            origins,
            (str, bytes, Mapping, set, frozenset),
        ),
        "Origins must be an ordered iterable of integers.",
    )
    try:
        chosen = tuple(origins)
    except TypeError as exc:
        raise ValueError("Origins must be iterable.") from exc
    require(bool(chosen), "Origins must not be empty.")
    require(
        all(
            isinstance(origin, Integral)
            and not isinstance(origin, (bool, np.bool_))
            and MIN_ORIGIN <= origin
            and origin + HORIZON <= fitting_cutoff
            for origin in chosen
        ),
        f"Every origin must be >=57 and all 28 targets "
        f"must be <={fitting_cutoff}.",
    )
    require(
        len(chosen) == len(set(chosen)),
        "Duplicate training origins.",
    )
    require(
        chosen == tuple(sorted(chosen)),
        "Origins must be chronological.",
    )

    weeks = (
        training_weeks(calendar)
        if fitting_cutoff == TRAIN_END
        else training_weeks(calendar, cutoff=fitting_cutoff)
    )
    feature_boundaries = {}
    if fitting_cutoff != TRAIN_END:
        feature_boundaries["max_origin"] = fitting_cutoff
    if max_forecast_day != VALIDATION_END:
        feature_boundaries["max_forecast_day"] = max_forecast_day

    records = []
    for series_id in sorted(histories):
        history = histories[series_id]
        require(
            len(history) >= chosen[-1] + HORIZON,
            f"{series_id}: missing historical training labels.",
        )
        for origin in chosen:
            rows = build_feature_rows(
                series_id,
                item_ids[series_id],
                history,
                origin,
                calendar,
                prices,
                weeks,
                **feature_boundaries,
            )
            for row in rows:
                target_day = row["target_day"]
                require(
                    target_day <= fitting_cutoff,
                    "Training label boundary exceeded.",
                )
                row["target_sales"] = integer_value(
                    history[target_day - 1],
                    f"{series_id}, training label d_{target_day}",
                )
                records.append(row)

    frame = pd.DataFrame(records, columns=EXAMPLE_COLUMNS)
    require(
        len(frame) == len(histories) * len(chosen) * HORIZON,
        "Incorrect supervised-example count.",
    )
    require(
        not frame.duplicated(
            ["id", "origin_day", "horizon"]
        ).any(),
        "Duplicate supervised-example keys.",
    )
    return frame


def price_coverage_report(
    manifest: pd.DataFrame,
    weeks: list[dict[str, Any]],
    prices: Mapping[tuple[str, str, int], float],
) -> dict[str, Any]:
    products = []
    total_missing = 0
    before_total = 0
    after_total = 0
    no_history_total = 0
    observed_total = 0
    for product in manifest.itertuples(index=False):
        present = [
            index for index, week in enumerate(weeks)
            if (STORE, product.item_id, week["wm_yr_wk"]) in prices
        ]
        missing = [
            index for index in range(len(weeks)) if index not in set(present)
        ]
        observed_total += len(present)
        total_missing += len(missing)
        if present:
            first = present[0]
            before = [index for index in missing if index < first]
            after = [index for index in missing if index > first]
            before_total += len(before)
            after_total += len(after)
            first_week = weeks[first]["wm_yr_wk"]
            last_week = weeks[present[-1]]["wm_yr_wk"]
            before_count, after_count = len(before), len(after)
            no_history = 0
        else:
            first_week = last_week = None
            before_count = after_count = None
            no_history = len(missing)
            no_history_total += no_history
        products.append({
            "id": product.id,
            "item_id": product.item_id,
            "training_calendar_weeks": len(weeks),
            "recorded_training_price_weeks": len(present),
            "missing_training_price_weeks": len(missing),
            "first_recorded_training_week": first_week,
            "last_recorded_training_week": last_week,
            "missing_weeks_before_first_recorded_price": before_count,
            "missing_weeks_after_first_recorded_price": after_count,
            "missing_weeks_with_no_recorded_history": no_history,
            "missing_week_examples": [
                weeks[index]["wm_yr_wk"] for index in missing[:10]
            ],
        })
    require(
        before_total + after_total + no_history_total == total_missing,
        "Price-gap classifications do not reconcile.",
    )
    require(
        observed_total + total_missing == len(manifest) * len(weeks),
        "Price coverage does not reconcile.",
    )
    return {
        "scope": "Selected store/items and weeks intersecting d_1..d_1857",
        "training_calendar_weeks": len(weeks),
        "expected_product_week_combinations": len(manifest) * len(weeks),
        "recorded_product_week_combinations": observed_total,
        "missing_product_week_combinations": total_missing,
        "missing_before_first_recorded_price": before_total,
        "missing_after_first_recorded_price": after_total,
        "missing_with_no_recorded_training_history": no_history_total,
        "products": products,
    }


def create_audit_report(
    manifest: pd.DataFrame,
    histories: Mapping[str, Any],
    calendar: Mapping[int, Mapping[str, Any]],
    prices: Mapping[tuple[str, str, int], float],
    source_paths: Mapping[str, str],
) -> dict[str, Any]:
    validate_calendar(calendar, VALIDATION_END)
    require(
        set(histories) == set(manifest["id"]),
        "Audit history/manifest IDs disagree.",
    )
    weeks = training_weeks(calendar)
    coverage = price_coverage_report(manifest, weeks, prices)
    item_ids = manifest.set_index("id")["item_id"].to_dict()
    probe_count = 0
    missing_price_feature_cells = 0
    for series_id in sorted(histories):
        require(
            len(histories[series_id]) == TRAIN_END,
            f"{series_id}: audit requires exactly 1857 training observations.",
        )
        checked_history_prefix(histories[series_id], TRAIN_END)
        # Small fixed integrity probes, not a model-training origin schedule.
        for origin in (MIN_ORIGIN, TRAIN_END):
            rows = build_feature_rows(
                series_id, item_ids[series_id], histories[series_id],
                origin, calendar, prices, weeks,
            )
            probe_count += len(rows)
            missing_price_feature_cells += sum(
                math.isnan(float(row[name]))
                for row in rows for name in PRICE_FEATURES
            )
    return {
        "status": "PASS",
        "dataset": "Walmart M5 Forecasting Accuracy",
        "scope": {"store_id": STORE, "dept_id": DEPARTMENT},
        "source_paths": dict(source_paths),
        "selected_product_count": len(manifest),
        "selected_ids_in_rank_order": manifest["id"].tolist(),
        "training_observation_count": sum(len(values) for values in histories.values()),
        "calendar_integrity": {
            "status": "PASS",
            "relevant_day_count": len(calendar),
            "first_day": "d_1",
            "last_day": "d_1885",
            "first_date": calendar[1]["date"].isoformat(),
            "last_date": calendar[VALIDATION_END]["date"].isoformat(),
            "unique_days_and_dates": True,
            "consecutive_dates": True,
            "prepared_training_calendar_matches_raw_calendar": True,
            "target_weekdays_derived_from_dates": True,
        },
        "price_coverage": coverage,
        "feature_names": list(FEATURE_COLUMNS),
        "feature_definitions": FEATURE_DEFINITIONS,
        "supervised_label": {
            "name": "target_sales",
            "definition": "Recorded sales at origin_day + horizon.",
            "maximum_day": TRAIN_END,
            "origin_schedule": "Supplied by caller; no model schedule selected here.",
        },
        "historical_availability_assumptions": [
            "Demand is observed only through the forecast origin.",
            "Scheduled calendar events and SNAP flags are known at the origin.",
            "Weekly prices are assumed known once their associated week begins.",
            "The current week's price is excluded unless its start is confirmed "
            "by a preceding calendar day with a different week ID.",
            "The first calendar week has an unconfirmed start; it is excluded "
            "while current, but can be used once it is a past week.",
            "Only selected-product price records in training calendar weeks are loaded.",
            "Missing weeks are not assigned zero or interpolated.",
            "A last-known price uses only an earlier eligible observation and "
            "is never propagated backward before the first eligible observation.",
        ],
        "leakage_safeguard_checks": {
            "status": "PASS",
            "training_sales_only": True,
            "validation_sales_accessed": False,
            "test_sales_accessed": False,
            "future_prices_excluded_per_origin": True,
            "sales_imputation_performed": False,
            "product_selection_changed": False,
            "model_trained": False,
            "validation_or_test_performance_evaluated": False,
            "integrity_probe_feature_rows": probe_count,
            "integrity_probe_missing_price_feature_cells": missing_price_feature_cells,
        },
        "warnings": (
            ["Historical price gaps exist; unavailable numeric price features remain NaN."]
            if coverage["missing_product_week_combinations"] else []
        ),
        "limitations": [
            "The high-volume selected subset is not representative of all products.",
            "Recorded sales do not necessarily equal unconstrained customer demand.",
            "Missing price records do not establish product-launch dates.",
            "Weekly files do not provide independently verified publication timestamps; "
            "week-start availability is an explicit modeling assumption.",
            "A change between nonadjacent observed price weeks does not reveal "
            "the exact unobserved change date.",
            "Price quality checks cover selected products and historical training weeks.",
            "Calendar fields through validation are checked as known covariates; "
            "validation demand is not inspected.",
            "No final-test calendar or demand is retained.",
            "No supervised training matrix is exported.",
            "This audit verifies feature inputs and construction, not model accuracy.",
        ],
    }


def write_audit(report: dict[str, Any], destination: Path) -> None:
    payload = json.dumps(
        report, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="",
            prefix=".xgb-feature-audit-", suffix=".tmp",
            dir=destination.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        with temporary.open("r", encoding="utf-8") as handle:
            require(json.load(handle) == report, "Staged audit verification failed.")
        temporary.replace(destination)
        temporary = None
        with destination.open("r", encoding="utf-8") as handle:
            require(json.load(handle) == report, "Published audit verification failed.")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_audit(root: Path = ROOT) -> dict[str, Any]:
    root = Path(root)
    relative_paths = {
        "selected_sales": "data/processed/selected_sales.csv",
        "selected_products": "data/processed/selected_products.csv",
        "selection_metadata": "data/processed/selection_metadata.json",
        "calendar": "data/raw/calendar.csv",
        "sell_prices": "data/raw/sell_prices.csv",
    }
    paths = {name: root / relative for name, relative in relative_paths.items()}
    for path in paths.values():
        require(path.is_file(), f"Required input missing: {path}")
    manifest = load_selection(root)
    calendar = load_calendar(paths["calendar"])
    histories = load_training_sales(paths["selected_sales"], manifest, calendar)
    weeks = training_weeks(calendar)
    prices = load_prices(paths["sell_prices"], manifest, weeks)
    report = create_audit_report(
        manifest, histories, calendar, prices, relative_paths
    )
    destination = root / "outputs/evaluation/xgb_feature_audit.json"
    write_audit(report, destination)
    return {"report": report, "output_path": destination}


def main() -> int:
    try:
        result = run_audit()
    except Exception as exc:
        print(
            f"XGBoost feature audit FAILED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print(
            "An existing report must not be treated as a successful current run.",
            file=sys.stderr,
        )
        return 1
    report = result["report"]
    coverage = report["price_coverage"]
    print("XGBoost feature audit: PASS")
    print(f"Selected products: {report['selected_product_count']}")
    print(f"Training observations: {report['training_observation_count']}")
    print(f"Training calendar weeks: {coverage['training_calendar_weeks']}")
    print(f"Recorded product-weeks: {coverage['recorded_product_week_combinations']}")
    print(f"Missing product-weeks: {coverage['missing_product_week_combinations']}")
    print(f"Warnings: {len(report['warnings'])}")
    print("No model trained. Validation/test sales were not accessed.")
    print(f"Report: {result['output_path']}")
    print("Exit code: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())