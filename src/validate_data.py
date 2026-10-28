"""Read-only validation of M5 identifiers, calendar, and CA_1 / FOODS_3 sales.

Run from the project root:
    python -m src.validate_data

Only the JSON report is written. Raw CSV files are never modified.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import warnings
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SALES_PATH = PROJECT_ROOT / "data" / "raw" / "sales_train_validation.csv"
CALENDAR_PATH = PROJECT_ROOT / "data" / "raw" / "calendar.csv"
REPORT_PATH = (
    PROJECT_ROOT / "outputs" / "evaluation" / "data_validation_report.json"
)

CHUNK_SIZE = 500
EXAMPLE_LIMIT = 5
STORE = "CA_1"
DEPARTMENT = "FOODS_3"
HISTORY_DAYS = 1913
TRAINING_DAYS = 1857
MIN_ACTIVITY_DAYS = 28
DESIRED_SERIES = 20
FORECAST_HORIZON = 28

IDENTIFIERS = ("id", "item_id", "dept_id", "cat_id", "store_id", "state_id")
DAY_PATTERN = re.compile(r"^d_[1-9][0-9]*$")
EXPECTED_DAYS = tuple(f"d_{number}" for number in range(1, HISTORY_DAYS + 1))
EXPECTED_DAY_SET = set(EXPECTED_DAYS)

# Decimal avoids rounding fractional values into integers and avoids float
# overflow turning a large, finite numeric token into infinity.
NUMBER_PATTERN = re.compile(
    r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$"
)
NONFINITE_PATTERN = re.compile(r"^[+-]?(?:inf(?:inity)?|s?nan)$", re.IGNORECASE)

CELL_CATEGORIES = (
    "missing",
    "non_numeric",
    "negative",
    "non_finite",
    "fractional",
    "valid_zero",
    "valid_positive",
)
INVALID_CATEGORIES = CELL_CATEGORIES[:5]


def bounded_text(value: Any, limit: int = 120) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "..."


def add_example(examples: list[Any], example: Any) -> None:
    if len(examples) < EXAMPLE_LIMIT:
        examples.append(example)


def add_check(
    report: dict[str, Any],
    check_id: str,
    description: str,
    affected_count: int | None,
    examples: list[Any] | None = None,
    severity: str = "ERROR",
    skipped_reason: str | None = None,
) -> None:
    if affected_count is None:
        status = "SKIP"
    elif affected_count:
        status = "WARN" if severity == "WARNING" else "FAIL"
    else:
        status = "PASS"

    check: dict[str, Any] = {
        "id": check_id,
        "description": description,
        "status": status,
        "severity": severity,
        "affected_count": affected_count,
        "examples": (examples or [])[:EXAMPLE_LIMIT],
    }
    if skipped_reason is not None:
        check["reason"] = skipped_reason

    report["checks"].append(check)
    if status in {"FAIL", "WARN"}:
        issue = {
            "check_id": check_id,
            "kind": "data_validation",
            "description": description,
            "affected_count": affected_count,
            "examples": check["examples"],
        }
        destination = "warnings" if status == "WARN" else "errors"
        report[destination].append(issue)


def execution_error(
    report: dict[str, Any], operation: str, exc: Exception
) -> None:
    description = (
        f"{operation}: {type(exc).__name__}: {bounded_text(exc, 1000)}"
    )
    report["errors"].append(
        {
            "check_id": "execution.failure",
            "kind": "execution",
            "description": description,
            "affected_count": None,
            "examples": [],
        }
    )
    print(f"Execution failure: {description}", file=sys.stderr)


def new_report() -> dict[str, Any]:
    metric_names = (
        "total_sales_rows_processed",
        "number_of_daily_columns",
        "target_slice_series",
        "target_series_with_valid_historical_sales",
        "target_series_with_any_positive_sales_day",
        "target_series_meeting_training_activity_threshold",
        "preliminarily_eligible_series",
        "at_least_20_preliminarily_eligible_series",
        "target_historical_sales_cells",
        "missing_sales_cells",
        "non_numeric_sales_cells",
        "negative_sales_cells",
        "non_finite_sales_cells",
        "fractional_sales_cells",
        "total_invalid_sales_cells",
        "valid_zero_sales_cells",
        "valid_positive_sales_cells",
        "zero_sales_percentage",
    )
    return {
        "status": "FAIL",
        "dataset": "M5 sales_train_validation",
        "scope": {"store_id": STORE, "dept_id": DEPARTMENT},
        "summary": {name: None for name in metric_names},
        "checks": [],
        "errors": [],
        "warnings": [],
        "coverage": {
            "sales_stream_completed": False,
            "calendar_read_completed": False,
            "identifier_integrity": (
                "Checked across the entire sales file, including across chunks, "
                "for every required identifier column that is present."
            ),
            "detailed_numeric_validation": (
                "Only exact CA_1 / FOODS_3 matches are numerically validated; "
                "every matching row and all 1913 historical cells are checked "
                "when the required historical columns are available."
            ),
            "raw_data_modified": False,
            "imputation_or_cleaning_performed": False,
            "preliminary_eligibility_is_final_product_selection": False,
            "preliminary_eligibility_definition": (
                "All historical sales values valid, all required identifiers "
                "present, no globally duplicated series ID, and at least "
                "28 valid positive-sales days in d_1 through d_1857. "
                "Calendar and structural errors still fail the overall run."
            ),
            "positive_activity_counts": (
                "Any-positive and training-activity counts use valid positive "
                "cells even if another cell in that row is invalid. "
                "Preliminary eligibility additionally requires row integrity."
            ),
            "zero_sales_percentage_denominator": (
                "All target-slice historical sales cells: target row count "
                "multiplied by 1913, including invalid and missing cells. "
                "Only valid numeric zeros enter the numerator."
            ),
            "historical_days": HISTORY_DAYS,
            "training_window": ["d_1", f"d_{TRAINING_DAYS}"],
            "minimum_positive_training_days": MIN_ACTIVITY_DAYS,
            "desired_future_series_count": DESIRED_SERIES,
            "future_forecast_horizon_days": FORECAST_HORIZON,
            "chunk_size_rows": CHUNK_SIZE,
            "examples_per_check_limit": EXAMPLE_LIMIT,
            "calendar_date_consecutiveness_checked": False,
            "limitations": [
                "Numeric sales outside the target slice are not validated.",
                "Calendar dates are parsed as ISO YYYY-MM-DD dates.",
                "Calendar date consecutiveness is not assumed or checked.",
                "Duplicate calendar day IDs are checked across all calendar rows.",
                "No final product selection, forecasting, or inventory logic.",
                "Metrics can represent partial progress if execution fails; "
                "consult completion flags and execution errors.",
            ],
        },
    }


def validate_header(
    report: dict[str, Any],
    header: list[str],
    prefix: str,
    required: tuple[str, ...],
) -> bool:
    counts = Counter(header)
    blank_positions = [
        number for number, name in enumerate(header, start=1) if not name.strip()
    ]
    duplicate_names = [name for name, count in counts.items() if count > 1]
    duplicate_excess = sum(counts[name] - 1 for name in duplicate_names)
    missing = [name for name in required if name not in counts]

    add_check(
        report,
        f"{prefix}.header.blank",
        "Column headers must not be empty or whitespace-only.",
        len(blank_positions),
        blank_positions[:EXAMPLE_LIMIT],
    )
    add_check(
        report,
        f"{prefix}.header.duplicate",
        "Column headers must be unique; count is excess occurrences.",
        duplicate_excess,
        [
            {"column": bounded_text(name), "occurrences": counts[name]}
            for name in duplicate_names[:EXAMPLE_LIMIT]
        ],
    )
    add_check(
        report,
        f"{prefix}.header.required",
        "Required columns must exist.",
        len(missing),
        missing,
    )
    return not blank_positions and not duplicate_names


def read_sales_header(report: dict[str, Any]) -> tuple[list[str], bool]:
    with SALES_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle, strict=True), [])

    safe_header = validate_header(report, header, "sales", IDENTIFIERS)
    daily = sorted(
        (name for name in header if DAY_PATTERN.fullmatch(name)),
        key=lambda name: int(name[2:]),
    )
    report["summary"]["number_of_daily_columns"] = len(daily)
    daily_counts = Counter(daily)
    missing = [day for day in EXPECTED_DAYS if day not in daily_counts]
    unexpected = sorted(
        set(daily) - EXPECTED_DAY_SET, key=lambda name: int(name[2:])
    )
    duplicates = [
        day for day, count in daily_counts.items() if count > 1
    ]
    affected = (
        len(missing)
        + len(unexpected)
        + sum(daily_counts[day] - 1 for day in duplicates)
    )
    examples: list[Any] = []
    for kind, names in (
        ("missing", missing),
        ("unexpected", unexpected),
        ("duplicate", duplicates),
    ):
        for name in names:
            add_example(examples, {"kind": kind, "column": name})

    add_check(
        report,
        "sales.header.historical_days",
        "Daily columns must be exactly d_1 through d_1913.",
        affected,
        examples,
    )
    history_available = (
        safe_header
        and not missing
        and not unexpected
        and not duplicates
    )
    return header, history_available


def parse_sales_token(raw: Any) -> str:
    if raw is None or pd.isna(raw):
        return "missing"
    token = str(raw).strip()
    if not token:
        return "missing"
    if NONFINITE_PATTERN.fullmatch(token):
        return "non_finite"
    if not NUMBER_PATTERN.fullmatch(token):
        return "non_numeric"

    try:
        value = Decimal(token)
    except InvalidOperation:
        return "non_numeric"

    if not value.is_finite():
        return "non_finite"
    # Categories are exclusive: negative fractional values count as negative.
    if value < 0:
        return "negative"
    if value != value.to_integral_value():
        return "fractional"
    return "valid_zero" if value == 0 else "valid_positive"


def validate_sales(report: dict[str, Any]) -> None:
    header, history_available = read_sales_header(report)
    summary = report["summary"]
    safe_header = bool(header) and all(name.strip() for name in header)
    safe_header = safe_header and len(header) == len(set(header))

    if not safe_header:
        for check_id, description in (
            ("sales.identifiers.missing", "Required identifiers must be populated."),
            ("sales.identifiers.duplicate_id", "Series IDs must be globally unique."),
            ("sales.target.available", "The target slice must contain rows."),
            ("sales.values.complete", "Target historical sales must be valid."),
        ):
            add_check(
                report, check_id, description, None,
                skipped_reason="Sales header is empty, blank, or duplicated.",
            )
        return

    present_ids = [name for name in IDENTIFIERS if name in header]
    can_identify_target = {"store_id", "dept_id"}.issubset(header)
    can_validate_values = can_identify_target and history_available

    # Keep all tokens as strings and disable NA inference. Thus "", "NA",
    # malformed strings, and "0" remain distinguishable. Only identifiers and
    # historical columns are materialized; only target rows are parsed numerically.
    columns = set(present_ids)
    if can_validate_values:
        columns.update(EXPECTED_DAYS)
    if not columns:
        # Preserve row counting even if no usable validation columns exist.
        columns.add(header[0])
    usecols = [name for name in header if name in columns]

    summary["total_sales_rows_processed"] = 0
    if can_identify_target:
        summary["target_slice_series"] = 0
    if can_validate_values:
        for name in (
            "target_series_with_valid_historical_sales",
            "target_series_with_any_positive_sales_day",
            "target_series_meeting_training_activity_threshold",
            "preliminarily_eligible_series",
            "target_historical_sales_cells",
            "missing_sales_cells",
            "non_numeric_sales_cells",
            "negative_sales_cells",
            "non_finite_sales_cells",
            "fractional_sales_cells",
            "total_invalid_sales_cells",
            "valid_zero_sales_cells",
            "valid_positive_sales_cells",
        ):
            summary[name] = 0

    id_counts: Counter[str] = Counter()
    missing_id_cells = 0
    missing_id_rows = 0
    missing_examples: list[Any] = []
    duplicate_examples: list[Any] = []
    duplicate_excess = 0
    category_counts = Counter({category: 0 for category in CELL_CATEGORIES})
    category_examples: dict[str, list[Any]] = {
        category: [] for category in INVALID_CATEGORIES
    }
    # Retain only small candidate records, never historical arrays, so a later
    # duplicate ID outside the target slice can disqualify an earlier candidate.
    candidate_ids: list[str] = []

    with warnings.catch_warnings():
        # A parser warning can indicate discarded fields; treat it as an
        # execution failure rather than accepting potentially truncated data.
        warnings.simplefilter("error", pd.errors.ParserWarning)
        with pd.read_csv(
            SALES_PATH,
            encoding="utf-8-sig",
            dtype=str,
            na_filter=False,
            keep_default_na=False,
            chunksize=CHUNK_SIZE,
            usecols=usecols,
            index_col=False,
            skip_blank_lines=False,
            on_bad_lines="error",
        ) as chunks:
            for chunk in chunks:
                positions = {
                    name: index for index, name in enumerate(chunk.columns)
                }
                for row in chunk.itertuples(index=False, name=None):
                    summary["total_sales_rows_processed"] += 1
                    record_number = summary["total_sales_rows_processed"]
                    identifiers: dict[str, str] = {}
                    row_missing_ids: list[str] = []
                    for name in present_ids:
                        raw = row[positions[name]]
                        text = "" if raw is None or pd.isna(raw) else str(raw)
                        identifiers[name] = text
                        if not text.strip():
                            row_missing_ids.append(name)
                            missing_id_cells += 1

                    if row_missing_ids:
                        missing_id_rows += 1
                        add_example(
                            missing_examples,
                            {
                                "data_record": record_number,
                                "columns": row_missing_ids,
                            },
                        )

                    series_id = identifiers.get("id", "")
                    if series_id.strip():
                        id_counts[series_id] += 1
                        if id_counts[series_id] > 1:
                            duplicate_excess += 1
                            add_example(
                                duplicate_examples,
                                {
                                    "data_record": record_number,
                                    "id": bounded_text(series_id),
                                },
                            )

                    is_target = (
                        can_identify_target
                        and identifiers["store_id"] == STORE
                        and identifiers["dept_id"] == DEPARTMENT
                    )
                    if not is_target:
                        continue

                    summary["target_slice_series"] += 1
                    if not can_validate_values:
                        continue

                    invalid_cells = 0
                    positive_days = 0
                    training_positive_days = 0
                    for day_number, day in enumerate(EXPECTED_DAYS, start=1):
                        raw = row[positions[day]]
                        category = parse_sales_token(raw)
                        category_counts[category] += 1
                        if category in INVALID_CATEGORIES:
                            invalid_cells += 1
                            add_example(
                                category_examples[category],
                                {
                                    "data_record": record_number,
                                    "id": bounded_text(series_id),
                                    "day": day,
                                    "raw_token": bounded_text(raw),
                                },
                            )
                        elif category == "valid_positive":
                            positive_days += 1
                            if day_number <= TRAINING_DAYS:
                                training_positive_days += 1

                    if not invalid_cells:
                        summary["target_series_with_valid_historical_sales"] += 1
                    if positive_days:
                        summary["target_series_with_any_positive_sales_day"] += 1
                    if training_positive_days >= MIN_ACTIVITY_DAYS:
                        summary[
                            "target_series_meeting_training_activity_threshold"
                        ] += 1
                        if (
                            not invalid_cells
                            and not row_missing_ids
                            and len(present_ids) == len(IDENTIFIERS)
                        ):
                            candidate_ids.append(series_id)

                # Publish completed-chunk metrics for useful partial reporting
                # if a subsequent CSV read fails.
                if can_validate_values:
                    update_cell_metrics(summary, category_counts)
                    summary["preliminarily_eligible_series"] = sum(
                        id_counts[series_id] == 1 for series_id in candidate_ids
                    )

    report["coverage"]["sales_stream_completed"] = True
    report["coverage"]["missing_identifier_rows"] = missing_id_rows
    report["coverage"]["identifier_columns_checked"] = present_ids

    add_check(
        report,
        "sales.identifiers.missing",
        "Present required identifier columns must have no blank or null cells.",
        missing_id_cells,
        missing_examples,
    )
    add_check(
        report,
        "sales.identifiers.duplicate_id",
        "Nonblank series IDs must be unique across the entire sales file; "
        "count is excess occurrences.",
        duplicate_excess if "id" in present_ids else None,
        duplicate_examples,
        skipped_reason=None if "id" in present_ids else "The id column is absent.",
    )
    target_count = summary["target_slice_series"]
    add_check(
        report,
        "sales.target.available",
        "The CA_1 / FOODS_3 target slice must contain at least one row.",
        None if target_count is None else int(target_count == 0),
        skipped_reason=(
            None if can_identify_target
            else "store_id or dept_id is absent."
        ),
    )

    if not can_validate_values:
        add_check(
            report,
            "sales.values.complete",
            "All target historical sales cells must be valid nonnegative integers.",
            None,
            skipped_reason="Target identifiers or exact historical columns unavailable.",
        )
        return

    update_cell_metrics(summary, category_counts)
    summary["preliminarily_eligible_series"] = sum(
        id_counts[series_id] == 1 for series_id in candidate_ids
    )
    summary["at_least_20_preliminarily_eligible_series"] = (
        summary["preliminarily_eligible_series"] >= DESIRED_SERIES
    )

    descriptions = {
        "missing": "Target sales cells must not be missing or blank.",
        "non_numeric": "Target sales cells must contain numeric tokens.",
        "negative": "Target sales cells must not be negative.",
        "non_finite": "Target sales cells must not contain infinity or NaN.",
        "fractional": "Target sales cells must not contain fractional values.",
    }
    for category in INVALID_CATEGORIES:
        add_check(
            report,
            f"sales.values.{category}",
            descriptions[category],
            category_counts[category],
            category_examples[category],
        )

    eligible = summary["preliminarily_eligible_series"]
    add_check(
        report,
        "sales.activity.minimum_pool",
        "At least 20 series should meet preliminary integrity and activity criteria; "
        "affected count is the shortfall.",
        max(0, DESIRED_SERIES - eligible),
        [{"eligible_series": eligible, "desired_series": DESIRED_SERIES}],
        severity="WARNING",
    )
    zero_percentage = summary["zero_sales_percentage"]
    add_check(
        report,
        "sales.sparsity.zero_prevalence",
        "Zero-sales prevalence should not exceed 80% of all target historical cells; "
        "affected count is valid zero cells when the threshold is exceeded.",
        (
            None if zero_percentage is None
            else summary["valid_zero_sales_cells"] if zero_percentage > 80 else 0
        ),
        (
            [] if zero_percentage is None
            else [{"zero_sales_percentage": zero_percentage}]
        ),
        severity="WARNING",
        skipped_reason=(
            "No target historical cells." if zero_percentage is None else None
        ),
    )


def update_cell_metrics(
    summary: dict[str, Any], counts: Counter[str]
) -> None:
    for category in CELL_CATEGORIES:
        summary[f"{category}_sales_cells"] = counts[category]
    summary["total_invalid_sales_cells"] = sum(
        counts[category] for category in INVALID_CATEGORIES
    )
    denominator = sum(counts.values())
    summary["target_historical_sales_cells"] = denominator
    summary["zero_sales_percentage"] = (
        100.0 * counts["valid_zero"] / denominator if denominator else None
    )


def validate_calendar(report: dict[str, Any]) -> None:
    day_counts: Counter[str] = Counter()
    mappings: dict[str, list[str]] = {}
    duplicate_count = 0
    duplicate_examples: list[Any] = []
    blank_day_count = 0
    blank_day_examples: list[Any] = []

    # One open and one sequential read of the calendar, including its header.
    # csv.reader preserves raw date/day tokens and exposes ragged records.
    with CALENDAR_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        header = next(reader, [])
        safe_header = validate_header(
            report, header, "calendar", ("d", "date")
        )
        if not safe_header or not {"d", "date"}.issubset(header):
            add_check(
                report,
                "calendar.historical.mapping",
                "Every historical day must map to exactly one parseable date.",
                None,
                skipped_reason="Calendar header is invalid or required columns absent.",
            )
            return

        day_index = header.index("d")
        date_index = header.index("date")
        for record_number, row in enumerate(reader, start=1):
            if len(row) != len(header):
                raise ValueError(
                    f"Malformed calendar CSV at data record {record_number}: "
                    f"expected {len(header)} fields, received {len(row)}."
                )
            day = row[day_index]
            date_token = row[date_index]
            if not day.strip():
                blank_day_count += 1
                add_example(
                    blank_day_examples, {"data_record": record_number}
                )
                continue
            day_counts[day] += 1
            if day_counts[day] > 1:
                duplicate_count += 1
                add_example(
                    duplicate_examples,
                    {"data_record": record_number, "day": bounded_text(day)},
                )
            if day in EXPECTED_DAY_SET:
                mappings.setdefault(day, []).append(date_token)

    report["coverage"]["calendar_read_completed"] = True
    add_check(
        report,
        "calendar.day.blank",
        "Calendar day IDs must not be blank.",
        blank_day_count,
        blank_day_examples,
    )
    add_check(
        report,
        "calendar.day.duplicate",
        "Calendar day IDs must be unique; count is excess occurrences.",
        duplicate_count,
        duplicate_examples,
    )

    missing_days = [day for day in EXPECTED_DAYS if day not in mappings]
    ambiguous_days = [
        day for day in EXPECTED_DAYS if len(mappings.get(day, [])) > 1
    ]
    missing_dates = 0
    invalid_dates = 0
    missing_date_examples: list[Any] = []
    invalid_date_examples: list[Any] = []

    from datetime import date

    for day in EXPECTED_DAYS:
        for token in mappings.get(day, []):
            if not token.strip():
                missing_dates += 1
                add_example(missing_date_examples, {"day": day})
                continue
            stripped = token.strip()
            try:
                if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", stripped):
                    raise ValueError("Expected YYYY-MM-DD.")
                date.fromisoformat(stripped)
            except ValueError:
                invalid_dates += 1
                add_example(
                    invalid_date_examples,
                    {"day": day, "raw_date": bounded_text(token)},
                )

    add_check(
        report,
        "calendar.historical.missing",
        "Every day from d_1 through d_1913 must exist in the calendar.",
        len(missing_days),
        missing_days[:EXAMPLE_LIMIT],
    )
    add_check(
        report,
        "calendar.historical.ambiguous",
        "Each historical day must have exactly one calendar row.",
        len(ambiguous_days),
        ambiguous_days[:EXAMPLE_LIMIT],
    )
    add_check(
        report,
        "calendar.historical.date_missing",
        "Historical calendar dates must be populated.",
        missing_dates,
        missing_date_examples,
    )
    add_check(
        report,
        "calendar.historical.date_invalid",
        "Historical calendar dates must be valid ISO YYYY-MM-DD dates.",
        invalid_dates,
        invalid_date_examples,
    )


def write_report(report: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Serialize first, then replace atomically so a failed write does not leave
    # a partially written report at the final destination.
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    temporary_path = REPORT_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(payload + "\n", encoding="utf-8")
    temporary_path.replace(REPORT_PATH)


def print_summary(
    report: dict[str, Any], exit_code: int, report_written: bool
) -> None:
    summary = report["summary"]

    def display(name: str) -> str:
        value = summary[name]
        return "unavailable" if value is None else str(value)

    zero_percentage = summary["zero_sales_percentage"]
    zero_text = (
        "unavailable" if zero_percentage is None else f"{zero_percentage:.2f}%"
    )
    print()
    print(f"M5 data validation: {report['status']}")
    print(f"Scope: {STORE} / {DEPARTMENT}")
    print(f"Sales rows processed: {display('total_sales_rows_processed')}")
    print(f"Daily columns: {display('number_of_daily_columns')}")
    print(f"Target series: {display('target_slice_series')}")
    print(
        "Valid historical-sales series: "
        f"{display('target_series_with_valid_historical_sales')}"
    )
    print(
        "Series with any positive day: "
        f"{display('target_series_with_any_positive_sales_day')}"
    )
    print(
        "Series with >=28 positive training days: "
        f"{display('target_series_meeting_training_activity_threshold')}"
    )
    print(f"Preliminarily eligible series: {display('preliminarily_eligible_series')}")
    print(
        "At least 20 preliminarily eligible: "
        f"{display('at_least_20_preliminarily_eligible_series')}"
    )
    print(f"Invalid sales cells: {display('total_invalid_sales_cells')}")
    print(f"Valid zero-sales cells: {display('valid_zero_sales_cells')}")
    print(f"Valid positive-sales cells: {display('valid_positive_sales_cells')}")
    print(f"Zero-sales prevalence: {zero_text}")
    print(f"Errors: {len(report['errors'])}; warnings: {len(report['warnings'])}")
    print(f"Report: {REPORT_PATH}")
    print(f"Report written this run: {'yes' if report_written else 'no'}")
    print(f"Exit code: {exit_code}")


def main() -> int:
    report = new_report()
    execution_failed = False

    # Keep independent file validation running after a failure in the other file.
    for operation, validator in (
        ("Sales validation", validate_sales),
        ("Calendar validation", validate_calendar),
    ):
        try:
            validator(report)
        except Exception as exc:
            execution_failed = True
            execution_error(report, operation, exc)

    exit_code = 2 if execution_failed else 1 if report["errors"] else 0
    report["status"] = "PASS" if exit_code == 0 else "FAIL"
    report["coverage"]["exit_code"] = exit_code

    report_written = False
    try:
        write_report(report)
        report_written = True
    except Exception as exc:
        execution_error(report, "JSON report writing", exc)
        exit_code = 2
        report["status"] = "FAIL"
        report["coverage"]["exit_code"] = exit_code
        # The filesystem may be unavailable. Preserve the current report on
        # stderr as a fallback; an older on-disk report must not imply success.
        print("Current report fallback:", file=sys.stderr)
        print(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
            file=sys.stderr,
        )

    print_summary(report, exit_code, report_written)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())