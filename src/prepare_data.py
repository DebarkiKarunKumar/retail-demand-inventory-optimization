"""Prepare the deterministic M5 CA_1 / FOODS_3 high-volume subset.

Run:
    python -m src.prepare_data

Raw inputs and the Stage 2A report are read-only. Historical completeness is
a quality gate; activity, volume, and ranking use training-period sales only.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import tempfile
import warnings
from collections import Counter
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SALES_PATH = ROOT / "data/raw/sales_train_validation.csv"
CALENDAR_PATH = ROOT / "data/raw/calendar.csv"
VALIDATION_PATH = ROOT / "outputs/evaluation/data_validation_report.json"
OUTPUT_DIR = ROOT / "data/processed"

MANIFEST_PATH = OUTPUT_DIR / "selected_products.csv"
PREPARED_PATH = OUTPUT_DIR / "selected_sales.csv"
METADATA_PATH = OUTPUT_DIR / "selection_metadata.json"

STORE = "CA_1"
DEPARTMENT = "FOODS_3"
CHUNK_SIZE = 500
HISTORY_DAYS = 1913
TRAIN_END = 1857
VALIDATION_END = 1885
MIN_POSITIVE_DAYS = 28
MAX_SERIES = 20
FORECAST_HORIZON = 28

IDENTIFIERS = ("id", "item_id", "dept_id", "cat_id", "store_id", "state_id")
DAYS = tuple(f"d_{number}" for number in range(1, HISTORY_DAYS + 1))
DAY_PATTERN = re.compile(r"^d_[1-9][0-9]*$")
NUMBER_PATTERN = re.compile(
    r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$"
)

MANIFEST_COLUMNS = [
    "selection_rank",
    "id",
    "item_id",
    "store_id",
    "dept_id",
    "train_total_units",
    "train_positive_days",
    "train_zero_sales_pct",
]
PREPARED_COLUMNS = [
    "id",
    "item_id",
    "store_id",
    "dept_id",
    "d",
    "day_number",
    "date",
    "sales",
    "split",
]
EXPECTED_SPLIT_COUNTS = {
    "train": TRAIN_END,
    "validation": VALIDATION_END - TRAIN_END,
    "test": HISTORY_DAYS - VALIDATION_END,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def short(value: Any, length: int = 100) -> str:
    text = str(value)
    return text if len(text) <= length else text[:length] + "..."


def reject_json_constant(token: str) -> None:
    raise ValueError(f"Invalid JSON numeric constant: {token}")


def check_prerequisite_report() -> None:
    """Use only the documented Stage 2A report contract."""
    try:
        with VALIDATION_PATH.open("r", encoding="utf-8-sig") as handle:
            report = json.load(handle, parse_constant=reject_json_constant)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Cannot read a valid Stage 2A report at {VALIDATION_PATH}: {exc}"
        ) from exc

    require(isinstance(report, dict), "Stage 2A report must be a JSON object.")
    require(
        report.get("status") == "PASS",
        "Stage 2A report must have status 'PASS'.",
    )
    require(
        isinstance(report.get("errors"), list) and not report["errors"],
        "Stage 2A report must contain an empty errors list.",
    )

    coverage = report.get("coverage")
    require(
        isinstance(coverage, dict),
        "Stage 2A report must contain a coverage object.",
    )
    require(
        coverage.get("sales_stream_completed") is True,
        "Stage 2A sales_stream_completed must be boolean true.",
    )
    require(
        coverage.get("calendar_read_completed") is True,
        "Stage 2A calendar_read_completed must be boolean true.",
    )


def check_header(
    header: list[str], required: tuple[str, ...], source: str
) -> None:
    require(bool(header), f"{source}: CSV header is empty.")
    blank_positions = [
        index for index, name in enumerate(header, start=1) if not name.strip()
    ]
    require(
        not blank_positions,
        f"{source}: blank column headers at positions {blank_positions[:5]}.",
    )
    counts = Counter(header)
    duplicates = [name for name, count in counts.items() if count > 1]
    require(
        not duplicates,
        f"{source}: duplicate column headers: {duplicates[:5]}.",
    )
    missing = [name for name in required if name not in counts]
    require(
        not missing,
        f"{source}: missing required columns: {missing}.",
    )


def read_sales_header() -> list[str]:
    """Opening the header also confirms sales-file existence and readability."""
    with SALES_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle, strict=True), [])

    check_header(header, IDENTIFIERS, SALES_PATH.name)
    daily = sorted(
        (name for name in header if DAY_PATTERN.fullmatch(name)),
        key=lambda name: int(name[2:]),
    )
    require(
        daily == list(DAYS),
        "Sales daily columns must be exactly d_1 through d_1913. "
        f"Missing examples: {[day for day in DAYS if day not in set(daily)][:5]}; "
        f"unexpected examples: {[day for day in daily if day not in set(DAYS)][:5]}.",
    )
    return header


def parse_sales(raw: Any, series_id: str, day: str) -> int:
    """Parse exactly; never replace missing or malformed tokens with zero."""
    location = f"series {short(series_id)!r}, {day}"
    require(
        isinstance(raw, str) and bool(raw.strip()),
        f"Missing sales value for {location}.",
    )
    token = raw.strip()
    require(
        NUMBER_PATTERN.fullmatch(token) is not None,
        f"Non-numeric or non-finite sales value {short(raw)!r} for {location}.",
    )
    try:
        value = Decimal(token)
    except InvalidOperation as exc:
        raise ValueError(
            f"Invalid numeric sales value {short(raw)!r} for {location}."
        ) from exc

    require(value.is_finite(), f"Non-finite sales value for {location}.")
    require(value >= 0, f"Negative sales value for {location}.")
    require(
        value == value.to_integral_value(),
        f"Fractional sales value {short(raw)!r} for {location}.",
    )
    # Python integers preserve exact values without silent int64 overflow.
    return int(value)


def training_statistics(values: tuple[int, ...]) -> dict[str, Any]:
    training = values[:TRAIN_END]
    positive_days = sum(value > 0 for value in training)
    zero_days = sum(value == 0 for value in training)
    return {
        "train_total_units": sum(training),
        "train_positive_days": positive_days,
        "train_zero_sales_pct": 100.0 * zero_days / TRAIN_END,
    }


def load_target_sales(
    header: list[str],
) -> tuple[list[dict[str, Any]], int]:
    """Stream the complete file once after its separate header inspection."""
    targets: list[dict[str, Any]] = []
    id_counts: Counter[str] = Counter()
    total_rows = 0

    # Raw strings and disabled NA inference distinguish blanks, malformed
    # values, and valid zeros. All CSV columns are parsed so extra fields are
    # not hidden by usecols. Only target rows survive beyond each chunk.
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.ParserWarning)
        with pd.read_csv(
            SALES_PATH,
            encoding="utf-8-sig",
            dtype=str,
            na_filter=False,
            keep_default_na=False,
            chunksize=CHUNK_SIZE,
            skip_blank_lines=False,
            index_col=False,
            on_bad_lines="error",
        ) as chunks:
            for chunk in chunks:
                require(
                    list(chunk.columns) == header,
                    "Sales columns differ from the inspected header.",
                )
                total_rows += len(chunk)

                # Count IDs globally to detect a target ID duplicated even
                # outside the requested slice or in a later chunk.
                for raw_id in chunk["id"]:
                    if isinstance(raw_id, str) and raw_id.strip():
                        id_counts[raw_id] += 1

                target_mask = (
                    chunk["store_id"].eq(STORE)
                    & chunk["dept_id"].eq(DEPARTMENT)
                )
                target = chunk.loc[
                    target_mask, list(IDENTIFIERS) + list(DAYS)
                ]
                for row in target.itertuples(index=False, name=None):
                    identifiers = dict(zip(IDENTIFIERS, row[:len(IDENTIFIERS)]))
                    for name, value in identifiers.items():
                        require(
                            isinstance(value, str) and bool(value.strip()),
                            f"Target series has an invalid {name}: {short(value)!r}.",
                        )

                    series_id = identifiers["id"]
                    values = tuple(
                        parse_sales(raw, series_id, day)
                        for day, raw in zip(DAYS, row[len(IDENTIFIERS):])
                    )
                    require(
                        len(values) == HISTORY_DAYS,
                        f"Series {series_id!r} does not have 1913 sales values.",
                    )
                    targets.append(
                        {
                            **identifiers,
                            "sales_values": values,
                            **training_statistics(values),
                        }
                    )

    require(
        bool(targets),
        f"No sales series exist for {STORE} / {DEPARTMENT}.",
    )
    duplicated = [
        row["id"] for row in targets if id_counts[row["id"]] != 1
    ]
    require(
        not duplicated,
        "Target series IDs must be globally unique. "
        f"Duplicate examples: {sorted(set(duplicated))[:5]}.",
    )
    return targets, total_rows


def load_calendar() -> dict[str, date]:
    """Read the calendar exactly once, including its header."""
    mappings: dict[str, date] = {}
    historical_days = set(DAYS)

    with CALENDAR_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        header = next(reader, [])
        check_header(header, ("d", "date"), CALENDAR_PATH.name)
        day_index = header.index("d")
        date_index = header.index("date")

        for record_number, row in enumerate(reader, start=1):
            require(
                len(row) == len(header),
                f"Calendar record {record_number} has {len(row)} fields; "
                f"expected {len(header)}.",
            )
            day = row[day_index]
            if day not in historical_days:
                continue

            require(
                day not in mappings,
                f"Duplicate historical calendar mapping for {day}.",
            )
            raw_date = row[date_index]
            require(
                re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", raw_date) is not None,
                f"Calendar date for {day} must be a populated YYYY-MM-DD value; "
                f"received {short(raw_date)!r}.",
            )
            try:
                mappings[day] = date.fromisoformat(raw_date)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid calendar date for {day}: {raw_date!r}."
                ) from exc

    missing = [day for day in DAYS if day not in mappings]
    require(
        not missing,
        f"Missing historical calendar mappings; examples: {missing[:5]}.",
    )
    for previous, current in zip(DAYS, DAYS[1:]):
        require(
            mappings[current] - mappings[previous] == timedelta(days=1),
            f"Calendar dates are not consecutive at {previous} -> {current}: "
            f"{mappings[previous]} -> {mappings[current]}.",
        )
    return mappings


def ranking_key(row: dict[str, Any]) -> tuple[int, str, str]:
    return (-row["train_total_units"], row["item_id"], row["id"])


def select_products(
    targets: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Complete-history validity has already been checked. Neither activity
    # calculations nor ranking access validation-period or test-period sales.
    eligible = sorted(
        (
            row for row in targets
            if row["train_positive_days"] >= MIN_POSITIVE_DAYS
        ),
        key=ranking_key,
    )
    require(
        bool(eligible),
        "No target series meet the minimum 28 positive training-sales days.",
    )
    return eligible, eligible[:MAX_SERIES]


def split_for_day(day_number: int) -> str:
    if day_number <= TRAIN_END:
        return "train"
    if day_number <= VALIDATION_END:
        return "validation"
    return "test"


def build_outputs(
    selected: list[dict[str, Any]],
    calendar: dict[str, date],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_records = []
    for rank, row in enumerate(selected, start=1):
        manifest_records.append(
            {
                "selection_rank": rank,
                "id": row["id"],
                "item_id": row["item_id"],
                "store_id": row["store_id"],
                "dept_id": row["dept_id"],
                "train_total_units": row["train_total_units"],
                "train_positive_days": row["train_positive_days"],
                "train_zero_sales_pct": row["train_zero_sales_pct"],
            }
        )
    manifest = pd.DataFrame(
        manifest_records, columns=MANIFEST_COLUMNS, dtype=object
    )

    records = []
    for row in sorted(selected, key=lambda record: record["id"]):
        for day_number, sales in enumerate(row["sales_values"], start=1):
            day = f"d_{day_number}"
            records.append(
                (
                    row["id"],
                    row["item_id"],
                    row["store_id"],
                    row["dept_id"],
                    day,
                    day_number,
                    calendar[day].isoformat(),
                    sales,
                    split_for_day(day_number),
                )
            )
    prepared = pd.DataFrame(
        records, columns=PREPARED_COLUMNS, dtype=object
    )
    return manifest, prepared


def validate_outputs(
    targets: list[dict[str, Any]],
    eligible: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    manifest: pd.DataFrame,
    prepared: pd.DataFrame,
    calendar: dict[str, date],
) -> dict[str, Any]:
    """Validate structures and reconcile individual source cells."""
    selected_ids = [row["id"] for row in selected]
    require(
        1 <= len(selected_ids) <= MAX_SERIES,
        "Selected count must be between 1 and 20.",
    )
    require(
        all(isinstance(value, str) and value.strip() for value in selected_ids),
        "Selected IDs must be nonempty strings.",
    )
    require(
        len(set(selected_ids)) == len(selected_ids),
        "Selected IDs are not unique.",
    )

    # Independently recalculate training statistics from retained source values.
    recalculated = []
    for source in targets:
        statistics = training_statistics(source["sales_values"])
        require(
            all(source[name] == value for name, value in statistics.items()),
            f"Training statistics disagree with source for {source['id']}.",
        )
        if statistics["train_positive_days"] >= MIN_POSITIVE_DAYS:
            recalculated.append({**source, **statistics})
    recalculated.sort(key=ranking_key)
    expected_ids = [row["id"] for row in recalculated[:MAX_SERIES]]
    require(
        [row["id"] for row in eligible] == [
            row["id"] for row in recalculated
        ],
        "Eligible pool does not match training-only recalculation.",
    )
    require(
        selected_ids == expected_ids,
        "Selection does not match the documented training-only ranking.",
    )

    require(
        list(manifest.columns) == MANIFEST_COLUMNS,
        "Manifest columns are incorrect.",
    )
    require(
        list(prepared.columns) == PREPARED_COLUMNS,
        "Prepared-sales columns are incorrect.",
    )
    require(
        len(manifest) == len(selected),
        "Manifest row count differs from selected count.",
    )
    require(
        manifest["id"].tolist() == selected_ids,
        "Manifest ordering does not match selection ranking.",
    )
    require(
        manifest["selection_rank"].tolist() == list(range(1, len(selected) + 1)),
        "Manifest ranks must be consecutive integers beginning at 1.",
    )
    require(
        not manifest.isna().any().any(),
        "Manifest contains missing values.",
    )

    for exported, source in zip(
        manifest.to_dict("records"), selected
    ):
        for name in MANIFEST_COLUMNS:
            if name == "selection_rank":
                continue
            require(
                exported[name] == source[name],
                f"Manifest field {name} disagrees with source for {source['id']}.",
            )
        require(
            source["train_positive_days"] >= MIN_POSITIVE_DAYS,
            f"Selected series {source['id']} fails the activity threshold.",
        )

    require(
        len(prepared) == len(selected) * HISTORY_DAYS,
        "Prepared-sales total row count is incorrect.",
    )
    require(
        not prepared.isna().any().any(),
        "Prepared-sales data contains missing values.",
    )
    for name in ("id", "item_id", "store_id", "dept_id", "d", "date", "split"):
        require(
            all(
                isinstance(value, str) and bool(value.strip())
                for value in prepared[name]
            ),
            f"Prepared field {name} contains blank or invalid strings.",
        )
    require(
        not prepared.duplicated(["id", "day_number"]).any(),
        "Prepared (id, day_number) keys are duplicated.",
    )
    require(
        set(prepared["id"]) == set(selected_ids),
        "Prepared IDs differ from selected IDs.",
    )
    require(
        all(type(value) is int and value >= 0 for value in prepared["sales"]),
        "Prepared sales must be nonnegative Python integers.",
    )
    require(
        all(
            type(value) is int and 1 <= value <= HISTORY_DAYS
            for value in prepared["day_number"]
        ),
        "Prepared day numbers must be integers from 1 through 1913.",
    )
    ordered = prepared.sort_values(
        ["id", "day_number"], kind="stable"
    ).reset_index(drop=True)
    require(
        prepared.reset_index(drop=True).equals(ordered),
        "Prepared rows are not sorted by id and day_number.",
    )

    for series_id, group in prepared.groupby("id", sort=True):
        require(
            group["day_number"].tolist() == list(range(1, HISTORY_DAYS + 1)),
            f"Series {series_id} does not contain every historical day once.",
        )
        counts = group["split"].value_counts().to_dict()
        require(
            counts == EXPECTED_SPLIT_COUNTS,
            f"Incorrect split counts for {series_id}: {counts}.",
        )

    source_by_id = {row["id"]: row for row in selected}
    mismatch_count = 0
    compared_values = 0
    for row in prepared.itertuples(index=False):
        source = source_by_id[row.id]
        require(
            row.item_id == source["item_id"]
            and row.store_id == source["store_id"]
            and row.dept_id == source["dept_id"],
            f"Prepared identifiers disagree with source for {row.id}.",
        )
        expected_day = f"d_{row.day_number}"
        require(
            row.d == expected_day,
            f"Incorrect day ID for {row.id}, day {row.day_number}.",
        )
        require(
            row.date == calendar[expected_day].isoformat(),
            f"Incorrect exported calendar date for {row.id}, {expected_day}.",
        )
        require(
            row.split == split_for_day(row.day_number),
            f"Incorrect split assignment for {row.id}, {expected_day}.",
        )
        compared_values += 1
        if row.sales != source["sales_values"][row.day_number - 1]:
            mismatch_count += 1

    require(
        mismatch_count == 0,
        f"Source-to-export reconciliation failed: "
        f"{mismatch_count} mismatched values out of {compared_values}.",
    )
    return {
        "status": "PASS",
        "comparison_key": ["id", "day_number"],
        "comparison_method": "Exact integer equality for every exported sales cell",
        "compared_values": compared_values,
        "mismatched_values": mismatch_count,
    }


def build_metadata(
    targets: list[dict[str, Any]],
    eligible: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    total_source_rows: int,
    prepared: pd.DataFrame,
    calendar: dict[str, date],
    reconciliation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset": "Walmart M5 Forecasting Accuracy",
        "source_filename": SALES_PATH.name,
        "calendar_source_filename": CALENDAR_PATH.name,
        "store_id": STORE,
        "dept_id": DEPARTMENT,
        "selection_method": "Deterministic high-volume training-sales ranking",
        "historical_quality_gate": {
            "required_days": ["d_1", "d_1913"],
            "required_day_count": HISTORY_DAYS,
            "requirements": [
                "All six required identifiers are nonempty strings.",
                "Target series IDs are globally unique.",
                "Every historical sales value is present, finite, numeric, "
                "nonnegative, and integral.",
            ],
            "failure_behavior": (
                "Invalid target identifiers or historical sales fail preparation; "
                "source rows are not silently repaired or discarded."
            ),
        },
        "training_only_eligibility_criteria": {
            "first_day": "d_1",
            "last_day": "d_1857",
            "minimum_positive_sales_days": MIN_POSITIVE_DAYS,
            "positive_sales_definition": "Daily integer unit sales greater than zero",
            "activity_and_ranking_use_validation_or_test_sales": False,
        },
        "ranking_rules": [
            {"field": "train_total_units", "order": "descending"},
            {"field": "item_id", "order": "ascending"},
            {"field": "id", "order": "ascending"},
        ],
        "maximum_requested_series": MAX_SERIES,
        "actual_selected_series_count": len(selected),
        "eligible_series_count": len(eligible),
        "target_series_count": len(targets),
        "total_source_sales_rows": total_source_rows,
        "selected_series_ids_in_rank_order": [row["id"] for row in selected],
        "splits": {
            "train": {
                "first_day": "d_1",
                "last_day": "d_1857",
                "first_day_number": 1,
                "last_day_number": TRAIN_END,
                "rows_per_series": TRAIN_END,
            },
            "validation": {
                "first_day": "d_1858",
                "last_day": "d_1885",
                "first_day_number": TRAIN_END + 1,
                "last_day_number": VALIDATION_END,
                "rows_per_series": 28,
            },
            "test": {
                "first_day": "d_1886",
                "last_day": "d_1913",
                "first_day_number": VALIDATION_END + 1,
                "last_day_number": HISTORY_DAYS,
                "rows_per_series": 28,
            },
        },
        "forecast_horizon_days": FORECAST_HORIZON,
        "total_exported_rows": len(prepared),
        "exported_rows_by_split": {
            split: int((prepared["split"] == split).sum())
            for split in EXPECTED_SPLIT_COUNTS
        },
        "source_to_export_reconciliation": reconciliation,
        "calendar_integrity": {
            "status": "PASS",
            "historical_mappings_checked": len(calendar),
            "missing_historical_mappings": 0,
            "duplicate_historical_mappings": 0,
            "dates_strictly_increasing": True,
            "dates_consecutive": True,
            "adjacent_day_interval_days": 1,
            "first_historical_date": calendar["d_1"].isoformat(),
            "last_historical_date": calendar["d_1913"].isoformat(),
        },
        "integrity_checks": {
            "status": "PASS",
            "product_selection": "PASS",
            "dataset_structure": "PASS",
            "chronological_splits": "PASS",
            "calendar_mapping": "PASS",
            "exact_source_reconciliation": "PASS",
        },
        "selection_limitations": [
            "This high-volume subset is not a statistically representative "
            "sample of FOODS_3 or the wider M5 dataset.",
            "Only CA_1 / FOODS_3 is eligible for selection.",
            "Complete historical values are an integrity prerequisite; "
            "validation/test demand does not enter activity or ranking calculations.",
            "The existing Stage 2A report is a prerequisite, not a source-file "
            "fingerprint; relevant source integrity is rechecked during preparation.",
            "Calendar validation covers historical d_1 through d_1913.",
            "No missing-value imputation, cleaning, forecasting, or inventory "
            "calculations are performed.",
        ],
        "raw_inputs_modified": False,
        "stage_2a_report_modified": False,
        "publication": {
            "individual_file_replacement": "atomic",
            "multiple_file_transaction": False,
            "rerun_behavior": (
                "A successful rerun replaces all three artifacts with a "
                "consistent newly constructed set."
            ),
        },
    }


def publish_outputs(
    manifest: pd.DataFrame,
    prepared: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    """Stage every artifact before replacing any final file."""
    metadata_text = json.dumps(
        metadata, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    staged: list[tuple[Path, Path]] = []
    replaced: list[Path] = []
    try:
        for destination, frame in (
            (MANIFEST_PATH, manifest),
            (PREPARED_PATH, prepared),
            (METADATA_PATH, None),
        ):
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=OUTPUT_DIR,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                staged.append((temporary, destination))
                if frame is None:
                    handle.write(metadata_text)
                else:
                    frame.to_csv(
                        handle,
                        index=False,
                        lineterminator="\n",
                    )

        # Each replacement is atomic individually. The set of three
        # replacements is explicitly not one atomic transaction.
        for temporary, destination in staged:
            temporary.replace(destination)
            replaced.append(destination)
    except Exception as exc:
        replaced_names = ", ".join(path.name for path in replaced) or "none"
        raise RuntimeError(
            "Output publication failed. Files replaced during this attempt: "
            f"{replaced_names}. Outputs may be from different runs. "
            "Resolve the underlying problem and rerun preparation successfully "
            f"before using the outputs. Original error: {exc}"
        ) from exc
    finally:
        for temporary, _ in staged:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as exc:
                print(
                    f"WARNING: Could not remove temporary file {temporary}: {exc}",
                    file=sys.stderr,
                )


def main() -> int:
    try:
        check_prerequisite_report()
        header = read_sales_header()
        # Reading here confirms calendar readability and integrity before
        # spending time streaming the much larger sales file.
        calendar = load_calendar()
        targets, total_source_rows = load_target_sales(header)
        eligible, selected = select_products(targets)
        manifest, prepared = build_outputs(selected, calendar)

        reconciliation = validate_outputs(
            targets,
            eligible,
            selected,
            manifest,
            prepared,
            calendar,
        )
        metadata = build_metadata(
            targets,
            eligible,
            selected,
            total_source_rows,
            prepared,
            calendar,
            reconciliation,
        )
        publish_outputs(manifest, prepared, metadata)

    except Exception as exc:
        print(
            f"Data preparation FAILED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print(
            "Existing outputs, if present, must not be interpreted as results "
            "of a successful current run.",
            file=sys.stderr,
        )
        return 1

    if len(eligible) < MAX_SERIES:
        print(
            f"WARNING: Only {len(eligible)} eligible series qualified; "
            f"selected all {len(selected)}."
        )
    print("Data preparation: PASS")
    print(f"Eligible series: {len(eligible)}")
    print(f"Selected series: {len(selected)}")
    print(f"Total exported rows: {len(prepared)}")
    for split, count in metadata["exported_rows_by_split"].items():
        print(f"  {split}: {count}")
    print(
        "Source reconciliation mismatches: "
        f"{reconciliation['mismatched_values']}"
    )
    print("All mandatory integrity checks passed.")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Prepared sales: {PREPARED_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    print("Exit code: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())