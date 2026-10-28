"""Frozen XGBoost holdout forecasting and separate one-attempt scoring.

Commands:
    python -m src.evaluate_holdout forecast
    python -m src.evaluate_holdout score

Forecasting may access sales only through d_1885. Scoring verifies the
published predictions before accessing d_1886..d_1913 actuals.

A scoring attempt is committed before holdout access. Failed, interrupted,
or successful scoring attempts cannot automatically repeat.

No claim is made about evaluations outside this script.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import inspect
import io
import json
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import src.evaluate as evaluation_module
import src.forecast_xgboost as experiment
import src.prepare_xgb_features as feature_module


ROOT = Path(__file__).resolve().parents[1]
CUTOFF = 1885
FIRST_HOLDOUT_DAY = 1886
LAST_HOLDOUT_DAY = 1913
HORIZON = 28
PRODUCT_COUNT = 20
CONFIGURATION = "A_squared_error"
MODEL = "xgboost"

FEATURE_NAMES = (
    "id",
    "target_weekday",
    "target_month",
    "target_event_type_1",
    "target_event_type_2",
    "origin_sales",
    "lag_7",
    "lag_14",
    "lag_28",
    "lag_56",
    "mean_7",
    "mean_28",
    "mean_56",
    "std_7",
    "std_28",
    "std_56",
    "zero_pct_28",
    "zero_pct_56",
    "same_weekday_mean_28",
    "horizon",
    "target_snap_CA",
    "target_has_event",
    "last_known_price",
    "price_change_pct",
    "trailing_4_observed_week_mean_price",
    "price_relative_to_trailing_mean",
    "weeks_since_last_observed_price_change",
    "recorded_price_weeks_to_origin",
    "price_history_missing_indicator",
)
CATEGORICAL_NAMES = (
    "id",
    "target_weekday",
    "target_month",
    "target_event_type_1",
    "target_event_type_2",
)
MODEL_PARAMETERS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_lambda": 5,
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": 4,
    "objective": "reg:squarederror",
}
EXPECTED_ENCODING = {
    "method": (
        "Dense OneHotEncoder; fitted independently per "
        "configuration and partition"
    ),
    "categorical_predictors": list(CATEGORICAL_NAMES),
    "unknown_product_ids": "Rejected",
    "unseen_calendar_categories": (
        "Encoded as an all-zero indicator block and explicitly "
        "recorded in each fit's diagnostics."
    ),
    "numeric_scaling": "None",
    "matrix_dtype": "float32",
    "missing_prices": "Preserved as NaN; rows retained",
}
ORIGINAL_SCRIPT_HASHES = {
    "src/forecast_xgboost.py": (
        "9D7934B723235628A6FCDAE180EDB80A8BEEECDFC8EE62FCA1A6A1528AC892AB"
    ),
    "src/prepare_xgb_features.py": (
        "DD2AAE1312C47661B90D81F42DAD245F3E0C29135EFD8D482599CF38F134040D"
    ),
}

FROZEN_COLUMNS = (
    "id",
    "item_id",
    "model",
    "forecast_origin_day",
    "horizon",
    "d",
    "date",
    "raw_predicted_sales",
    "predicted_sales",
)
SCORED_COLUMNS = (
    "id",
    "item_id",
    "model",
    "forecast_origin_day",
    "horizon",
    "d",
    "date",
    "actual_sales",
    "raw_predicted_sales",
    "predicted_sales",
)
METRIC_NAMES = ("mae", "rmse", "wape_pct", "bias_units")
PRODUCT_COLUMNS = (
    "id", "item_id", "model", "observations", *METRIC_NAMES,
)
SUMMARY_COLUMNS = (
    "model", "evaluation_level", "observations", *METRIC_NAMES,
)

FROZEN_NAME = "holdout_predictions_frozen.csv"
SCORED_NAME = "holdout_predictions_scored.csv"
PRODUCT_NAME = "holdout_metrics_by_product.csv"
SUMMARY_NAME = "holdout_metrics_summary.csv"
METADATA_NAME = "holdout_run_metadata.json"

ALL_OUTPUT_NAMES = (
    FROZEN_NAME,
    SCORED_NAME,
    PRODUCT_NAME,
    SUMMARY_NAME,
    METADATA_NAME,
)
SCORING_OUTPUT_NAMES = (SCORED_NAME, PRODUCT_NAME, SUMMARY_NAME)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def reject_json_constant(token: str) -> None:
    raise ValueError(f"Invalid JSON constant: {token}")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        result = json.load(
            handle,
            parse_constant=reject_json_constant,
        )
    require(isinstance(result, dict), f"{path.name}: expected a JSON object.")
    return result


def json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
    ).encode("utf-8")


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(
        index=False,
        na_rep="",
        lineterminator="\n",
    ).encode("utf-8")


def output_directory(root: Path) -> Path:
    return root / "outputs/evaluation"


def source_paths(root: Path) -> dict[str, Path]:
    """Exclude selected_sales.csv: its full hash is deferred until scoring."""
    return {
        "manifest": root / "data/processed/selected_products.csv",
        "selection_metadata": root / "data/processed/selection_metadata.json",
        "calendar": root / "data/raw/calendar.csv",
        "prices": root / "data/raw/sell_prices.csv",
        "xgb_metadata": root / "outputs/evaluation/xgb_run_metadata.json",
        "feature_script": root / "src/prepare_xgb_features.py",
        "model_script": root / "src/forecast_xgboost.py",
        "metric_script": root / "src/evaluate.py",
        "holdout_script": root / "src/evaluate_holdout.py",
    }


def require_parameter(
    function: Any,
    parameter: str,
    default: int,
) -> None:
    require(callable(function), "Required integration helper is unavailable.")
    parameters = inspect.signature(function).parameters
    require(
        parameter in parameters
        and parameters[parameter].default == default
        and parameters[parameter].kind
        == inspect.Parameter.KEYWORD_ONLY,
        f"Compatibility adapter required: {function.__module__}."
        f"{function.__name__} needs keyword-only "
        f"{parameter}={default}. See Stage 5A compatibility requirements.",
    )


def check_compatibility() -> None:
    """Fail explicitly rather than override globals or recreate feature logic."""
    adapters = (
        (feature_module.load_calendar, "max_day", 1885),
        (feature_module.training_weeks, "cutoff", 1857),
        (feature_module.checked_history_prefix, "max_origin", 1857),
        (feature_module.eligible_price_weeks, "max_origin", 1857),
        (feature_module.historical_price_features, "max_origin", 1857),
        (feature_module.build_feature_rows, "max_origin", 1857),
        (feature_module.build_feature_rows, "max_forecast_day", 1885),
        (feature_module.build_feature_row, "max_origin", 1857),
        (feature_module.build_feature_row, "max_forecast_day", 1885),
        (feature_module.build_training_examples, "fitting_cutoff", 1857),
        (feature_module.build_training_examples, "max_forecast_day", 1885),
        (experiment.generate_origins, "max_cutoff", 1857),
        (experiment.build_fit_examples, "max_fitting_cutoff", 1857),
        (experiment.build_fit_examples, "max_forecast_day", 1885),
        (experiment.prepare_partition, "max_fitting_cutoff", 1857),
        (experiment.prepare_partition, "max_forecast_day", 1885),
        (experiment.fit_and_forecast, "max_fitting_cutoff", 1857),
        (experiment.fit_and_forecast, "max_forecast_day", 1885),
        (experiment.load_demand_window, "max_permitted_day", 1885),
    )
    for function, name, default in adapters:
        require_parameter(function, name, default)

    require(
        feature_module.TRAIN_END == 1857
        and feature_module.VALIDATION_END == 1885
        and experiment.TRAIN_END == 1857
        and experiment.VALIDATION_END == 1885,
        "Stage 4 constants/default boundaries must remain unchanged.",
    )
    require(
        tuple(experiment.PREDICTOR_COLUMNS) == FEATURE_NAMES
        and tuple(experiment.CATEGORICAL_PREDICTORS) == CATEGORICAL_NAMES,
        "Existing predictor interface differs from the frozen 29 features.",
    )
    require(
        callable(evaluation_module.calculate_metrics)
        and callable(evaluation_module.evaluate_records),
        "Required src.evaluate interfaces are unavailable.",
    )


@contextmanager
def run_lock(root: Path):
    directory = output_directory(root)
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".holdout_run.lock"
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        )
    except FileExistsError as exc:
        raise ValueError(
            "A holdout lock exists. Another or interrupted run requires review."
        ) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()}\n")
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".holdout-state-",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
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


def logical_demand_sha256(histories: dict[str, list[int]]) -> str:
    """Hash only parsed observations supplied by the guarded history reader."""
    hasher = hashlib.sha256()
    for series_id in sorted(histories):
        for day, value in enumerate(histories[series_id], start=1):
            record = json.dumps(
                [series_id, day, value],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            hasher.update((record + "\n").encode("utf-8"))
    return hasher.hexdigest()


def load_frozen_configuration(
    root: Path,
    manifest: pd.DataFrame,
) -> dict[str, Any]:
    metadata = read_json(
        root / "outputs/evaluation/xgb_run_metadata.json"
    )
    require(
        metadata.get("model") == MODEL
        and metadata.get("selected_configuration") == CONFIGURATION
        and metadata.get("selected_objective") == "reg:squarederror",
        "Frozen model/configuration/objective differs.",
    )

    configurations = metadata.get("model_configurations")
    require(
        isinstance(configurations, dict)
        and configurations.get(CONFIGURATION) == MODEL_PARAMETERS
        and configurations == experiment.CONFIGURATIONS,
        "Frozen configuration dictionaries differ from the existing model.",
    )
    require(
        metadata.get("feature_names") == list(FEATURE_NAMES)
        and tuple(experiment.PREDICTOR_COLUMNS) == FEATURE_NAMES,
        "Frozen feature names or order differ.",
    )
    require(
        len(manifest) == PRODUCT_COUNT
        and type(metadata.get("selected_product_count")) is int
        and metadata["selected_product_count"] == PRODUCT_COUNT,
        "Frozen run requires the same 20 selected products.",
    )

    # This key is emitted by the previously documented Stage 4 implementation.
    # Missing population provenance is a blocking interface mismatch.
    require(
        metadata.get("selected_product_ids_in_rank_order")
        == manifest["id"].tolist(),
        "Frozen population provenance is missing or differs: expected "
        "selected_product_ids_in_rank_order matching the manifest.",
    )
    require(
        metadata.get("scope") == {
            "store_id": "CA_1",
            "dept_id": "FOODS_3",
        },
        "Frozen scope differs.",
    )
    windows = metadata.get("chronological_windows")
    require(
        isinstance(windows, dict)
        and windows.get("training") == [1, 1857]
        and windows.get("validation") == [1858, 1885]
        and windows.get("excluded_final_test") == [1886, 1913]
        and windows.get("forecast_horizon") == HORIZON,
        "Frozen chronological design differs.",
    )
    encoding = metadata.get("categorical_encoding")
    require(
        isinstance(encoding, dict)
        and all(
            encoding.get(name) == value
            for name, value in EXPECTED_ENCODING.items()
        ),
        "Frozen preprocessing descriptor differs.",
    )
    return metadata


def original_script_hashes() -> dict[str, str]:
    result = dict(ORIGINAL_SCRIPT_HASHES)
    require(
        all(
            re.fullmatch(r"[0-9A-F]{64}", digest) is not None
            for digest in result.values()
        ),
        "Invalid supplied original source hash.",
    )
    return result


def finite_number(value: Any, context: str) -> float:
    require(
        not isinstance(value, (bool, np.bool_)),
        f"{context}: boolean value is invalid.",
    )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{context}: invalid numeric value.") from exc
    require(np.isfinite(result), f"{context}: non-finite value.")
    return result


def verify_frozen_predictions(
    frame: pd.DataFrame,
    manifest: pd.DataFrame,
    calendar: dict[int, dict[str, Any]],
) -> None:
    require(
        tuple(frame.columns) == FROZEN_COLUMNS
        and "actual_sales" not in frame.columns,
        "Frozen schema must exclude actual sales.",
    )
    require(
        len(manifest) == PRODUCT_COUNT
        and not manifest["id"].duplicated().any(),
        "Exactly 20 unique selected products are required.",
    )
    require(
        len(frame) == PRODUCT_COUNT * HORIZON
        and set(frame["id"]) == set(manifest["id"])
        and not frame.isna().any().any()
        and not frame.duplicated(["id", "model", "horizon"]).any()
        and not frame.duplicated(["id", "d"]).any(),
        "Frozen count, completeness, or uniqueness failure.",
    )
    require(
        frame.reset_index(drop=True).equals(
            frame.sort_values(
                ["model", "id", "horizon"]
            ).reset_index(drop=True)
        ),
        "Frozen rows must be sorted by model, id, horizon.",
    )
    items = manifest.set_index("id")["item_id"].to_dict()
    for series_id, group in frame.groupby("id", sort=True):
        require(
            group["horizon"].tolist() == list(range(1, HORIZON + 1)),
            f"{series_id}: incomplete horizons.",
        )
        for row in group.itertuples(index=False):
            require(
                isinstance(row.horizon, (int, np.integer))
                and not isinstance(row.horizon, (bool, np.bool_)),
                "Horizons must be integers.",
            )
            day = CUTOFF + row.horizon
            raw = finite_number(row.raw_predicted_sales, "Raw prediction")
            clipped = finite_number(row.predicted_sales, "Clipped prediction")
            require(
                row.model == MODEL
                and row.item_id == items[series_id]
                and row.forecast_origin_day == "d_1885"
                and FIRST_HOLDOUT_DAY <= day <= LAST_HOLDOUT_DAY
                and row.d == f"d_{day}"
                and row.date == calendar[day]["date"].isoformat()
                and clipped == max(0.0, raw),
                "Frozen origin/calendar/identifier/clipping mismatch.",
            )


def read_frozen_predictions(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Hash the exact bytes before decoding or parsing CSV records."""
    payload = path.read_bytes()
    digest = sha256_bytes(payload)

    if expected_sha256 is not None:
        require(
            isinstance(expected_sha256, str)
            and re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is not None,
            "Invalid recorded frozen prediction SHA256.",
        )
        require(
            digest == expected_sha256,
            "Frozen prediction SHA256 mismatch.",
        )

    reader = csv.reader(
        io.StringIO(payload.decode("utf-8-sig"), newline=""),
        strict=True,
    )
    names = next(reader, [])
    require(
        tuple(names) == FROZEN_COLUMNS,
        "Unexpected frozen CSV columns.",
    )

    records = []
    for row in reader:
        require(
            len(row) == len(names),
            "Malformed frozen prediction record.",
        )
        record = dict(zip(names, row))
        token = record["horizon"]
        require(
            token.isascii() and token.isdecimal(),
            "Malformed frozen horizon.",
        )
        record["horizon"] = int(token)
        for name in ("raw_predicted_sales", "predicted_sales"):
            record[name] = finite_number(
                record[name],
                f"Frozen {name}",
            )
        records.append(record)

    return (
        pd.DataFrame(records, columns=FROZEN_COLUMNS),
        digest,
    )

def verify_frozen_csv(expected: pd.DataFrame, path: Path) -> str:
    """Dedicated prediction-only round-trip verification, including SHA256."""
    require(
        tuple(expected.columns) == FROZEN_COLUMNS
        and "actual_sales" not in expected.columns,
        "Frozen export must use the prediction-only schema.",
    )
    require(
        len(expected) == PRODUCT_COUNT * HORIZON
        and expected["id"].nunique() == PRODUCT_COUNT
        and not expected.isna().any().any()
        and not expected.duplicated(["id", "model", "horizon"]).any()
        and not expected.duplicated(["id", "d"]).any(),
        "Frozen export count/completeness/keys are invalid.",
    )
    actual, digest = read_frozen_predictions(path)
    require(
        digest == sha256_bytes(csv_bytes(expected)),
        "Frozen CSV SHA256 differs from its canonical expected payload.",
    )
    require(
        len(actual) == len(expected)
        and not actual.isna().any().any()
        and not actual.duplicated(["id", "model", "horizon"]).any()
        and not actual.duplicated(["id", "d"]).any(),
        "Frozen round-trip completeness or keys differ.",
    )
    for column in (
        "id", "item_id", "model", "forecast_origin_day", "d", "date"
    ):
        require(
            all(
                isinstance(value, str) and value.strip()
                for value in expected[column]
            )
            and actual[column].tolist() == expected[column].tolist(),
            f"Frozen CSV exact string mismatch: {column}.",
        )
    for source, saved in zip(
        expected.itertuples(index=False),
        actual.itertuples(index=False),
    ):
        require(
            isinstance(source.horizon, (int, np.integer))
            and not isinstance(source.horizon, (bool, np.bool_))
            and saved.horizon == source.horizon
            and 1 <= saved.horizon <= HORIZON,
            "Frozen integer horizon mismatch.",
        )
        for name in ("raw_predicted_sales", "predicted_sales"):
            require(
                finite_number(getattr(source, name), name)
                == finite_number(getattr(saved, name), name),
                f"Frozen numeric round-trip mismatch: {name}.",
            )
        require(
            saved.predicted_sales == max(0.0, saved.raw_predicted_sales),
            "Frozen clipping relationship differs.",
        )
    for series_id, group in actual.groupby("id", sort=True):
        require(
            group["horizon"].tolist() == list(range(1, HORIZON + 1)),
            f"{series_id}: incomplete frozen horizons.",
        )
    require(
        sha256_file(path) == digest,
        "Frozen artifact changed during verification.",
    )
    return digest


def publish_phase(
    root: Path,
    frames: dict[str, pd.DataFrame],
    metadata: dict[str, Any],
    prior_metadata_sha256: str | None,
) -> None:
    """Stage outputs and commit metadata last.

    Scoring rollback restores SCORING_STARTED, never FORECAST_READY.
    The six-file/five-file directory contents are not an atomic transaction.
    """
    directory = output_directory(root)
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / METADATA_NAME
    prior_bytes = metadata_path.read_bytes() if metadata_path.exists() else None
    require(
        (
            prior_metadata_sha256 is None
            and prior_bytes is None
        )
        or (
            prior_metadata_sha256 is not None
            and prior_bytes is not None
            and sha256_bytes(prior_bytes) == prior_metadata_sha256
        ),
        "Holdout metadata changed before publication.",
    )

    scoring = prior_bytes is not None
    if scoring:
        prior = read_json(metadata_path)
        require(
            set(frames) == set(SCORING_OUTPUT_NAMES)
            and prior.get("status") == "SCORING_STARTED"
            and prior.get("scoring_attempts_by_this_script") == 1
            and prior.get("successful_scoring_publications_by_this_script") == 0
            and metadata.get("status") == "EVALUATED"
            and metadata.get("scoring_attempts_by_this_script") == 1
            and metadata.get(
                "successful_scoring_publications_by_this_script"
            ) == 1
            and metadata.get("frozen_prediction_sha256")
            == prior.get("frozen_prediction_sha256")
            and metadata.get("scoring_started_at_utc")
            == prior.get("scoring_started_at_utc"),
            "Invalid scoring publication state transition.",
        )
        require(
            sha256_file(directory / FROZEN_NAME)
            == metadata["frozen_prediction_sha256"],
            "Frozen artifact changed before result publication.",
        )
    else:
        require(
            set(frames) == {FROZEN_NAME}
            and metadata.get("status") == "FORECAST_READY"
            and metadata.get("scoring_attempts_by_this_script") == 0
            and metadata.get(
                "successful_scoring_publications_by_this_script"
            ) == 0,
            "Invalid forecast publication state transition.",
        )
        require(
            metadata.get("frozen_prediction_sha256")
            == sha256_bytes(csv_bytes(frames[FROZEN_NAME])),
            "Forecast metadata does not match the prediction payload hash.",
        )

    for name in frames:
        require(
            name in ALL_OUTPUT_NAMES
            and name != METADATA_NAME
            and not (directory / name).exists(),
            f"Refusing to overwrite holdout artifact {name}.",
        )

    published: dict[Path, bytes] = {}
    metadata_payload = json_bytes(metadata)
    metadata_committed = False
    try:
        with tempfile.TemporaryDirectory(
            prefix=".holdout-stage-", dir=directory
        ) as temporary:
            stage = Path(temporary)
            for name, frame in frames.items():
                path = stage / name
                with path.open("wb") as handle:
                    handle.write(csv_bytes(frame))
                    handle.flush()
                    os.fsync(handle.fileno())
                if name == FROZEN_NAME:
                    require(
                        verify_frozen_csv(frame, path)
                        == metadata["frozen_prediction_sha256"],
                        "Staged frozen artifact hash differs.",
                    )
                else:
                    # The documented Stage 4 verifier is used only for its
                    # established scored-prediction and metric schemas.
                    keys = (
                        ("id", "model", "horizon")
                        if name == SCORED_NAME
                        else ("id", "model")
                        if name == PRODUCT_NAME
                        else ("model", "evaluation_level")
                    )
                    experiment.verify_staged_csv(frame, path, keys)

            staged_metadata = stage / METADATA_NAME
            with staged_metadata.open("wb") as handle:
                handle.write(metadata_payload)
                handle.flush()
                os.fsync(handle.fileno())
            require(
                read_json(staged_metadata) == metadata,
                "Staged metadata differs.",
            )
            current = (
                metadata_path.read_bytes() if metadata_path.exists() else None
            )
            require(current == prior_bytes, "Metadata changed while staging.")

            for name in frames:
                source = stage / name
                destination = directory / name
                payload = source.read_bytes()
                require(
                    not destination.exists(),
                    f"Output appeared during publication: {name}",
                )
                source.replace(destination)
                published[destination] = payload
                require(
                    destination.read_bytes() == payload,
                    f"Published artifact differs: {name}",
                )
                if name == FROZEN_NAME:
                    verify_frozen_csv(frames[name], destination)

            if scoring:
                require(
                    sha256_file(directory / FROZEN_NAME)
                    == metadata["frozen_prediction_sha256"],
                    "Frozen artifact changed during publication.",
                )
            staged_metadata.replace(metadata_path)
            metadata_committed = True
            require(
                metadata_path.read_bytes() == metadata_payload,
                "Published metadata differs.",
            )
    except Exception as exc:
        cleanup_errors = []
        if metadata_committed:
            try:
                if prior_bytes is None:
                    metadata_path.unlink(missing_ok=True)
                else:
                    atomic_bytes(metadata_path, prior_bytes)
            except Exception as rollback_error:
                cleanup_errors.append(f"metadata rollback: {rollback_error}")
        for path, payload in published.items():
            try:
                require(
                    path.read_bytes() == payload,
                    f"Refusing to remove externally changed {path.name}.",
                )
                path.unlink()
            except Exception as cleanup_error:
                cleanup_errors.append(f"{path.name}: {cleanup_error}")
        suffix = (
            " Cleanup issues: " + "; ".join(cleanup_errors)
            if cleanup_errors else ""
        )
        raise RuntimeError(
            f"Holdout publication failed: {exc}.{suffix}"
        ) from exc


def forecast(root: Path = ROOT) -> dict[str, Any]:
    root = Path(root)
    with run_lock(root):
        directory = output_directory(root)
        require(
            not any((directory / name).exists() for name in ALL_OUTPUT_NAMES),
            "Holdout artifacts already exist. Forecast regeneration is refused.",
        )
        check_compatibility()
        sources = source_paths(root)
        sales_path = root / "data/processed/selected_sales.csv"
        for path in (*sources.values(), sales_path):
            require(path.is_file(), f"Required input missing: {path}")

        manifest = feature_module.load_selection(root)
        frozen_configuration = load_frozen_configuration(root, manifest)
        source_hashes = {
            name: {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
            }
            for name, path in sources.items()
        }

        calendar = feature_module.load_calendar(
            sources["calendar"], max_day=LAST_HOLDOUT_DAY
        )
        weeks = feature_module.training_weeks(calendar, cutoff=CUTOFF)
        prices = feature_module.load_prices(
            sources["prices"], manifest, weeks
        )
        training_hash = {}

        def training_reader(first: int, last: int) -> dict[str, list[int]]:
            require(
                first == 1 and last == CUTOFF,
                "Forecast phase permits only d_1..d_1885 history.",
            )
            histories = experiment.load_demand_window(
                sales_path, manifest, calendar, first, last,
                max_permitted_day=CUTOFF,
            )
            training_hash["sha256"] = logical_demand_sha256(histories)
            return histories

        predictions, diagnostics = experiment.fit_and_forecast(
            CUTOFF,
            CONFIGURATION,
            manifest,
            calendar,
            prices,
            training_reader,
            max_fitting_cutoff=CUTOFF,
            max_forecast_day=LAST_HOLDOUT_DAY,
        )
        require(
            "actual_sales" not in predictions.columns,
            "Forecast helper unexpectedly returned actual sales.",
        )
        require(
            set(predictions.columns) == set(FROZEN_COLUMNS),
            "Forecast helper returned an incompatible prediction schema.",
        )
        predictions = predictions[list(FROZEN_COLUMNS)]
        verify_frozen_predictions(predictions, manifest, calendar)
        require(
            "sha256" in training_hash,
            "Forecast helper did not use the guarded training reader.",
        )
        origins = experiment.generate_origins(CUTOFF, max_cutoff=CUTOFF)
        require(
            diagnostics["training_example_count"]
            == PRODUCT_COUNT * len(origins) * HORIZON
            and diagnostics["maximum_training_target_day"] <= CUTOFF
            and diagnostics["forecast_origin_day"] == "d_1885",
            "Fitting diagnostics violate the final holdout design.",
        )
        for name, path in sources.items():
            require(
                sha256_file(path) == source_hashes[name]["sha256"],
                f"Source changed during forecasting: {path}",
            )

        metadata = {
            "status": "FORECAST_READY",
            "dataset": "Walmart M5 Forecasting Accuracy",
            "model": MODEL,
            "selected_configuration": CONFIGURATION,
            "selected_objective": "reg:squarederror",
            "selected_model_parameters": frozen_configuration[
                "model_configurations"
            ][CONFIGURATION],
            "selected_product_ids_in_rank_order": manifest["id"].tolist(),
            "selected_product_count": len(manifest),
            "feature_names": list(FEATURE_NAMES),
            "feature_definitions": feature_module.FEATURE_DEFINITIONS,
            "categorical_encoding": frozen_configuration["categorical_encoding"],
            "training_cutoff": CUTOFF,
            "forecast_origin": CUTOFF,
            "holdout_boundaries": [FIRST_HOLDOUT_DAY, LAST_HOLDOUT_DAY],
            "forecast_horizon": HORIZON,
            "historical_origins": list(origins),
            "training_example_count": diagnostics["training_example_count"],
            "prediction_count": len(predictions),
            "clipping": diagnostics["clipping"],
            "fitting_diagnostics": diagnostics,
            "source_file_sha256": source_hashes,
            "selected_sales_fingerprint": {
                "path": "data/processed/selected_sales.csv",
                "training_logical_sha256": training_hash["sha256"],
                "scope": "Parsed series ID/day/sales for d_1..d_1885 only",
                "full_file_sha256": None,
                "full_file_hash_deferred_until_scoring": True,
            },
            "original_prechange_script_sha256": original_script_hashes(),
            "original_hash_provenance": (
                "Original hashes supplied by the project owner. "
                "Backups were not read or modified."
            ),
            "validation_benchmark_reference": {
                "source": "outputs/evaluation/xgb_run_metadata.json",
                "source_sha256": source_hashes["xgb_metadata"]["sha256"],
                "validation_period": [1858, 1885],
                "purpose": "Separate prior validation reference; no model selection.",
            },
            "frozen_prediction_artifact": FROZEN_NAME,
            "frozen_prediction_sha256": sha256_bytes(csv_bytes(predictions)),
            "forecast_generated_at_utc": utc_timestamp(),
            "holdout_evaluated_at_utc": None,
            "scoring_attempts_by_this_script": 0,
            "successful_scoring_publications_by_this_script": 0,
            "scoring_policy": (
                "One attempt. After the actual-access marker is committed, "
                "successful, failed, or interrupted scoring cannot automatically repeat."
            ),
            "integrity_checks": {
                "prediction_only_schema": True,
                "unique_product_day_predictions": PRODUCT_COUNT * HORIZON,
                "holdout_actuals_accessed_during_forecast": False,
                "nonnegative_unrounded_forecasts": True,
            },
            "limitations": [
                "No claim is made about evaluations outside this script.",
                "Earlier validation observations are now approved training data.",
                "Calendar/event/SNAP fields are assumed known in advance.",
                "Price availability follows the existing origin-based feature policy.",
                "Full sales-file hashing is deferred to avoid forecast-phase holdout access.",
                "The model is not claimed to be production-deployed.",
                "Individual replacements are atomic; the multi-file set is not.",
                "Interrupted runs and stale locks require manual review.",
                "File synchronization cannot guarantee protection against every "
                "hardware failure or external file modification.",
            ],
        }
        publish_phase(root, {FROZEN_NAME: predictions}, metadata, None)
        return metadata


def score_metrics(
    scored: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Delegate every metric calculation to src.evaluate."""
    product_rows = []
    for series_id, group in scored.groupby("id", sort=True):
        group = group.sort_values("horizon")
        require(
            len(group) == HORIZON
            and group["horizon"].tolist() == list(range(1, HORIZON + 1)),
            f"{series_id}: metric coverage is not 28 horizons.",
        )
        metrics = evaluation_module.calculate_metrics(
            group["actual_sales"].tolist(),
            group["predicted_sales"].tolist(),
        )
        require(metrics["observations"] == len(group), "Metric count mismatch.")
        product_rows.append({
            "id": series_id,
            "item_id": group["item_id"].iloc[0],
            "model": MODEL,
            **{name: metrics[name] for name in METRIC_NAMES},
            "observations": len(group),
        })

    records = [
        {
            "id": row.id,
            "day_number": CUTOFF + row.horizon,
            "actual": row.actual_sales,
            "predicted": row.predicted_sales,
        }
        for row in scored.itertuples(index=False)
    ]
    summary_rows = []
    for level in ("pooled_product_day", "aggregate_daily"):
        metrics = evaluation_module.evaluate_records(records, level)
        expected = PRODUCT_COUNT * HORIZON if level == "pooled_product_day" else HORIZON
        require(metrics["observations"] == expected, "Summary count mismatch.")
        summary_rows.append({
            "model": MODEL,
            "evaluation_level": level,
            "observations": metrics["observations"],
            **{name: metrics[name] for name in METRIC_NAMES},
        })
    return (
        pd.DataFrame(product_rows, columns=PRODUCT_COLUMNS, dtype=object),
        pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS, dtype=object)
        .sort_values("evaluation_level").reset_index(drop=True),
    )


def score(root: Path = ROOT) -> dict[str, Any]:
    root = Path(root)
    with run_lock(root):
        directory = output_directory(root)
        metadata_path = directory / METADATA_NAME
        metadata = read_json(metadata_path)
        require(
            metadata.get("status") == "FORECAST_READY"
            and metadata.get("scoring_attempts_by_this_script") == 0,
            "Scoring is permitted only once from FORECAST_READY. "
            "Failed, interrupted, or completed attempts require review.",
        )
        require(
            not any((directory / name).exists() for name in SCORING_OUTPUT_NAMES),
            "Scoring outputs already exist; refusing to overwrite them.",
        )

        sources = source_paths(root)
        require(
            set(metadata["source_file_sha256"]) == set(sources),
            "Source-hash coverage differs.",
        )
        for name, record in metadata["source_file_sha256"].items():
            require(
                sources[name].relative_to(root).as_posix() == record["path"]
                and sha256_file(sources[name]) == record["sha256"],
                f"Frozen source hash mismatch: {name}.",
            )
        manifest = feature_module.load_selection(root)
        frozen_configuration = load_frozen_configuration(root, manifest)
        require(
            metadata["selected_product_ids_in_rank_order"]
            == manifest["id"].tolist()
            and metadata["selected_model_parameters"]
            == frozen_configuration["model_configurations"][CONFIGURATION]
            and metadata["selected_configuration"] == CONFIGURATION
            and metadata["selected_objective"] == "reg:squarederror"
            and metadata["feature_names"] == list(FEATURE_NAMES)
            and metadata["categorical_encoding"]
            == frozen_configuration["categorical_encoding"]
            and metadata["original_prechange_script_sha256"] == ORIGINAL_SCRIPT_HASHES
            and metadata["training_cutoff"] == CUTOFF
            and metadata["forecast_origin"] == CUTOFF
            and metadata["holdout_boundaries"]
            == [FIRST_HOLDOUT_DAY, LAST_HOLDOUT_DAY]
            and metadata["forecast_horizon"] == HORIZON,
            "Holdout run metadata is inconsistent.",
        )

        calendar = feature_module.load_calendar(
            sources["calendar"], max_day=LAST_HOLDOUT_DAY
        )
        frozen_path = directory / FROZEN_NAME
        recorded_hash = metadata.get("frozen_prediction_sha256")
        require(
            isinstance(recorded_hash, str)
            and re.fullmatch(r"[0-9a-f]{64}", recorded_hash) is not None,
            "Missing or invalid frozen prediction SHA256 in metadata.",
        )
        frozen, frozen_hash = read_frozen_predictions(
            frozen_path,
            expected_sha256=recorded_hash,
        )
        verify_frozen_predictions(frozen, manifest, calendar)

        sales_path = root / "data/processed/selected_sales.csv"
        training = experiment.load_demand_window(
            sales_path, manifest, calendar, 1, CUTOFF,
            max_permitted_day=CUTOFF,
        )
        require(
            logical_demand_sha256(training)
            == metadata["selected_sales_fingerprint"]["training_logical_sha256"],
            "Training demand changed after forecast publication.",
        )
        del training
        require(
            sha256_file(frozen_path) == frozen_hash,
            "Frozen predictions changed before scoring.",
        )

        started = copy.deepcopy(metadata)
        started["status"] = "SCORING_STARTED"
        started["scoring_attempts_by_this_script"] = 1
        started["scoring_started_at_utc"] = utc_timestamp()
        started_payload = json_bytes(started)
        atomic_bytes(metadata_path, started_payload)
        require(
            metadata_path.read_bytes() == started_payload
            and read_json(metadata_path) == started,
            "Scoring-attempt marker was not published correctly.",
        )
        started_hash = sha256_bytes(started_payload)

        try:
            # Full-file hashing is first permitted here, after the committed
            # scoring marker and frozen prediction verification.
            scoring_sales_hash = sha256_file(sales_path)
            actuals = experiment.load_demand_window(
                sales_path, manifest, calendar,
                FIRST_HOLDOUT_DAY, LAST_HOLDOUT_DAY,
                max_permitted_day=LAST_HOLDOUT_DAY,
            )
            require(
                sha256_file(sales_path) == scoring_sales_hash,
                "Sales source changed while holdout actuals were read.",
            )
            require(
                set(actuals) == set(manifest["id"])
                and all(len(values) == HORIZON for values in actuals.values()),
                "Holdout actual coverage differs.",
            )
            actual_by_key = {
                (series_id, CUTOFF + horizon): value
                for series_id, values in actuals.items()
                for horizon, value in enumerate(values, start=1)
            }
            require(
                len(actual_by_key) == PRODUCT_COUNT * HORIZON,
                "Holdout product-day actual keys are incomplete.",
            )
            scored = frozen.copy()
            scored["actual_sales"] = [
                feature_module.integer_value(
                    actual_by_key[(row.id, CUTOFF + row.horizon)],
                    "Holdout actual",
                )
                for row in scored.itertuples(index=False)
            ]
            scored = scored[list(SCORED_COLUMNS)]
            require(
                not scored.isna().any().any(),
                "Scored artifact contains missing values.",
            )
            pd.testing.assert_frame_equal(
                scored[list(FROZEN_COLUMNS)],
                frozen,
                check_dtype=True,
                check_exact=True,
            )
            products, summary = score_metrics(scored)
            require(
                len(products) == PRODUCT_COUNT
                and products["observations"].eq(HORIZON).all(),
                "Per-product holdout coverage differs.",
            )
            require(
                sha256_file(frozen_path) == frozen_hash,
                "Frozen predictions changed during scoring.",
            )

            completed = copy.deepcopy(started)
            completed["status"] = "EVALUATED"
            completed["holdout_evaluated_at_utc"] = utc_timestamp()
            completed["successful_scoring_publications_by_this_script"] = 1
            completed["selected_sales_fingerprint"]["full_file_sha256"] = (
                scoring_sales_hash
            )
            completed["scored_prediction_sha256"] = sha256_bytes(csv_bytes(scored))
            completed["holdout_metrics_summary"] = summary.to_dict("records")
            completed["evaluation_statement"] = (
                "This script performed one scoring attempt and one successful "
                "publication without fitting, tuning, recalibration, model "
                "selection, or prediction changes during scoring. "
                "It makes no claim about external evaluations."
            )
            completed["integrity_checks"].update({
                "scored_rows": len(scored),
                "per_product_metric_rows": len(products),
                "pooled_observations": PRODUCT_COUNT * HORIZON,
                "aggregate_daily_observations": HORIZON,
                "scored_predictions_exactly_match_frozen_artifact": True,
                "actual_join_uses_exact_product_day_keys": True,
                "model_refitted_during_scoring": False,
                "metric_implementation": "src.evaluate",
            })
            publish_phase(
                root,
                {
                    SCORED_NAME: scored,
                    PRODUCT_NAME: products,
                    SUMMARY_NAME: summary,
                },
                completed,
                started_hash,
            )
            return completed
        except Exception as exc:
            failed = copy.deepcopy(started)
            failed["status"] = "SCORING_FAILED"
            failed["scoring_failed_at_utc"] = utc_timestamp()
            failed["error"] = f"{type(exc).__name__}: {str(exc)[:1000]}"
            failed["evaluation_statement"] = (
                "One scoring attempt started but successful result publication "
                "did not complete. Automatic retry is refused. "
                "No claim is made about external evaluations."
            )
            atomic_bytes(metadata_path, json_bytes(failed))
            raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Frozen forecast generation or separate holdout scoring."
    )
    parser.add_argument("command", choices=("forecast", "score"))
    arguments = parser.parse_args()
    try:
        result = forecast() if arguments.command == "forecast" else score()
    except Exception as exc:
        print(
            f"Holdout {arguments.command} FAILED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    print(f"Holdout state: {result['status']}")
    print(f"Predictions: {result['prediction_count']}")
    print(f"Metadata: {output_directory(ROOT) / METADATA_NAME}")
    print(
        "Holdout actuals were not accessed; scoring is separate."
        if arguments.command == "forecast"
        else "Saved predictions were scored without refitting or tuning."
    )
    print("Exit code: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())