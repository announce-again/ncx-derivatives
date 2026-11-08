from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from enum import Enum
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from typing import Any

from ncx_derivatives.utils._files import file_sha256

from .cleaning import CleaningResult, EnrichedCleaningResult
from .models import OptionChainSnapshot
from .pandas_interop import (
    OPTION_CHAIN_COLUMNS,
    _snapshot_quote_records,
    _snapshot_trade_records,
    cleaning_result_to_records,
    static_arbitrage_report_to_records,
    validation_report_to_records,
)
from .static_arbitrage import StaticArbitrageReport
from .validation import ValidationReport

DATASET_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    schema_version: str
    source: str
    as_of: datetime
    ingestion_timestamp: datetime
    valuation_timestamp: datetime | None
    day_count: str | None
    source_information: tuple[tuple[str, Any], ...]
    normalisation_config: tuple[tuple[str, Any], ...]
    cleaning_config: tuple[tuple[str, Any], ...]
    rate_dividend_assumptions: tuple[tuple[str, Any], ...]
    input_quote_count: int
    canonical_quote_count: int
    accepted_quote_count: int
    rejected_quote_count: int
    validation_issue_count: int
    cleaning_diagnostic_count: int
    arbitrage_diagnostic_count: int
    input_hash: str
    output_hash: str
    dataset_id: str

    def __post_init__(self) -> None:
        _validate_non_empty_text(self.schema_version, "schema_version")
        _validate_non_empty_text(self.source, "source")
        _validate_aware_datetime(self.as_of, "as_of")
        _validate_aware_datetime(self.ingestion_timestamp, "ingestion_timestamp")
        if self.valuation_timestamp is not None:
            _validate_aware_datetime(self.valuation_timestamp, "valuation_timestamp")
        _validate_optional_text(self.day_count, "day_count")
        for field_name in (
            "source_information",
            "normalisation_config",
            "cleaning_config",
            "rate_dividend_assumptions",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalise_string_pairs(getattr(self, field_name), field_name),
            )
        for field_name in (
            "input_quote_count",
            "canonical_quote_count",
            "accepted_quote_count",
            "rejected_quote_count",
            "validation_issue_count",
            "cleaning_diagnostic_count",
            "arbitrage_diagnostic_count",
        ):
            _validate_non_negative_int(getattr(self, field_name), field_name)
        _validate_hash(self.input_hash, "input_hash")
        _validate_hash(self.output_hash, "output_hash")
        _validate_hash(self.dataset_id, "dataset_id")
        if self.canonical_quote_count != self.input_quote_count:
            raise ValueError(
                "canonical_quote_count must equal input_quote_count",
            )
        if (
            self.accepted_quote_count + self.rejected_quote_count
            != self.canonical_quote_count
        ):
            raise ValueError(
                "accepted_quote_count plus rejected_quote_count "
                "must equal canonical_quote_count",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "as_of": _canonical_datetime(self.as_of),
            "ingestion_timestamp": _canonical_datetime(self.ingestion_timestamp),
            "valuation_timestamp": (
                None
                if self.valuation_timestamp is None
                else _canonical_datetime(self.valuation_timestamp)
            ),
            "day_count": self.day_count,
            "source_information": dict(self.source_information),
            "normalisation_config": dict(self.normalisation_config),
            "cleaning_config": dict(self.cleaning_config),
            "rate_dividend_assumptions": dict(self.rate_dividend_assumptions),
            "input_quote_count": self.input_quote_count,
            "canonical_quote_count": self.canonical_quote_count,
            "accepted_quote_count": self.accepted_quote_count,
            "rejected_quote_count": self.rejected_quote_count,
            "validation_issue_count": self.validation_issue_count,
            "cleaning_diagnostic_count": self.cleaning_diagnostic_count,
            "arbitrage_diagnostic_count": self.arbitrage_diagnostic_count,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "dataset_id": self.dataset_id,
        }


@dataclass(frozen=True, slots=True)
class DatasetSnapshotWriteResult:
    root: Path
    manifest: DatasetManifest
    paths: tuple[tuple[str, Path], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path):
            raise ValueError("root must be a Path")
        if not isinstance(self.manifest, DatasetManifest):
            raise ValueError("manifest must be a DatasetManifest")
        paths = tuple(self.paths)
        for key, path in paths:
            _validate_non_empty_text(key, "path key")
            if not isinstance(path, Path):
                raise ValueError("paths must contain Path values")
        object.__setattr__(self, "paths", paths)

    @property
    def path_map(self) -> dict[str, Path]:
        return dict(self.paths)


def canonical_snapshot_to_json(snapshot: OptionChainSnapshot) -> dict[str, Any]:
    if not isinstance(snapshot, OptionChainSnapshot):
        raise ValueError("snapshot must be an OptionChainSnapshot")
    payload = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "underlying_symbol": snapshot.underlying_symbol,
        "as_of": snapshot.as_of,
        "records": _snapshot_quote_records(snapshot),
    }
    if snapshot.trades:
        payload["trades"] = _snapshot_trade_records(snapshot)
    if not snapshot.quotes and snapshot.underlying_quote is not None:
        payload["underlying_quote"] = asdict(snapshot.underlying_quote)
    if not snapshot.quotes and snapshot.metadata is not None:
        payload["metadata"] = asdict(snapshot.metadata)
    return _canonical_json_value(payload)


def write_canonical_json(
    snapshot: OptionChainSnapshot,
    path: str | Path,
) -> Path:
    return _write_json(path, canonical_snapshot_to_json(snapshot))


def write_option_chain_csv(
    snapshot: OptionChainSnapshot,
    path: str | Path,
) -> Path:
    if not isinstance(snapshot, OptionChainSnapshot):
        raise ValueError("snapshot must be an OptionChainSnapshot")
    if not isinstance(snapshot, OptionChainSnapshot):
        raise ValueError("snapshot must be an OptionChainSnapshot")
    return _write_records_csv(path, _snapshot_quote_records(snapshot), OPTION_CHAIN_COLUMNS)


def write_market_data_dataset(
    root: str | Path,
    snapshot: OptionChainSnapshot,
    *,
    source: str,
    raw_source_path: str | Path | None = None,
    validation_report: ValidationReport | None = None,
    cleaning_result: CleaningResult | EnrichedCleaningResult | None = None,
    static_arbitrage_report: StaticArbitrageReport | None = None,
    ingestion_timestamp: datetime | None = None,
    valuation_timestamp: datetime | None = None,
    day_count: str | None = None,
    source_information: Mapping[str, Any] | None = None,
    normalisation_config: Mapping[str, Any] | None = None,
    cleaning_config: Mapping[str, Any] | None = None,
    rate_dividend_assumptions: Mapping[str, Any] | None = None,
) -> DatasetSnapshotWriteResult:
    if not isinstance(snapshot, OptionChainSnapshot):
        raise ValueError("snapshot must be an OptionChainSnapshot")
    _validate_non_empty_text(source, "source")
    if validation_report is not None and not isinstance(validation_report, ValidationReport):
        raise ValueError("validation_report must be a ValidationReport or None")
    if cleaning_result is not None and not isinstance(
        cleaning_result,
        (CleaningResult, EnrichedCleaningResult),
    ):
        raise ValueError("cleaning_result must be a cleaning result or None")
    if static_arbitrage_report is not None and not isinstance(
        static_arbitrage_report,
        StaticArbitrageReport,
    ):
        raise ValueError(
            "static_arbitrage_report must be a StaticArbitrageReport or None",
        )
    if ingestion_timestamp is None:
        ingestion_timestamp = datetime.now(timezone.utc)
    _validate_aware_datetime(ingestion_timestamp, "ingestion_timestamp")
    if valuation_timestamp is not None:
        _validate_aware_datetime(valuation_timestamp, "valuation_timestamp")

    root_path = Path(root)
    root_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".ncx-dataset-", dir=root_path.parent) as work:
        staging = Path(work) / "staged"
        raw_dir = staging / "raw"
        processed_dir = staging / "processed"
        rejected_dir = staging / "rejected"
        diagnostics_dir = staging / "diagnostics"
        for directory in (raw_dir, processed_dir, rejected_dir, diagnostics_dir):
            directory.mkdir(parents=True, exist_ok=True)

        paths: dict[str, Path] = {}

        raw_hash_payload: Any
        if raw_source_path is not None:
            source_path = Path(raw_source_path)
            if not source_path.is_file():
                raise ValueError("raw_source_path must be an existing file")
            raw_target = raw_dir / source_path.name
            shutil.copyfile(source_path, raw_target)
            paths["raw_source"] = raw_target
            input_hash = file_sha256(raw_target)
            raw_hash_payload = {"raw_source_hash": input_hash}
        else:
            raw_hash_payload = canonical_snapshot_to_json(snapshot)
            input_hash = _stable_hash(raw_hash_payload)

        accepted_quotes, rejected_quotes = _cleaning_quote_partition(snapshot, cleaning_result)
        accepted_snapshot = _snapshot_with_quotes(snapshot, accepted_quotes)
        rejected_snapshot = _snapshot_with_quotes(snapshot, rejected_quotes)
        canonical_records = _snapshot_quote_records(snapshot)
        accepted_records = _snapshot_quote_records(accepted_snapshot)
        rejected_records = _snapshot_quote_records(rejected_snapshot)

        canonical_csv = _write_records_csv(
            processed_dir / "canonical_option_chain.csv",
            canonical_records,
            OPTION_CHAIN_COLUMNS,
        )
        canonical_json = write_canonical_json(
            snapshot,
            processed_dir / "canonical_option_chain.json",
        )
        accepted_csv = _write_records_csv(
            processed_dir / "accepted_quotes.csv",
            accepted_records,
            OPTION_CHAIN_COLUMNS,
        )
        paths["canonical_csv"] = canonical_csv
        paths["canonical_json"] = canonical_json
        paths["accepted_csv"] = accepted_csv

        validation_records = (
            () if validation_report is None else validation_report_to_records(validation_report)
        )
        cleaning_records = (
            () if cleaning_result is None else cleaning_result_to_records(cleaning_result)
        )
        arbitrage_records = (
            ()
            if static_arbitrage_report is None
            else static_arbitrage_report_to_records(static_arbitrage_report)
        )

        rejected_csv = _write_records_csv(
            rejected_dir / "rejected_quotes.csv",
            rejected_records,
            OPTION_CHAIN_COLUMNS,
        )
        validation_json = _write_json(
            diagnostics_dir / "validation.json",
            {"issues": validation_records},
        )
        cleaning_json = _write_json(
            diagnostics_dir / "cleaning.json",
            {"diagnostics": cleaning_records},
        )
        arbitrage_json = _write_json(
            diagnostics_dir / "arbitrage.json",
            {"diagnostics": arbitrage_records},
        )
        paths["rejected_quotes_csv"] = rejected_csv
        paths["validation_json"] = validation_json
        paths["cleaning_json"] = cleaning_json
        paths["arbitrage_json"] = arbitrage_json

        output_payload = {
            "canonical": canonical_snapshot_to_json(snapshot),
            "accepted": accepted_records,
            "rejected": rejected_records,
            "validation": validation_records,
            "cleaning": cleaning_records,
            "arbitrage": arbitrage_records,
            "raw": raw_hash_payload,
        }
        output_hash = _stable_hash(output_payload)
        identity_payload = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "source": source,
            "as_of": snapshot.as_of,
            "valuation_timestamp": valuation_timestamp,
            "day_count": day_count,
            "source_information": _mapping_to_pairs(source_information),
            "normalisation_config": _mapping_to_pairs(normalisation_config),
            "cleaning_config": _mapping_to_pairs(cleaning_config),
            "rate_dividend_assumptions": _mapping_to_pairs(rate_dividend_assumptions),
            "input_quote_count": len(snapshot.quotes),
            "canonical_quote_count": len(snapshot.quotes),
            "accepted_quote_count": len(accepted_quotes),
            "rejected_quote_count": len(rejected_quotes),
            "validation_issue_count": len(validation_records),
            "cleaning_diagnostic_count": len(cleaning_records),
            "arbitrage_diagnostic_count": len(arbitrage_records),
            "input_hash": input_hash,
            "output_hash": output_hash,
        }
        dataset_id = _stable_hash(identity_payload)
        manifest_payload = {
            **identity_payload,
            "ingestion_timestamp": ingestion_timestamp,
        }
        manifest = DatasetManifest(
            dataset_id=dataset_id,
            **manifest_payload,
        )
        manifest_path = _write_json(staging / "manifest.json", manifest.to_dict())
        paths["manifest"] = manifest_path
        relative_paths = {key: path.relative_to(staging) for key, path in paths.items()}
        _publish_dataset(staging, root_path, tuple(relative_paths.values()))

    return DatasetSnapshotWriteResult(
        root=root_path,
        manifest=manifest,
        paths=tuple(sorted((key, root_path / path) for key, path in relative_paths.items())),
    )


def _publish_dataset(staging: Path, root: Path, paths: tuple[Path, ...]) -> None:
    lock = root.parent / f".{root.name}.ncx-dataset.lock"
    handle = lock.open("x")
    try:
        if not root.exists():
            staging.rename(root)
            return
        for directory in staging.iterdir():
            if directory.is_dir():
                (root / directory.name).mkdir(exist_ok=True)
        _replace_dataset_files(staging, root, paths)
    finally:
        handle.close()
        lock.unlink()


def _replace_dataset_files(staging: Path, root: Path, paths: tuple[Path, ...]) -> None:
    backup = Path(mkdtemp(prefix=".ncx-dataset-backup-", dir=root.parent))
    manifest = Path("manifest.json")
    replacements: list[tuple[Path, Path | None]] = []
    published = False
    try:
        # Withdraw the old manifest before replacing any of its files.
        for relative in (manifest, *(path for path in paths if path != manifest)):
            target = root / relative
            if target.is_dir():
                raise IsADirectoryError(target)
            previous = None
            if target.exists() or target.is_symlink():
                previous = backup / relative
                previous.parent.mkdir(parents=True, exist_ok=True)
                target.replace(previous)
            replacements.append((target, previous))
            if relative != manifest:
                (staging / relative).replace(target)
        (staging / manifest).replace(root / manifest)
        published = True
    except BaseException:
        # Restore data first. A restoration error leaves the manifest withdrawn.
        for target, previous in reversed(replacements):
            if previous is None:
                target.unlink(missing_ok=True)
            else:
                previous.replace(target)
        raise
    finally:
        # If restoration itself fails, retain any remaining originals for recovery.
        if published or not any(path.is_file() or path.is_symlink() for path in backup.rglob("*")):
            shutil.rmtree(backup, ignore_errors=True)


def _write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_stable_json(payload) + "\n", encoding="utf-8")
    return target


def _write_records_csv(
    path: str | Path,
    records: tuple[Mapping[str, Any], ...],
    columns: tuple[str, ...],
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({column: _csv_value(record.get(column)) for column in columns})
    return target


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return _canonical_datetime(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return _stable_json(value)
    if isinstance(value, (tuple, list)):
        return _stable_json(value)
    return value


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _stable_json(payload: Any) -> str:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=_json_default,
        )
    except ValueError as error:
        raise ValueError("payload must be valid canonical JSON") from error


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return _canonical_datetime(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serialisable")


def _mapping_to_pairs(value: Mapping[str, Any] | None) -> tuple[tuple[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise ValueError("configuration values must be mappings or None")
    return tuple(_canonical_mapping(value, "configuration").items())


def _canonical_metadata_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("configuration floats must be finite")
        return value
    if isinstance(value, datetime):
        return _canonical_datetime(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return _canonical_mapping(value, "configuration")
    if isinstance(value, (tuple, list)):
        return [_canonical_metadata_value(item) for item in value]
    raise ValueError(
        f"unsupported configuration value type: {type(value).__name__}",
    )


def _canonical_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("canonical JSON floats must be finite")
        return value
    if isinstance(value, datetime):
        return _canonical_datetime(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(
                    "canonical JSON object keys must be non-empty strings",
                )
            if key in result:
                raise ValueError("canonical JSON object keys must be unique")
            result[key] = _canonical_json_value(item)
        return dict(sorted(result.items()))
    if isinstance(value, (tuple, list)):
        return [_canonical_json_value(item) for item in value]
    raise ValueError(
        f"unsupported canonical JSON value type: {type(value).__name__}",
    )


def _normalise_string_pairs(
    value: tuple[tuple[str, Any], ...],
    field_name: str,
) -> tuple[tuple[str, Any], ...]:
    pairs = tuple(value)
    for item in pairs:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(f"{field_name} must contain key-value pairs")
        key, pair_value = item
        _validate_non_empty_text(key, f"{field_name} key")
        _canonical_json_value(pair_value)
    return pairs


def _canonical_mapping(
    value: Mapping[Any, Any],
    field_name: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{field_name} keys must be non-empty strings")
        if key in result:
            raise ValueError(f"{field_name} keys must be unique")
        result[key] = _canonical_metadata_value(item)
    return dict(sorted(result.items()))


def _snapshot_with_quotes(
    snapshot: OptionChainSnapshot,
    quotes: tuple[Any, ...],
) -> OptionChainSnapshot:
    return OptionChainSnapshot(
        underlying_symbol=snapshot.underlying_symbol,
        as_of=snapshot.as_of,
        quotes=quotes,
        underlying_quote=snapshot.underlying_quote,
        metadata=snapshot.metadata,
    )


def _cleaning_quote_partition(
    snapshot: OptionChainSnapshot,
    cleaning_result: CleaningResult | EnrichedCleaningResult | None,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    if cleaning_result is None:
        return snapshot.quotes, ()
    accepted_quotes = tuple(
        item.quote if hasattr(item, "quote") else item
        for item in cleaning_result.accepted
    )
    rejected_quotes = cleaning_result.rejected_quotes
    combined = accepted_quotes + rejected_quotes
    if len(combined) != len(snapshot.quotes):
        raise ValueError("cleaning_result quote count must match snapshot quotes")
    if set(combined) != set(snapshot.quotes):
        raise ValueError("cleaning_result quotes must match snapshot quotes")
    if len(set(combined)) != len(combined):
        raise ValueError("cleaning_result quotes must not overlap")
    return accepted_quotes, rejected_quotes


def _canonical_datetime(value: datetime) -> str:
    _validate_aware_datetime(value, "datetime")
    utc_value = value.astimezone(timezone.utc)
    return utc_value.isoformat().replace("+00:00", "Z")


def _validate_non_empty_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_optional_text(value: str | None, field_name: str) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError(f"{field_name} must be a non-empty string or None")


def _validate_aware_datetime(value: datetime, field_name: str) -> None:
    if not isinstance(value, datetime):
        raise ValueError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def _validate_non_negative_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _validate_hash(value: str, field_name: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field_name} must be a sha256 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{field_name} must be a sha256 hex digest") from error
