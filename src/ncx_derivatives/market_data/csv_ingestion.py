from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timezone, tzinfo
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import TextIO

from .models import (
    ExerciseStyle,
    OptionChainSnapshot,
    OptionContract,
    OptionQuote,
    OptionType,
    SourceMetadata,
    UnderlyingQuote,
)
from .normalisation import (
    DEFAULT_MISSING_VALUES,
    normalise_date,
    normalise_datetime_utc,
    normalise_float,
    normalise_int,
)


class BuiltinCsvIngestionCode(str, Enum):
    EMPTY_CSV = "EMPTY_CSV"
    DUPLICATE_COLUMN = "DUPLICATE_COLUMN"
    MISSING_REQUIRED_COLUMN = "MISSING_REQUIRED_COLUMN"
    MISSING_MAPPED_COLUMN = "MISSING_MAPPED_COLUMN"
    ROW_PARSE_ERROR = "ROW_PARSE_ERROR"
    INCONSISTENT_UNDERLYING_QUOTE = "INCONSISTENT_UNDERLYING_QUOTE"


@dataclass(frozen=True, slots=True)
class CsvColumnMapping:
    underlying_symbol: str
    expiration: str
    strike: str
    option_type: str
    quote_timestamp: str
    bid: str
    ask: str
    snapshot_timestamp: str | None = None
    exercise_style: str | None = None
    contract_multiplier: str | None = None
    currency: str | None = None
    source_contract_id: str | None = None
    display_symbol: str | None = None
    listing_exchange: str | None = None
    bid_size: str | None = None
    ask_size: str | None = None
    bid_venue: str | None = None
    ask_venue: str | None = None
    session_volume: str | None = None
    open_interest: str | None = None
    open_interest_date: str | None = None
    underlying_price: str | None = None
    underlying_bid: str | None = None
    underlying_ask: str | None = None
    underlying_timestamp: str | None = None

    @property
    def required_columns(self) -> tuple[str, ...]:
        return (
            self.underlying_symbol,
            self.expiration,
            self.strike,
            self.option_type,
            self.quote_timestamp,
            self.bid,
            self.ask,
        )

    @property
    def optional_columns(self) -> tuple[str, ...]:
        return tuple(
            column
            for column in (
                self.snapshot_timestamp,
                self.exercise_style,
                self.contract_multiplier,
                self.currency,
                self.source_contract_id,
                self.display_symbol,
                self.listing_exchange,
                self.bid_size,
                self.ask_size,
                self.bid_venue,
                self.ask_venue,
                self.session_volume,
                self.open_interest,
                self.open_interest_date,
                self.underlying_price,
                self.underlying_bid,
                self.underlying_ask,
                self.underlying_timestamp,
            )
            if column is not None
        )

    @property
    def all_columns(self) -> tuple[str, ...]:
        return self.required_columns + self.optional_columns


@dataclass(frozen=True, slots=True)
class CsvIngestionConfig:
    mapping: CsvColumnMapping
    source_metadata: SourceMetadata | None = None
    missing_values: frozenset[str] = DEFAULT_MISSING_VALUES
    option_type_values: Mapping[str, OptionType] = field(default_factory=dict)
    exercise_style_values: Mapping[str, ExerciseStyle] = field(default_factory=dict)
    assume_timezone: tzinfo = timezone.utc

    def __post_init__(self) -> None:
        if not isinstance(self.mapping, CsvColumnMapping):
            raise ValueError("mapping must be a CsvColumnMapping")
        if self.source_metadata is not None and not isinstance(
            self.source_metadata,
            SourceMetadata,
        ):
            raise ValueError("source_metadata must be SourceMetadata or None")
        object.__setattr__(
            self,
            "missing_values",
            frozenset(value.upper() for value in self.missing_values),
        )
        if not isinstance(self.assume_timezone, tzinfo):
            raise ValueError("assume_timezone must be a tzinfo")
        probe = datetime(2000, 1, 1, tzinfo=self.assume_timezone)
        if probe.utcoffset() is None:
            raise ValueError("assume_timezone must produce aware datetimes")
        object.__setattr__(
            self,
            "option_type_values",
            MappingProxyType(dict(self.option_type_values)),
        )
        object.__setattr__(
            self,
            "exercise_style_values",
            MappingProxyType(dict(self.exercise_style_values)),
        )


@dataclass(frozen=True, slots=True)
class CsvRawRecord:
    row_number: int
    values: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))


@dataclass(frozen=True, slots=True)
class CsvIngestionError:
    row_number: int
    column: str | None
    code: str
    message: str
    raw_record: CsvRawRecord | None = None


@dataclass(frozen=True, slots=True)
class CsvIngestionResult:
    snapshots: tuple[OptionChainSnapshot, ...]
    raw_records: tuple[CsvRawRecord, ...]
    errors: tuple[CsvIngestionError, ...]

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    @property
    def successful_row_count(self) -> int:
        return sum(len(snapshot.quotes) for snapshot in self.snapshots)

    @property
    def failed_row_count(self) -> int:
        return len(
            {
                error.row_number
                for error in self.errors
                if error.row_number > 0
            },
        )

    @property
    def schema_errors(self) -> tuple[CsvIngestionError, ...]:
        return tuple(error for error in self.errors if error.row_number == 0)


def ingest_option_chain_csv(
    path: str | Path,
    config: CsvIngestionConfig,
) -> CsvIngestionResult:
    with Path(path).open(newline="", encoding="utf-8") as file:
        return ingest_option_chain_csv_file(file, config)


def ingest_option_chain_csv_file(
    file: TextIO,
    config: CsvIngestionConfig,
) -> CsvIngestionResult:
    reader = csv.DictReader(file)
    errors: list[CsvIngestionError] = []

    if reader.fieldnames is None:
        return CsvIngestionResult(
            snapshots=(),
            raw_records=(),
            errors=(
                CsvIngestionError(
                    row_number=0,
                    column=None,
                    code=BuiltinCsvIngestionCode.EMPTY_CSV.value,
                    message="CSV has no header row",
                ),
            ),
        )

    duplicate_columns = [
        column for column, count in Counter(reader.fieldnames).items() if count > 1
    ]
    if duplicate_columns:
        return CsvIngestionResult(
            snapshots=(),
            raw_records=(),
            errors=tuple(
                CsvIngestionError(
                    row_number=0,
                    column=column,
                    code=BuiltinCsvIngestionCode.DUPLICATE_COLUMN.value,
                    message=f"duplicate column: {column}",
                )
                for column in duplicate_columns
            ),
        )

    missing_required = [
        column
        for column in config.mapping.required_columns
        if column not in reader.fieldnames
    ]
    if missing_required:
        return CsvIngestionResult(
            snapshots=(),
            raw_records=(),
            errors=tuple(
                CsvIngestionError(
                    row_number=0,
                    column=column,
                    code=BuiltinCsvIngestionCode.MISSING_REQUIRED_COLUMN.value,
                    message=f"missing required column: {column}",
                )
                for column in missing_required
            ),
        )
    missing_mapped = [
        column
        for column in config.mapping.optional_columns
        if column not in reader.fieldnames
    ]
    if missing_mapped:
        return CsvIngestionResult(
            snapshots=(),
            raw_records=(),
            errors=tuple(
                CsvIngestionError(
                    row_number=0,
                    column=column,
                    code=BuiltinCsvIngestionCode.MISSING_MAPPED_COLUMN.value,
                    message=f"missing mapped column: {column}",
                )
                for column in missing_mapped
            ),
        )

    raw_records: list[CsvRawRecord] = []
    quotes_by_snapshot: dict[tuple[str, datetime], list[OptionQuote]] = {}
    underlying_by_snapshot: dict[tuple[str, datetime], UnderlyingQuote] = {}

    for row_number, row in enumerate(reader, start=2):
        raw_record = CsvRawRecord(row_number=row_number, values=dict(row))
        raw_records.append(raw_record)

        parsed = _parse_quote_row(raw_record, config)
        if isinstance(parsed, CsvIngestionError):
            errors.append(parsed)
            continue

        snapshot_key = (parsed.contract.underlying_symbol, parsed.snapshot_as_of)
        if parsed.underlying_quote is not None:
            existing = underlying_by_snapshot.get(snapshot_key)
            if existing is not None and existing != parsed.underlying_quote:
                errors.append(
                    CsvIngestionError(
                        row_number=row_number,
                        column=None,
                        code=(
                            BuiltinCsvIngestionCode
                            .INCONSISTENT_UNDERLYING_QUOTE
                            .value
                        ),
                        message=(
                            "underlying quote differs within one snapshot group"
                        ),
                        raw_record=raw_record,
                    ),
                )
                continue
            underlying_by_snapshot[snapshot_key] = parsed.underlying_quote

        quotes_by_snapshot.setdefault(snapshot_key, []).append(parsed.quote)

    snapshots = tuple(
        OptionChainSnapshot(
            underlying_symbol=underlying_symbol,
            as_of=as_of,
            quotes=tuple(quotes),
            underlying_quote=underlying_by_snapshot.get(
                (underlying_symbol, as_of),
            ),
            metadata=config.source_metadata,
        )
        for (underlying_symbol, as_of), quotes in sorted(
            quotes_by_snapshot.items(),
            key=lambda item: (item[0][0], item[0][1]),
        )
    )

    return CsvIngestionResult(
        snapshots=snapshots,
        raw_records=tuple(raw_records),
        errors=tuple(errors),
    )


@dataclass(frozen=True, slots=True)
class _ParsedQuoteRow:
    contract: OptionContract
    quote: OptionQuote
    snapshot_as_of: datetime
    underlying_quote: UnderlyingQuote | None


def _parse_quote_row(
    raw_record: CsvRawRecord,
    config: CsvIngestionConfig,
) -> _ParsedQuoteRow | CsvIngestionError:
    mapping = config.mapping

    try:
        underlying_symbol = _required_text(raw_record, mapping.underlying_symbol, config)
        expiration = _date(raw_record, mapping.expiration, config)
        strike = _float(raw_record, mapping.strike, config)
        option_type = _option_type(raw_record, mapping.option_type, config)
        quote_timestamp = _datetime(
            raw_record,
            mapping.quote_timestamp,
            config,
        )
        snapshot_as_of = (
            _datetime(raw_record, mapping.snapshot_timestamp, config)
            if mapping.snapshot_timestamp is not None
            else quote_timestamp
        )

        contract = OptionContract(
            underlying_symbol=underlying_symbol,
            expiration=expiration,
            strike=strike,
            option_type=option_type,
            exercise_style=_optional_exercise_style(
                raw_record,
                mapping.exercise_style,
                config,
            ),
            contract_multiplier=_optional_float(
                raw_record,
                mapping.contract_multiplier,
                config,
            ),
            currency=_optional_text(raw_record, mapping.currency, config),
            source_contract_id=_optional_text(
                raw_record,
                mapping.source_contract_id,
                config,
            ),
            display_symbol=_optional_text(
                raw_record,
                mapping.display_symbol,
                config,
            ),
            listing_exchange=_optional_text(
                raw_record,
                mapping.listing_exchange,
                config,
            ),
            metadata=config.source_metadata,
        )
        quote = OptionQuote(
            contract=contract,
            quote_timestamp=quote_timestamp,
            bid=_optional_float(raw_record, mapping.bid, config),
            ask=_optional_float(raw_record, mapping.ask, config),
            bid_size=_optional_int(raw_record, mapping.bid_size, config),
            ask_size=_optional_int(raw_record, mapping.ask_size, config),
            bid_venue=_optional_text(raw_record, mapping.bid_venue, config),
            ask_venue=_optional_text(raw_record, mapping.ask_venue, config),
            session_volume=_optional_int(
                raw_record,
                mapping.session_volume,
                config,
            ),
            open_interest=_optional_int(
                raw_record,
                mapping.open_interest,
                config,
            ),
            open_interest_date=_optional_date(
                raw_record,
                mapping.open_interest_date,
                config,
            ),
            metadata=config.source_metadata,
        )
        underlying_quote = _underlying_quote(
            raw_record,
            underlying_symbol,
            quote_timestamp,
            config,
        )
    except ValueError as error:
        column = _column_from_error(error)
        return CsvIngestionError(
            row_number=raw_record.row_number,
            column=column,
            code=BuiltinCsvIngestionCode.ROW_PARSE_ERROR.value,
            message=str(error),
            raw_record=raw_record,
        )

    return _ParsedQuoteRow(
        contract=contract,
        quote=quote,
        snapshot_as_of=snapshot_as_of,
        underlying_quote=underlying_quote,
    )


def _underlying_quote(
    raw_record: CsvRawRecord,
    underlying_symbol: str,
    fallback_timestamp: datetime,
    config: CsvIngestionConfig,
) -> UnderlyingQuote | None:
    mapping = config.mapping
    has_underlying_fields = any(
        _optional_text(raw_record, column, config) is not None
        for column in (
            mapping.underlying_price,
            mapping.underlying_bid,
            mapping.underlying_ask,
        )
    )
    if not has_underlying_fields:
        return None

    timestamp = (
        _datetime(raw_record, mapping.underlying_timestamp, config)
        if mapping.underlying_timestamp is not None
        else fallback_timestamp
    )
    return UnderlyingQuote(
        symbol=underlying_symbol,
        quote_timestamp=timestamp,
        price=_optional_float(raw_record, mapping.underlying_price, config),
        bid=_optional_float(raw_record, mapping.underlying_bid, config),
        ask=_optional_float(raw_record, mapping.underlying_ask, config),
        metadata=config.source_metadata,
    )


def _raw_value(
    raw_record: CsvRawRecord,
    column: str | None,
    config: CsvIngestionConfig,
) -> str | None:
    if column is None:
        return None
    value = raw_record.values.get(column)
    if value is None:
        return None
    stripped = value.strip()
    if stripped.upper() in config.missing_values:
        return None
    return stripped


def _required_text(
    raw_record: CsvRawRecord,
    column: str,
    config: CsvIngestionConfig,
) -> str:
    value = _raw_value(raw_record, column, config)
    if value is None:
        raise ValueError(f"{column}: required value is missing")
    return value


def _optional_text(
    raw_record: CsvRawRecord,
    column: str | None,
    config: CsvIngestionConfig,
) -> str | None:
    return _raw_value(raw_record, column, config)


def _float(
    raw_record: CsvRawRecord,
    column: str,
    config: CsvIngestionConfig,
) -> float:
    value = _required_text(raw_record, column, config)
    try:
        parsed = normalise_float(value)
    except ValueError as error:
        raise ValueError(f"{column}: {error}") from error
    if parsed is None:
        raise ValueError(f"{column}: required value is missing")
    return parsed


def _optional_float(
    raw_record: CsvRawRecord,
    column: str | None,
    config: CsvIngestionConfig,
) -> float | None:
    value = _raw_value(raw_record, column, config)
    if value is None:
        return None
    try:
        return normalise_float(value)
    except ValueError as error:
        raise ValueError(f"{column}: {error}") from error


def _optional_int(
    raw_record: CsvRawRecord,
    column: str | None,
    config: CsvIngestionConfig,
) -> int | None:
    value = _raw_value(raw_record, column, config)
    if value is None:
        return None
    try:
        return normalise_int(value)
    except ValueError as error:
        raise ValueError(f"{column}: {error}") from error


def _date(
    raw_record: CsvRawRecord,
    column: str,
    config: CsvIngestionConfig,
) -> date:
    value = _required_text(raw_record, column, config)
    try:
        parsed = normalise_date(value)
    except ValueError as error:
        raise ValueError(f"{column}: {error}") from error
    if parsed is None:
        raise ValueError(f"{column}: required value is missing")
    return parsed


def _optional_date(
    raw_record: CsvRawRecord,
    column: str | None,
    config: CsvIngestionConfig,
) -> date | None:
    value = _raw_value(raw_record, column, config)
    if value is None:
        return None
    try:
        return normalise_date(value)
    except ValueError as error:
        raise ValueError(f"{column}: {error}") from error


def _datetime(
    raw_record: CsvRawRecord,
    column: str | None,
    config: CsvIngestionConfig,
) -> datetime:
    if column is None:
        raise ValueError("datetime column is required")
    value = _required_text(raw_record, column, config)
    try:
        parsed = normalise_datetime_utc(value, config.assume_timezone)
    except ValueError as error:
        raise ValueError(f"{column}: {error}") from error
    if parsed is None:
        raise ValueError(f"{column}: required value is missing")
    return parsed


def _option_type(
    raw_record: CsvRawRecord,
    column: str,
    config: CsvIngestionConfig,
) -> OptionType:
    value = _required_text(raw_record, column, config)
    mapped = config.option_type_values.get(value)
    if mapped is not None:
        return mapped
    try:
        return OptionType(value)
    except ValueError as error:
        raise ValueError(f"{column}: invalid option type {value!r}") from error


def _optional_exercise_style(
    raw_record: CsvRawRecord,
    column: str | None,
    config: CsvIngestionConfig,
) -> ExerciseStyle | None:
    value = _raw_value(raw_record, column, config)
    if value is None:
        return None
    mapped = config.exercise_style_values.get(value)
    if mapped is not None:
        return mapped
    try:
        return ExerciseStyle(value)
    except ValueError as error:
        raise ValueError(f"{column}: invalid exercise style {value!r}") from error


def _column_from_error(error: ValueError) -> str | None:
    message = str(error)
    if ":" not in message:
        return None
    return message.split(":", 1)[0]
