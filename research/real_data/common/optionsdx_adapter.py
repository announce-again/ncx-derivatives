from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from ncx_derivatives.market_data import (
    CsvColumnMapping,
    CsvIngestionConfig,
    SourceMetadata,
)
from research.real_data.common._optional import require_pandas

STANDARD_COLUMNS = (
    "underlying_symbol",
    "quote_date",
    "valuation_timestamp",
    "underlying_price",
    "expiration",
    "vendor_dte",
    "time_to_maturity",
    "strike",
    "option_type",
    "bid",
    "ask",
    "last",
    "volume",
    "open_interest",
    "vendor_iv",
    "vendor_delta",
    "vendor_gamma",
    "vendor_vega",
    "vendor_theta",
    "vendor_rho",
    "source_file",
    "source_row",
)


_COMMON_ALIASES = {
    "quote_date": ("QUOTE_DATE", "quoteDate", "date"),
    "expiration": ("EXPIRE_DATE", "EXPIRATION", "expiration_date", "expiry"),
    "dte": ("DTE", "days_to_expiration", "days_to_expiry"),
    "underlying_price": ("UNDERLYING_LAST", "UNDERLYING_PRICE", "spot", "underlying"),
    "strike": ("STRIKE", "strike_price"),
}

_SIDE_ALIASES = {
    "bid": ("{p}_BID", "{word}_BID"),
    "ask": ("{p}_ASK", "{word}_ASK"),
    "last": ("{p}_LAST", "{word}_LAST"),
    "volume": ("{p}_VOLUME", "{p}_VOL", "{word}_VOLUME"),
    "open_interest": ("{p}_OPEN_INTEREST", "{p}_OI", "{word}_OPEN_INTEREST"),
    "vendor_iv": ("{p}_IV", "{p}_IMPLIED_VOLATILITY", "{word}_IV"),
    "vendor_delta": ("{p}_DELTA", "{word}_DELTA"),
    "vendor_gamma": ("{p}_GAMMA", "{word}_GAMMA"),
    "vendor_vega": ("{p}_VEGA", "{word}_VEGA"),
    "vendor_theta": ("{p}_THETA", "{word}_THETA"),
    "vendor_rho": ("{p}_RHO", "{word}_RHO"),
}


class OptionsDxSchemaError(ValueError):
    """Raised when a wide OptionsDX table cannot be mapped safely."""


def standardize_optionsdx_wide(
    frame,
    *,
    underlying_symbol: str,
    source_file: str = "",
    snapshot_time: time = time(16, 0),
    snapshot_timezone: str = "America/New_York",
):
    """Expand one-row-per-strike call/put data into canonical long records.

    Column matching ignores punctuation, whitespace, brackets, and case.  The
    original frame is never mutated and source row numbers remain attached.
    """

    pandas = require_pandas()
    if not underlying_symbol.strip():
        raise ValueError("underlying_symbol must not be empty")
    mapping = _resolve_schema(tuple(str(column) for column in frame.columns))
    base = pandas.DataFrame(index=frame.index)
    base["underlying_symbol"] = underlying_symbol.strip().upper()
    base["quote_date"] = pandas.to_datetime(
        frame[mapping["quote_date"]], errors="coerce"
    ).dt.normalize()
    base["expiration"] = pandas.to_datetime(
        frame[mapping["expiration"]], errors="coerce"
    ).dt.normalize()
    base["underlying_price"] = pandas.to_numeric(
        frame[mapping["underlying_price"]], errors="coerce"
    )
    base["strike"] = pandas.to_numeric(frame[mapping["strike"]], errors="coerce")
    calendar_dte = (base["expiration"] - base["quote_date"]).dt.days
    if mapping.get("dte") is None:
        base["vendor_dte"] = calendar_dte
    else:
        base["vendor_dte"] = pandas.to_numeric(frame[mapping["dte"]], errors="coerce")
    base["time_to_maturity"] = base["vendor_dte"] / 365.0
    zone = ZoneInfo(snapshot_timezone)
    base["valuation_timestamp"] = base["quote_date"].map(
        lambda value: pandas.NaT
        if pandas.isna(value)
        else datetime.combine(value.date(), snapshot_time, tzinfo=zone)
    )
    base["source_file"] = source_file
    base["source_row"] = (
        pandas.to_numeric(frame["source_row"], errors="coerce").astype("Int64")
        if "source_row" in frame.columns
        else range(2, len(base) + 2)
    )

    sides = []
    for option_type in ("call", "put"):
        side = base.copy()
        side["option_type"] = option_type
        for field in _SIDE_ALIASES:
            column = mapping.get(f"{option_type}_{field}")
            side[field] = (
                pandas.to_numeric(frame[column], errors="coerce")
                if column is not None
                else float("nan")
            )
        for field in ("volume", "open_interest"):
            side[field] = side[field].astype("Int64")
        sides.append(side)
    result = pandas.concat(sides, ignore_index=True)
    return result.loc[:, STANDARD_COLUMNS].sort_values(
        ["underlying_symbol", "quote_date", "expiration", "strike", "option_type"],
        kind="mergesort",
        ignore_index=True,
    )


def read_optionsdx_files(
    paths: Iterable[str | Path],
    *,
    underlying_symbol: str,
):
    """Read CSV/Parquet source files and return one deterministically sorted table."""

    pandas = require_pandas()
    standardized = []
    for raw_path in sorted((Path(path) for path in paths), key=lambda item: item.as_posix()):
        suffix = raw_path.suffix.lower()
        if suffix == ".csv":
            frame = pandas.read_csv(raw_path, low_memory=False)
        elif suffix in {".parquet", ".pq"}:
            try:
                frame = pandas.read_parquet(raw_path)
            except ImportError as error:
                raise RuntimeError(
                    "reading Kaggle Parquet files requires pyarrow or fastparquet"
                ) from error
        else:
            raise ValueError(f"unsupported options file: {raw_path}")
        standardized.append(
            standardize_optionsdx_wide(
                frame,
                underlying_symbol=underlying_symbol,
                source_file=raw_path.name,
            )
        )
    if not standardized:
        return pandas.DataFrame(columns=STANDARD_COLUMNS)
    return pandas.concat(standardized, ignore_index=True).sort_values(
        ["underlying_symbol", "quote_date", "expiration", "strike", "option_type", "source_file", "source_row"],
        kind="mergesort",
        ignore_index=True,
    )


def resolved_optionsdx_schema(columns: Sequence[str]) -> Mapping[str, str | None]:
    return dict(_resolve_schema(columns))


def optionsdx_stage2_csv_config(
    *, dataset_id: str | None = None,
) -> CsvIngestionConfig:
    """Return the Stage 2 mapping for adapter-produced long CSV files."""

    return CsvIngestionConfig(
        mapping=CsvColumnMapping(
            underlying_symbol="underlying_symbol",
            expiration="expiration",
            strike="strike",
            option_type="option_type",
            quote_timestamp="valuation_timestamp",
            snapshot_timestamp="valuation_timestamp",
            bid="bid",
            ask="ask",
            session_volume="volume",
            open_interest="open_interest",
            underlying_price="underlying_price",
            underlying_timestamp="valuation_timestamp",
        ),
        source_metadata=SourceMetadata(
            provider="OptionsDX via Kaggle",
            dataset=dataset_id,
            schema="research001_optionsdx_long_v1",
        ),
    )


def _resolve_schema(columns: Sequence[str]) -> dict[str, str | None]:
    lookup: dict[str, list[str]] = {}
    for column in columns:
        lookup.setdefault(_token(column), []).append(column)

    def find(aliases: Iterable[str], *, required: bool) -> str | None:
        matches = []
        for alias in aliases:
            matches.extend(lookup.get(_token(alias), ()))
        unique = tuple(dict.fromkeys(matches))
        if len(unique) > 1:
            raise OptionsDxSchemaError(f"ambiguous columns for {tuple(aliases)!r}: {unique}")
        if not unique:
            if required:
                raise OptionsDxSchemaError(f"missing required column; tried {tuple(aliases)!r}")
            return None
        return unique[0]

    resolved = {
        field: find(aliases, required=field != "dte")
        for field, aliases in _COMMON_ALIASES.items()
    }
    for option_type, prefix, word in (("call", "C", "CALL"), ("put", "P", "PUT")):
        for field, patterns in _SIDE_ALIASES.items():
            aliases = tuple(pattern.format(p=prefix, word=word) for pattern in patterns)
            resolved[f"{option_type}_{field}"] = find(
                aliases,
                required=field in {"bid", "ask"},
            )
    return resolved


def _token(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())
