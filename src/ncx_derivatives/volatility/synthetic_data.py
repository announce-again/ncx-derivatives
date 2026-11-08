from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from math import gcd, log, sqrt
from pathlib import Path

from ncx_derivatives.market_data import (
    CarryAssumptions,
    CleaningConfig,
    CsvColumnMapping,
    CsvIngestionConfig,
    FlatDividendYieldCurve,
    FlatZeroRateCurve,
    SourceMetadata,
    year_fraction,
)
from ncx_derivatives.pricing import call_price, put_price
from ncx_derivatives.utils._files import file_sha256

SYNTHETIC_OPTION_QUOTE_COLUMNS = (
    "Underlying Symbol",
    "Expiration",
    "Strike",
    "Option Type",
    "Quote Datetime",
    "Snapshot Datetime",
    "Exercise Style",
    "Contract Multiplier",
    "Currency",
    "Source Contract ID",
    "Bid",
    "Ask",
    "Bid Size",
    "Ask Size",
    "Session Volume",
    "Open Interest",
    "Open Interest Date",
    "Active Underlying Price",
    "Underlying Bid",
    "Underlying Ask",
    "Underlying Datetime",
)

_SYMBOLS = (
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "TSLA",
    "JPM",
    "XOM",
    "SPY",
)
_BASE_SPOTS = (210.0, 420.0, 145.0, 225.0, 690.0, 195.0, 330.0, 295.0, 115.0, 640.0)
_EXPIRY_DAYS = (30, 60, 90, 180, 365)
_BAD_ROW_INTERVAL = 10_000


@dataclass(frozen=True, slots=True)
class SyntheticOptionDatasetConfig:
    row_count: int = 50_000
    seed: int = 20260801
    valuation_timestamp: datetime = datetime(
        2026,
        7,
        30,
        14,
        30,
        tzinfo=timezone.utc,
    )
    valuation_date: date = date(2026, 7, 30)
    risk_free_rate: float = 0.04
    dividend_yield: float = 0.01

    def __post_init__(self) -> None:
        if isinstance(self.row_count, bool) or not isinstance(self.row_count, int):
            raise ValueError("row_count must be an integer")
        if self.row_count <= 0:
            raise ValueError("row_count must be positive")
        if self.valuation_timestamp.tzinfo is None:
            raise ValueError("valuation_timestamp must be timezone-aware")
        if self.valuation_timestamp.date() != self.valuation_date:
            raise ValueError("valuation_timestamp date must match valuation_date")


@dataclass(frozen=True, slots=True)
class SyntheticOptionDatasetSummary:
    path: Path
    row_count: int
    ingestion_bad_row_count: int
    cleaning_bad_row_count: int
    iv_bad_row_count: int
    byte_count: int
    sha256: str


def synthetic_option_quote_csv_config() -> CsvIngestionConfig:
    return CsvIngestionConfig(
        mapping=CsvColumnMapping(
            underlying_symbol="Underlying Symbol",
            expiration="Expiration",
            strike="Strike",
            option_type="Option Type",
            quote_timestamp="Quote Datetime",
            snapshot_timestamp="Snapshot Datetime",
            exercise_style="Exercise Style",
            contract_multiplier="Contract Multiplier",
            currency="Currency",
            source_contract_id="Source Contract ID",
            bid="Bid",
            ask="Ask",
            bid_size="Bid Size",
            ask_size="Ask Size",
            session_volume="Session Volume",
            open_interest="Open Interest",
            open_interest_date="Open Interest Date",
            underlying_price="Active Underlying Price",
            underlying_bid="Underlying Bid",
            underlying_ask="Underlying Ask",
            underlying_timestamp="Underlying Datetime",
        ),
        source_metadata=SourceMetadata(
            provider="ncx-synthetic",
            dataset="deterministic-volatility-pipeline",
            schema="synthetic-option-quotes-v1",
        ),
    )


def synthetic_volatility_pipeline_cleaning_config() -> CleaningConfig:
    return CleaningConfig(
        reject_missing_bid=True,
        reject_missing_ask=True,
        reject_crossed_market=True,
        reject_zero_midpoint=True,
        min_open_interest=1,
        max_quote_age=timedelta(minutes=5),
    )


def synthetic_volatility_pipeline_carry(
    config: SyntheticOptionDatasetConfig,
) -> CarryAssumptions:
    return CarryAssumptions(
        risk_free_curve=FlatZeroRateCurve(config.risk_free_rate),
        dividend_curve=FlatDividendYieldCurve(config.dividend_yield),
    )


def write_synthetic_option_quote_csv(
    path: str | Path,
    config: SyntheticOptionDatasetConfig = SyntheticOptionDatasetConfig(),
) -> SyntheticOptionDatasetSummary:
    """Generate deterministic quote records with failures at three boundaries."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ingestion_bad_rows = 0
    cleaning_bad_rows = 0
    iv_bad_rows = 0
    multiplier = _coprime_multiplier(config.row_count)
    offset = config.seed % config.row_count

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=SYNTHETIC_OPTION_QUOTE_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        for output_index in range(config.row_count):
            source_index = (output_index * multiplier + offset) % config.row_count
            row, bad_stage = _synthetic_row(source_index, config)
            writer.writerow(row)
            if bad_stage == "ingestion":
                ingestion_bad_rows += 1
            elif bad_stage == "cleaning":
                cleaning_bad_rows += 1
            elif bad_stage == "iv":
                iv_bad_rows += 1

    return SyntheticOptionDatasetSummary(
        path=output_path,
        row_count=config.row_count,
        ingestion_bad_row_count=ingestion_bad_rows,
        cleaning_bad_row_count=cleaning_bad_rows,
        iv_bad_row_count=iv_bad_rows,
        byte_count=output_path.stat().st_size,
        sha256=file_sha256(output_path),
    )


def _synthetic_row(
    index: int,
    config: SyntheticOptionDatasetConfig,
) -> tuple[dict[str, object], str | None]:
    option_type = "call" if index % 2 == 0 else "put"
    strike_index = (index // 2) % 50
    expiry_index = (index // 100) % len(_EXPIRY_DAYS)
    snapshot_index = (index // 500) % 10
    underlying_index = index // 5_000
    symbol = (
        _SYMBOLS[underlying_index]
        if underlying_index < len(_SYMBOLS)
        else f"SYN{underlying_index:03d}"
    )
    base_spot = _BASE_SPOTS[underlying_index % len(_BASE_SPOTS)]
    spot = base_spot * (1.0 + 0.0015 * (snapshot_index - 4.5))
    strike = round(base_spot * (0.70 + 0.012 * strike_index), 2)
    expiry_days = _EXPIRY_DAYS[expiry_index]
    expiration = config.valuation_date + timedelta(days=expiry_days)
    maturity = year_fraction(config.valuation_date, expiration)
    snapshot_timestamp = config.valuation_timestamp + timedelta(
        minutes=snapshot_index,
    )
    quote_timestamp = snapshot_timestamp - timedelta(seconds=(index * 7) % 45)
    log_moneyness = log(strike / spot)
    volatility = (
        0.17
        + 0.006 * (underlying_index % 7)
        + 0.10 * log_moneyness * log_moneyness
        + 0.015 * sqrt(maturity)
    )
    price_function = call_price if option_type == "call" else put_price
    fair_value = price_function(
        spot,
        strike,
        maturity,
        config.risk_free_rate,
        volatility,
        config.dividend_yield,
    )
    lower_bound = price_function(
        spot,
        strike,
        maturity,
        config.risk_free_rate,
        0.0,
        config.dividend_yield,
    )
    spread = max(0.01, min(0.50, 0.015 * fair_value + 0.005))
    bid = max(lower_bound + 1e-8, fair_value - spread / 2.0)
    ask = fair_value + spread / 2.0
    bad_stage = None
    bad_case = index % _BAD_ROW_INTERVAL
    strike_value: object = _format_float(strike)

    if bad_case == 0:
        strike_value = "INVALID_STRIKE"
        bad_stage = "ingestion"
    elif bad_case == 1:
        bid = None
        bad_stage = "cleaning"
    elif bad_case == 2:
        bid = fair_value + 0.25
        ask = fair_value
        bad_stage = "cleaning"
    elif bad_case == 3:
        bid = 2.0 * max(spot, strike)
        ask = bid + 0.10
        bad_stage = "iv"

    contract_id = f"{symbol}-{expiration:%Y%m%d}-{strike:.2f}-{option_type[0].upper()}"
    return (
        {
            "Underlying Symbol": symbol,
            "Expiration": expiration.isoformat(),
            "Strike": strike_value,
            "Option Type": option_type,
            "Quote Datetime": quote_timestamp.isoformat(),
            "Snapshot Datetime": snapshot_timestamp.isoformat(),
            "Exercise Style": "european",
            "Contract Multiplier": "100",
            "Currency": "USD",
            "Source Contract ID": contract_id,
            "Bid": "" if bid is None else _format_float(bid),
            "Ask": _format_float(ask),
            "Bid Size": 1 + (index * 11) % 250,
            "Ask Size": 1 + (index * 13) % 250,
            "Session Volume": 10 + (index * 17) % 20_000,
            "Open Interest": 25 + (index * 19) % 50_000,
            "Open Interest Date": (config.valuation_date - timedelta(days=1)).isoformat(),
            "Active Underlying Price": _format_float(spot),
            "Underlying Bid": _format_float(spot - 0.01),
            "Underlying Ask": _format_float(spot + 0.01),
            "Underlying Datetime": snapshot_timestamp.isoformat(),
        },
        bad_stage,
    )


def _coprime_multiplier(row_count: int) -> int:
    multiplier = 104_729
    while gcd(multiplier, row_count) != 1:
        multiplier += 2
    return multiplier


def _format_float(value: float) -> str:
    return f"{value:.10f}"
