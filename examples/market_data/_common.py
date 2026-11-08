from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from ncx_derivatives.market_data import (
    CarryAssumptions,
    CleaningConfig,
    CsvColumnMapping,
    CsvIngestionConfig,
    ExerciseStyle,
    FlatDividendYieldCurve,
    FlatZeroRateCurve,
    OptionChainSnapshot,
    OptionContract,
    OptionQuote,
    OptionType,
    SourceMetadata,
    UnderlyingQuote,
    ingest_option_chain_csv,
    standard_exercise_style_value_map,
    standard_option_type_value_map,
)

UTC = timezone.utc
VALUATION_TIMESTAMP = datetime(2026, 7, 30, 14, 30, tzinfo=UTC)
VALUATION_DATE = date(2026, 7, 30)
SPOT = 100.0


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sample_csv_path() -> Path:
    return repo_root() / "examples" / "data" / "cboe_option_chain_sample.csv"


def output_root() -> Path:
    configured = os.environ.get("NCX_EXAMPLES_OUTPUT_DIR")
    if configured:
        return Path(configured)
    return repo_root() / ".tmp" / "examples_output"


def source_metadata() -> SourceMetadata:
    return SourceMetadata(
        provider="example",
        dataset="synthetic-cboe-style-option-chain",
        schema="cboe-style-demo-v1",
    )


def sample_csv_config() -> CsvIngestionConfig:
    return CsvIngestionConfig(
        mapping=CsvColumnMapping(
            underlying_symbol="Underlying Symbol",
            expiration="Expiration",
            strike="Strike",
            option_type="Option Type",
            quote_timestamp="Quote Datetime",
            exercise_style="Exercise Style",
            contract_multiplier="Contract Multiplier",
            currency="Currency",
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
        ),
        source_metadata=source_metadata(),
        option_type_values=standard_option_type_value_map(),
        exercise_style_values=standard_exercise_style_value_map(),
        assume_timezone=UTC,
    )


def ingest_sample_csv():
    return ingest_option_chain_csv(sample_csv_path(), sample_csv_config())


def sample_snapshot() -> OptionChainSnapshot:
    ingestion = ingest_sample_csv()
    if not ingestion.snapshots:
        raise RuntimeError("sample CSV did not produce a snapshot")
    return ingestion.snapshots[0]


def carry_assumptions() -> CarryAssumptions:
    return CarryAssumptions(
        risk_free_curve=FlatZeroRateCurve(0.04),
        dividend_curve=FlatDividendYieldCurve(0.01),
    )


def cleaning_config() -> CleaningConfig:
    return CleaningConfig(
        reject_missing_bid=True,
        reject_missing_ask=True,
        reject_crossed_market=True,
        reject_locked_market=True,
        reject_zero_midpoint=True,
        max_relative_spread=0.80,
        min_open_interest=1,
        max_quote_age=timedelta(minutes=5),
        min_maturity=0.0,
        max_maturity=1.0,
        min_forward_moneyness=0.75,
        max_forward_moneyness=1.30,
        min_option_price=0.01,
    )


def contract(
    option_type: OptionType,
    strike: float,
    expiration: date = date(2026, 8, 21),
) -> OptionContract:
    return OptionContract(
        underlying_symbol="AAPL",
        expiration=expiration,
        strike=strike,
        option_type=option_type,
        exercise_style=ExerciseStyle.EUROPEAN,
        contract_multiplier=100.0,
        currency="USD",
        display_symbol=f"AAPL {expiration.isoformat()} {option_type.value} {strike}",
        metadata=source_metadata(),
    )


def quote(
    option_type: OptionType,
    strike: float,
    bid: float | None,
    ask: float | None,
    expiration: date = date(2026, 8, 21),
    timestamp: datetime = VALUATION_TIMESTAMP,
    open_interest: int | None = 100,
) -> OptionQuote:
    return OptionQuote(
        contract=contract(option_type, strike, expiration),
        quote_timestamp=timestamp,
        bid=bid,
        ask=ask,
        bid_size=10,
        ask_size=10,
        session_volume=50,
        open_interest=open_interest,
        open_interest_date=date(2026, 7, 29),
        metadata=source_metadata(),
    )


def manual_snapshot() -> OptionChainSnapshot:
    return OptionChainSnapshot(
        underlying_symbol="AAPL",
        as_of=VALUATION_TIMESTAMP,
        quotes=(
            quote(OptionType.CALL, 100.0, 4.90, 5.10),
            quote(OptionType.PUT, 100.0, 4.70, 4.90),
            quote(OptionType.CALL, 105.0, 2.35, 2.50),
            quote(OptionType.PUT, 105.0, 6.80, 7.05),
        ),
        underlying_quote=UnderlyingQuote(
            symbol="AAPL",
            quote_timestamp=VALUATION_TIMESTAMP,
            price=SPOT,
            bid=99.98,
            ask=100.02,
        ),
        metadata=source_metadata(),
    )
