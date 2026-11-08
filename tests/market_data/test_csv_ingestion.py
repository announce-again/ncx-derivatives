from io import StringIO
from pathlib import Path
from types import MappingProxyType

import pytest
from _helpers import require_zoneinfo

from ncx_derivatives.market_data import (
    BuiltinCsvIngestionCode,
    CsvColumnMapping,
    CsvIngestionConfig,
    OptionType,
    SourceMetadata,
    ingest_option_chain_csv,
    ingest_option_chain_csv_file,
)


def _canonical_mapping() -> CsvColumnMapping:
    return CsvColumnMapping(
        underlying_symbol="underlying",
        expiration="expiration",
        strike="strike",
        option_type="option_type",
        quote_timestamp="quote_timestamp",
        snapshot_timestamp="snapshot_timestamp",
        bid="bid",
        ask="ask",
        exercise_style="exercise_style",
        contract_multiplier="multiplier",
        currency="currency",
        source_contract_id="source_contract_id",
        display_symbol="display_symbol",
        bid_size="bid_size",
        ask_size="ask_size",
        session_volume="session_volume",
        open_interest="open_interest",
        open_interest_date="open_interest_date",
        underlying_price="underlying_price",
        underlying_bid="underlying_bid",
        underlying_ask="underlying_ask",
        underlying_timestamp="underlying_timestamp",
    )


def test_duplicate_csv_columns_are_reported_before_parsing_rows():
    mapping = CsvColumnMapping("underlying", "expiration", "strike", "type", "quote_ts", "bid", "ask")
    data = StringIO(
        "underlying,expiration,strike,type,quote_ts,bid,bid,ask\n"
        "AAPL,2026-08-21,100,call,2026-07-30T14:30:00Z,1,4,5\n"
    )
    result = ingest_option_chain_csv_file(data, CsvIngestionConfig(mapping))

    assert not result.snapshots and not result.raw_records
    (error,) = result.errors
    assert error.column == "bid"
    assert error.code == BuiltinCsvIngestionCode.DUPLICATE_COLUMN.value


@pytest.mark.parametrize(
    ("wall_time", "message"),
    [("2026-03-08T02:30:00", "nonexistent"), ("2026-11-01T01:30:00", "ambiguous")],
)
def test_csv_requires_offset_for_dst_transition_times(wall_time, message):
    config = CsvIngestionConfig(
        CsvColumnMapping("underlying", "expiration", "strike", "type", "quote_ts", "bid", "ask"),
        assume_timezone=require_zoneinfo("America/New_York"),
    )
    data = StringIO(
        "underlying,expiration,strike,type,quote_ts,bid,ask\n"
        f"AAPL,2026-12-21,100,call,{wall_time},4,5\n"
        f"AAPL,2026-12-21,100,call,{wall_time}-04:00,4,5\n"
    )
    result = ingest_option_chain_csv_file(data, config)

    (error,) = result.errors
    assert error.column == "quote_ts"
    assert message in error.message
    assert len(result.snapshots) == 1


def test_ingests_canonical_csv_into_snapshots() -> None:
    csv_text = """underlying,expiration,strike,option_type,quote_timestamp,snapshot_timestamp,bid,ask,exercise_style,multiplier,currency,source_contract_id,display_symbol,bid_size,ask_size,session_volume,open_interest,open_interest_date,underlying_price,underlying_bid,underlying_ask,underlying_timestamp
AAPL,2026-08-21,180.0,put,2026-07-30T14:30:00Z,2026-07-30T14:30:00Z,3.70,3.85,american,100,USD,put-id,AAPL  260821P00180000,10,11,200,980,2026-07-29,181.22,181.20,181.24,2026-07-30T14:30:00Z
AAPL,2026-08-21,180.0,call,2026-07-30T14:30:00Z,2026-07-30T14:30:00Z,4.90,5.05,american,100,USD,call-id,AAPL  260821C00180000,12,15,300,1200,2026-07-29,181.22,181.20,181.24,2026-07-30T14:30:00Z
"""

    result = ingest_option_chain_csv_file(
        StringIO(csv_text),
        CsvIngestionConfig(
            mapping=_canonical_mapping(),
            source_metadata=SourceMetadata(provider="unit_test"),
        ),
    )

    assert not result.has_errors
    assert len(result.raw_records) == 2
    assert len(result.snapshots) == 1

    snapshot = result.snapshots[0]
    assert snapshot.underlying_symbol == "AAPL"
    assert snapshot.underlying_quote is not None
    assert snapshot.underlying_quote.price == 181.22
    assert [quote.contract.option_type for quote in snapshot.quotes] == [
        OptionType.CALL,
        OptionType.PUT,
    ]
    assert snapshot.contracts[0].source_contract_id == "call-id"


def test_ingests_cboe_style_fixture_with_explicit_value_mapping() -> None:
    path = Path("tests/fixtures/market_data/cboe_intervals/normal.csv")
    mapping = CsvColumnMapping(
        underlying_symbol="Underlying Symbol",
        expiration="Expiration",
        strike="Strike",
        option_type="Option Type",
        quote_timestamp="Quote Datetime",
        bid="Bid",
        ask="Ask",
        bid_size="Bid Size",
        ask_size="Ask Size",
        open_interest="Open Interest",
        underlying_price="Active Underlying Price",
        underlying_bid="Underlying Bid",
        underlying_ask="Underlying Ask",
    )
    config = CsvIngestionConfig(
        mapping=mapping,
        source_metadata=SourceMetadata(provider="fixture", schema="cboe_style"),
        option_type_values={"C": OptionType.CALL, "P": OptionType.PUT},
    )

    result = ingest_option_chain_csv(path, config)

    assert not result.has_errors
    assert len(result.snapshots) == 2
    assert [len(snapshot.quotes) for snapshot in result.snapshots] == [3, 3]
    assert result.snapshots[0].as_of.isoformat() == "2026-07-30T14:30:00+00:00"
    assert result.snapshots[1].as_of.isoformat() == "2026-07-30T14:31:00+00:00"


def test_missing_required_columns_are_reported_without_rows() -> None:
    csv_text = "underlying,expiration,strike,option_type,quote_timestamp,bid\nAAPL,2026-08-21,180,call,2026-07-30T14:30:00Z,4.90\n"

    result = ingest_option_chain_csv_file(
        StringIO(csv_text),
        CsvIngestionConfig(mapping=_canonical_mapping()),
    )

    assert result.has_errors
    assert result.snapshots == ()
    assert result.raw_records == ()
    assert {error.column for error in result.errors} >= {"ask"}
    assert {error.code for error in result.errors} == {
        BuiltinCsvIngestionCode.MISSING_REQUIRED_COLUMN.value,
    }


def test_missing_optional_mapped_columns_are_schema_errors() -> None:
    csv_text = "underlying,expiration,strike,option_type,quote_timestamp,bid,ask\nAAPL,2026-08-21,180,call,2026-07-30T14:30:00Z,4.90,5.05\n"

    result = ingest_option_chain_csv_file(
        StringIO(csv_text),
        CsvIngestionConfig(
            mapping=CsvColumnMapping(
                underlying_symbol="underlying",
                expiration="expiration",
                strike="strike",
                option_type="option_type",
                quote_timestamp="quote_timestamp",
                bid="bid",
                ask="ask",
                bid_size="bid_size",
            ),
        ),
    )

    assert result.has_errors
    assert result.schema_errors == result.errors
    assert result.errors[0].code == BuiltinCsvIngestionCode.MISSING_MAPPED_COLUMN.value
    assert result.errors[0].column == "bid_size"


def test_row_parse_errors_preserve_raw_record_and_continue() -> None:
    csv_text = """underlying,expiration,strike,option_type,quote_timestamp,snapshot_timestamp,bid,ask,exercise_style,multiplier,currency,source_contract_id,display_symbol,bid_size,ask_size,session_volume,open_interest,open_interest_date,underlying_price,underlying_bid,underlying_ask,underlying_timestamp
AAPL,2026-08-21,180.0,call,2026-07-30T14:30:00Z,2026-07-30T14:30:00Z,4.90,5.05,american,100,USD,call-id,AAPL  260821C00180000,12,15,300,1200,2026-07-29,181.22,181.20,181.24,2026-07-30T14:30:00Z
AAPL,2026-08-21,not-a-number,put,2026-07-30T14:30:00Z,2026-07-30T14:30:00Z,3.70,3.85,american,100,USD,put-id,AAPL  260821P00180000,10,11,200,980,2026-07-29,181.22,181.20,181.24,2026-07-30T14:30:00Z
"""

    result = ingest_option_chain_csv_file(
        StringIO(csv_text),
        CsvIngestionConfig(mapping=_canonical_mapping()),
    )

    assert result.has_errors
    assert len(result.raw_records) == 2
    assert len(result.snapshots) == 1
    assert len(result.snapshots[0].quotes) == 1
    assert result.successful_row_count == 1
    assert result.failed_row_count == 1
    assert result.errors[0].row_number == 3
    assert result.errors[0].column == "strike"
    assert result.errors[0].raw_record is not None
    assert result.errors[0].raw_record.values["strike"] == "not-a-number"


def test_missing_values_are_interpreted_as_none() -> None:
    csv_text = """underlying,expiration,strike,option_type,quote_timestamp,snapshot_timestamp,bid,ask,exercise_style,multiplier,currency,source_contract_id,display_symbol,bid_size,ask_size,session_volume,open_interest,open_interest_date,underlying_price,underlying_bid,underlying_ask,underlying_timestamp
AAPL,2026-08-21,180.0,call,2026-07-30T14:30:00Z,2026-07-30T14:30:00Z,NA,null,american,100,USD,call-id,AAPL  260821C00180000,,,,,NA,,,,2026-07-30T14:30:00Z
"""

    result = ingest_option_chain_csv_file(
        StringIO(csv_text),
        CsvIngestionConfig(mapping=_canonical_mapping()),
    )

    assert not result.has_errors
    quote = result.snapshots[0].quotes[0]
    assert quote.bid is None
    assert quote.ask is None
    assert quote.open_interest is None
    assert result.snapshots[0].underlying_quote is None


def test_inconsistent_underlying_quote_in_group_is_reported() -> None:
    csv_text = """underlying,expiration,strike,option_type,quote_timestamp,snapshot_timestamp,bid,ask,exercise_style,multiplier,currency,source_contract_id,display_symbol,bid_size,ask_size,session_volume,open_interest,open_interest_date,underlying_price,underlying_bid,underlying_ask,underlying_timestamp
AAPL,2026-08-21,180.0,call,2026-07-30T14:30:00Z,2026-07-30T14:30:00Z,4.90,5.05,american,100,USD,call-id,AAPL  260821C00180000,12,15,300,1200,2026-07-29,181.22,181.20,181.24,2026-07-30T14:30:00Z
AAPL,2026-08-21,180.0,put,2026-07-30T14:30:00Z,2026-07-30T14:30:00Z,3.70,3.85,american,100,USD,put-id,AAPL  260821P00180000,10,11,200,980,2026-07-29,182.00,181.20,181.24,2026-07-30T14:30:00Z
"""

    result = ingest_option_chain_csv_file(
        StringIO(csv_text),
        CsvIngestionConfig(mapping=_canonical_mapping()),
    )

    assert result.has_errors
    assert {error.code for error in result.errors} == {
        BuiltinCsvIngestionCode.INCONSISTENT_UNDERLYING_QUOTE.value,
    }
    assert len(result.snapshots) == 1
    assert len(result.snapshots[0].quotes) == 1
    assert result.successful_row_count == 1
    assert result.failed_row_count == 1


def test_config_accepts_zoneinfo_for_naive_datetimes() -> None:
    csv_text = """underlying,expiration,strike,option_type,quote_timestamp,bid,ask
AAPL,2026-08-21,180.0,call,2026-07-30T09:30:00,4.90,5.05
"""

    result = ingest_option_chain_csv_file(
        StringIO(csv_text),
        CsvIngestionConfig(
            mapping=CsvColumnMapping(
                underlying_symbol="underlying",
                expiration="expiration",
                strike="strike",
                option_type="option_type",
                quote_timestamp="quote_timestamp",
                bid="bid",
                ask="ask",
            ),
            assume_timezone=require_zoneinfo("America/New_York"),
        ),
    )

    assert not result.has_errors
    quote_timestamp = result.snapshots[0].quotes[0].quote_timestamp
    assert quote_timestamp.tzinfo is not None
    assert quote_timestamp.utcoffset() is not None


def test_config_rejects_invalid_assume_timezone() -> None:
    with pytest.raises(ValueError, match="assume_timezone"):
        CsvIngestionConfig(
            mapping=_canonical_mapping(),
            assume_timezone=None,  # type: ignore[arg-type]
        )


def test_config_value_mappings_are_exact_and_immutable() -> None:
    option_type_values = {"C": OptionType.CALL}
    config = CsvIngestionConfig(
        mapping=CsvColumnMapping(
            underlying_symbol="underlying",
            expiration="expiration",
            strike="strike",
            option_type="option_type",
            quote_timestamp="quote_timestamp",
            bid="bid",
            ask="ask",
        ),
        option_type_values=option_type_values,
    )

    option_type_values["C"] = OptionType.PUT

    assert isinstance(config.option_type_values, MappingProxyType)
    assert config.option_type_values["C"] is OptionType.CALL
    with pytest.raises(TypeError):
        config.option_type_values["P"] = OptionType.PUT  # type: ignore[index]


def test_raw_record_values_are_immutable() -> None:
    csv_text = """underlying,expiration,strike,option_type,quote_timestamp,bid,ask
AAPL,2026-08-21,180.0,call,2026-07-30T14:30:00Z,4.90,5.05
"""

    result = ingest_option_chain_csv_file(
        StringIO(csv_text),
        CsvIngestionConfig(
            mapping=CsvColumnMapping(
                underlying_symbol="underlying",
                expiration="expiration",
                strike="strike",
                option_type="option_type",
                quote_timestamp="quote_timestamp",
                bid="bid",
                ask="ask",
            ),
        ),
    )

    assert isinstance(result.raw_records[0].values, MappingProxyType)
    with pytest.raises(TypeError):
        result.raw_records[0].values["strike"] = "999"  # type: ignore[index]


def test_without_snapshot_column_quote_timestamp_is_grouping_timestamp() -> None:
    csv_text = """underlying,expiration,strike,option_type,quote_timestamp,bid,ask
AAPL,2026-08-21,180.0,call,2026-07-30T14:30:00Z,4.90,5.05
AAPL,2026-08-21,180.0,put,2026-07-30T14:30:01Z,3.70,3.85
"""

    result = ingest_option_chain_csv_file(
        StringIO(csv_text),
        CsvIngestionConfig(
            mapping=CsvColumnMapping(
                underlying_symbol="underlying",
                expiration="expiration",
                strike="strike",
                option_type="option_type",
                quote_timestamp="quote_timestamp",
                bid="bid",
                ask="ask",
            ),
        ),
    )

    assert not result.has_errors
    assert len(result.snapshots) == 2
