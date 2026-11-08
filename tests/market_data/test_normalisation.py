from datetime import date, datetime, timezone

import pytest
from _helpers import require_zoneinfo

from ncx_derivatives.market_data import (
    ExerciseStyle,
    OptionType,
    SourceMetadata,
    cboe_option_intervals_csv_config,
    ingest_option_chain_csv,
    normalise_contract_multiplier,
    normalise_date,
    normalise_datetime_utc,
    normalise_exchange,
    normalise_exercise_style,
    normalise_float,
    normalise_int,
    normalise_missing_value,
    normalise_option_type,
    normalise_symbol,
    standard_option_type_value_map,
)


def test_normalises_standard_option_type_representations() -> None:
    assert normalise_option_type("C") is OptionType.CALL
    assert normalise_option_type(" c ") is OptionType.CALL
    assert normalise_option_type("CALL") is OptionType.CALL
    assert normalise_option_type("call") is OptionType.CALL
    assert normalise_option_type("CaLl") is OptionType.CALL
    assert normalise_option_type("P") is OptionType.PUT
    assert normalise_option_type("PUT") is OptionType.PUT
    assert normalise_option_type("put") is OptionType.PUT
    assert normalise_option_type(" pUt ") is OptionType.PUT

    with pytest.raises(ValueError, match="option type"):
        normalise_option_type("X")


def test_normalises_standard_exercise_style_representations() -> None:
    assert normalise_exercise_style("A") is ExerciseStyle.AMERICAN
    assert normalise_exercise_style("american") is ExerciseStyle.AMERICAN
    assert normalise_exercise_style(" AMeRiCaN ") is ExerciseStyle.AMERICAN
    assert normalise_exercise_style("EUROPEAN") is ExerciseStyle.EUROPEAN
    assert normalise_exercise_style("B") is ExerciseStyle.BERMUDAN
    assert normalise_exercise_style("") is None


def test_normalises_missing_values() -> None:
    assert normalise_missing_value("") is None
    assert normalise_missing_value(" NA ") is None
    assert normalise_missing_value("null") is None
    assert normalise_missing_value(" AAPL ") == "AAPL"
    assert normalise_missing_value(1.25) == 1.25


def test_normalises_numeric_strings() -> None:
    assert normalise_float(" 12.50 ") == 12.5
    assert normalise_float(12.5) == 12.5
    assert normalise_float("NA") is None
    assert normalise_int("10") == 10
    assert normalise_int(10) == 10
    assert normalise_int("") is None

    with pytest.raises(ValueError, match="bool"):
        normalise_float(True)
    with pytest.raises(ValueError, match="bool"):
        normalise_int(False)
    with pytest.raises(ValueError, match="integer"):
        normalise_int("10.0")


def test_normalises_dates_and_datetimes_to_utc() -> None:
    assert normalise_date("2026-08-21") == date(2026, 8, 21)
    assert normalise_date(date(2026, 8, 21)) == date(2026, 8, 21)
    with pytest.raises(ValueError, match="datetime"):
        normalise_date(datetime(2026, 8, 21, 9, 30))

    utc_datetime = normalise_datetime_utc("2026-07-30T14:30:00Z")
    assert utc_datetime == datetime(2026, 7, 30, 14, 30, tzinfo=timezone.utc)

    eastern_datetime = normalise_datetime_utc(
        "2026-07-30T09:30:00",
        assume_timezone=require_zoneinfo("America/New_York"),
    )
    assert eastern_datetime == datetime(2026, 7, 30, 13, 30, tzinfo=timezone.utc)


def test_normalises_symbol_exchange_and_multiplier() -> None:
    assert normalise_symbol(" aapl ") == "AAPL"
    assert normalise_exchange(" xnas ") == "XNAS"
    assert normalise_contract_multiplier("100") == 100.0
    assert normalise_contract_multiplier("") is None

    with pytest.raises(ValueError, match="positive"):
        normalise_contract_multiplier("0")


def test_standard_value_maps_are_defensive_copies() -> None:
    mapping = standard_option_type_value_map()
    mapping["C"] = OptionType.PUT

    assert normalise_option_type("C") is OptionType.CALL


def test_cboe_convention_helper_ingests_fixture() -> None:
    config = cboe_option_intervals_csv_config(
        source_metadata=SourceMetadata(provider="fixture"),
    )

    result = ingest_option_chain_csv(
        "tests/fixtures/market_data/cboe_intervals/normal.csv",
        config,
    )

    assert not result.has_errors
    assert len(result.snapshots) == 2
    assert result.successful_row_count == 6
    assert all(
        quote.quote_timestamp.tzinfo is timezone.utc
        for snapshot in result.snapshots
        for quote in snapshot.quotes
    )


def test_cboe_convention_helper_supports_optional_field_variants() -> None:
    config = cboe_option_intervals_csv_config(
        include_open_interest=False,
        include_calculated_underlying=False,
    )

    assert config.mapping.open_interest is None
    assert config.mapping.underlying_price is None
