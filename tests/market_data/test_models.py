from dataclasses import FrozenInstanceError
from datetime import date, datetime, timezone

import pytest

from ncx_derivatives.market_data import (
    ContractPairingKey,
    ExerciseStyle,
    OptionChainSnapshot,
    OptionContract,
    OptionQuote,
    OptionTrade,
    OptionType,
    SourceMetadata,
    UnderlyingQuote,
)

UTC = timezone.utc


def _contract(
    option_type: OptionType = OptionType.CALL,
    strike: float = 180.0,
) -> OptionContract:
    return OptionContract(
        underlying_symbol="AAPL",
        expiration=date(2026, 8, 21),
        strike=strike,
        option_type=option_type,
        exercise_style=ExerciseStyle.AMERICAN,
        contract_multiplier=None,
        source_contract_id="1001",
        display_symbol="AAPL  260821C00180000",
        listing_exchange=None,
        metadata=SourceMetadata(provider="fixture"),
    )


def test_option_contract_is_immutable_and_hashable() -> None:
    contract = _contract()

    assert hash(contract)
    with pytest.raises(FrozenInstanceError):
        contract.strike = 181.0  # type: ignore[misc]


def test_option_contract_separates_identity_from_display_symbol() -> None:
    contract = _contract()

    assert contract.source_contract_id == "1001"
    assert contract.display_symbol == "AAPL  260821C00180000"
    assert contract.pairing_key == ContractPairingKey(
        underlying_symbol="AAPL",
        expiration=date(2026, 8, 21),
        strike=180.0,
        exercise_style=ExerciseStyle.AMERICAN,
        contract_multiplier=None,
        currency=None,
    )


def test_contract_equality_ignores_source_metadata_fields() -> None:
    contract_a = OptionContract(
        underlying_symbol="AAPL",
        expiration=date(2026, 9, 18),
        strike=200.0,
        option_type=OptionType.CALL,
        source_contract_id="provider-a-id",
        display_symbol="AAPL_PROVIDER_A",
        listing_exchange="EXCHANGE_A",
        metadata=SourceMetadata(provider="provider-a"),
    )
    contract_b = OptionContract(
        underlying_symbol="AAPL",
        expiration=date(2026, 9, 18),
        strike=200.0,
        option_type=OptionType.CALL,
        source_contract_id="provider-b-id",
        display_symbol="AAPL_PROVIDER_B",
        listing_exchange="EXCHANGE_B",
        metadata=SourceMetadata(provider="provider-b"),
    )

    assert contract_a == contract_b
    assert hash(contract_a) == hash(contract_b)


def test_contract_equality_includes_economic_identity_fields() -> None:
    standard = OptionContract(
        underlying_symbol="AAPL",
        expiration=date(2026, 9, 18),
        strike=200.0,
        option_type=OptionType.CALL,
        exercise_style=ExerciseStyle.AMERICAN,
        contract_multiplier=100.0,
        currency="USD",
    )
    different_multiplier = OptionContract(
        underlying_symbol="AAPL",
        expiration=date(2026, 9, 18),
        strike=200.0,
        option_type=OptionType.CALL,
        exercise_style=ExerciseStyle.AMERICAN,
        contract_multiplier=10.0,
        currency="USD",
    )

    assert standard != different_multiplier


def test_call_put_pairing_key_is_stable_across_option_type() -> None:
    call = _contract(OptionType.CALL)
    put = _contract(OptionType.PUT)

    assert call != put
    assert call.pairing_key == put.pairing_key


def test_pairing_key_distinguishes_multiplier_and_currency() -> None:
    standard_call = OptionContract(
        underlying_symbol="AAPL",
        expiration=date(2026, 9, 18),
        strike=200.0,
        option_type=OptionType.CALL,
        exercise_style=ExerciseStyle.AMERICAN,
        contract_multiplier=100.0,
        currency="USD",
    )
    adjusted_put = OptionContract(
        underlying_symbol="AAPL",
        expiration=date(2026, 9, 18),
        strike=200.0,
        option_type=OptionType.PUT,
        exercise_style=ExerciseStyle.AMERICAN,
        contract_multiplier=10.0,
        currency="USD",
    )
    eur_put = OptionContract(
        underlying_symbol="AAPL",
        expiration=date(2026, 9, 18),
        strike=200.0,
        option_type=OptionType.PUT,
        exercise_style=ExerciseStyle.AMERICAN,
        contract_multiplier=100.0,
        currency="EUR",
    )

    assert standard_call.pairing_key != adjusted_put.pairing_key
    assert standard_call.pairing_key != eur_put.pairing_key


def test_pairing_key_validates_direct_construction() -> None:
    with pytest.raises(ValueError, match="underlying_symbol"):
        ContractPairingKey(
            underlying_symbol="",
            expiration=date(2026, 9, 18),
            strike=200.0,
            exercise_style=None,
            contract_multiplier=None,
            currency=None,
        )

    with pytest.raises(ValueError, match="expiration"):
        ContractPairingKey(
            underlying_symbol="AAPL",
            expiration=datetime(2026, 9, 18, tzinfo=UTC),  # type: ignore[arg-type]
            strike=200.0,
            exercise_style=None,
            contract_multiplier=None,
            currency=None,
        )


def test_contract_rejects_invalid_primitives() -> None:
    with pytest.raises(ValueError, match="underlying_symbol"):
        OptionContract("", date(2026, 8, 21), 180.0, OptionType.CALL)

    with pytest.raises(ValueError, match="strike"):
        OptionContract("AAPL", date(2026, 8, 21), 0.0, OptionType.CALL)

    with pytest.raises(ValueError, match="contract_multiplier"):
        OptionContract(
            "AAPL",
            date(2026, 8, 21),
            180.0,
            OptionType.CALL,
            contract_multiplier=0.0,
        )


def test_quote_allows_independently_missing_and_zero_prices() -> None:
    timestamp = datetime(2026, 7, 30, 14, 30, tzinfo=UTC)

    missing_bid = OptionQuote(
        contract=_contract(),
        quote_timestamp=timestamp,
        bid=None,
        ask=5.05,
    )
    zero_quote = OptionQuote(
        contract=_contract(),
        quote_timestamp=timestamp,
        bid=0.0,
        ask=0.0,
        bid_size=0,
        ask_size=0,
    )

    assert missing_bid.bid is None
    assert missing_bid.ask == 5.05
    assert zero_quote.bid == 0.0
    assert zero_quote.ask == 0.0


def test_quote_timestamp_must_be_timezone_aware() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        OptionQuote(
            contract=_contract(),
            quote_timestamp=datetime(2026, 7, 30, 14, 30),
            bid=1.0,
            ask=1.1,
        )


def test_quote_keeps_open_interest_reference_date_separate() -> None:
    quote = OptionQuote(
        contract=_contract(),
        quote_timestamp=datetime(2026, 7, 30, 14, 30, tzinfo=UTC),
        bid=4.90,
        ask=5.05,
        open_interest=1200,
        open_interest_date=date(2026, 7, 29),
    )

    assert quote.open_interest == 1200
    assert quote.open_interest_date == date(2026, 7, 29)
    assert quote.quote_timestamp.date() == date(2026, 7, 30)


def test_quote_rejects_datetime_open_interest_date() -> None:
    with pytest.raises(ValueError, match="open_interest_date"):
        OptionQuote(
            contract=_contract(),
            quote_timestamp=datetime(2026, 7, 30, 14, 30, tzinfo=UTC),
            bid=4.90,
            ask=5.05,
            open_interest=1200,
            open_interest_date=datetime(2026, 7, 29, tzinfo=UTC),  # type: ignore[arg-type]
        )


def test_quote_rejects_negative_prices() -> None:
    with pytest.raises(ValueError, match="bid"):
        OptionQuote(
            contract=_contract(),
            quote_timestamp=datetime(2026, 7, 30, 14, 30, tzinfo=UTC),
            bid=-0.01,
            ask=5.05,
        )


def test_trade_keeps_trade_time_and_venue_separate_from_quote() -> None:
    trade = OptionTrade(
        contract=_contract(),
        trade_timestamp=datetime(2026, 7, 30, 14, 29, 50, tzinfo=UTC),
        price=4.95,
        size=2,
        trade_venue="316",
    )

    assert trade.trade_timestamp == datetime(
        2026,
        7,
        30,
        14,
        29,
        50,
        tzinfo=UTC,
    )
    assert trade.trade_venue == "316"


def test_trade_rejects_negative_price() -> None:
    with pytest.raises(ValueError, match="price"):
        OptionTrade(
            contract=_contract(),
            trade_timestamp=datetime(2026, 7, 30, 14, 29, 50, tzinfo=UTC),
            price=-0.01,
        )


def test_underlying_quote_can_omit_spot() -> None:
    quote = UnderlyingQuote(
        symbol="AAPL",
        quote_timestamp=datetime(2026, 7, 30, 14, 30, tzinfo=UTC),
        bid=181.20,
        ask=181.24,
    )

    assert quote.price is None
    assert quote.bid == 181.20
    assert quote.ask == 181.24


def test_underlying_quote_rejects_negative_prices() -> None:
    with pytest.raises(ValueError, match="price"):
        UnderlyingQuote(
            symbol="AAPL",
            quote_timestamp=datetime(2026, 7, 30, 14, 30, tzinfo=UTC),
            price=-181.22,
        )


def test_snapshot_sorts_quotes_deterministically() -> None:
    timestamp = datetime(2026, 7, 30, 14, 30, tzinfo=UTC)
    high_strike = OptionQuote(
        contract=_contract(strike=185.0),
        quote_timestamp=timestamp,
        bid=5.25,
        ask=5.40,
    )
    low_strike = OptionQuote(
        contract=_contract(strike=180.0),
        quote_timestamp=timestamp,
        bid=4.90,
        ask=5.05,
    )

    snapshot = OptionChainSnapshot(
        underlying_symbol="AAPL",
        as_of=timestamp,
        quotes=(high_strike, low_strike),
    )

    assert snapshot.quotes == (low_strike, high_strike)
    assert snapshot.contracts == (low_strike.contract, high_strike.contract)


def test_snapshot_allows_missing_underlying_quote() -> None:
    snapshot = OptionChainSnapshot(
        underlying_symbol="AAPL",
        as_of=datetime(2026, 7, 30, 14, 30, tzinfo=UTC),
        quotes=(),
        underlying_quote=None,
    )

    assert snapshot.underlying_quote is None


def test_snapshot_detects_inconsistent_underlying_quote() -> None:
    with pytest.raises(ValueError, match="underlying_quote symbol"):
        OptionChainSnapshot(
            underlying_symbol="AAPL",
            as_of=datetime(2026, 7, 30, 14, 30, tzinfo=UTC),
            underlying_quote=UnderlyingQuote(
                symbol="MSFT",
                quote_timestamp=datetime(2026, 7, 30, 14, 30, tzinfo=UTC),
                price=500.0,
            ),
        )


def test_snapshot_validates_quote_types_before_sorting() -> None:
    with pytest.raises(ValueError, match="quotes must contain OptionQuote"):
        OptionChainSnapshot(
            underlying_symbol="AAPL",
            as_of=datetime(2026, 7, 30, 14, 30, tzinfo=UTC),
            quotes=("invalid",),  # type: ignore[arg-type]
        )


def test_snapshot_materializes_generator_inputs_once() -> None:
    quote = OptionQuote(
        contract=_contract(),
        quote_timestamp=datetime(2026, 7, 30, 14, 30, tzinfo=UTC),
        bid=4.90,
        ask=5.05,
    )

    snapshot = OptionChainSnapshot(
        underlying_symbol="AAPL",
        as_of=datetime(2026, 7, 30, 14, 30, tzinfo=UTC),
        quotes=(item for item in (quote,)),  # type: ignore[arg-type]
    )

    assert snapshot.quotes == (quote,)


def test_snapshot_detects_inconsistent_quote_contract_underlying() -> None:
    msft_contract = OptionContract(
        underlying_symbol="MSFT",
        expiration=date(2026, 8, 21),
        strike=500.0,
        option_type=OptionType.CALL,
    )

    with pytest.raises(ValueError, match="quote contract underlying"):
        OptionChainSnapshot(
            underlying_symbol="AAPL",
            as_of=datetime(2026, 7, 30, 14, 30, tzinfo=UTC),
            quotes=(
                OptionQuote(
                    contract=msft_contract,
                    quote_timestamp=datetime(2026, 7, 30, 14, 30, tzinfo=UTC),
                    bid=1.0,
                    ask=1.1,
                ),
            ),
        )


def test_snapshot_contracts_dedupe_canonical_contract_identity() -> None:
    timestamp = datetime(2026, 7, 30, 14, 30, tzinfo=UTC)
    contract_a = OptionContract(
        underlying_symbol="AAPL",
        expiration=date(2026, 9, 18),
        strike=200.0,
        option_type=OptionType.CALL,
        source_contract_id="provider-a-id",
    )
    contract_b = OptionContract(
        underlying_symbol="AAPL",
        expiration=date(2026, 9, 18),
        strike=200.0,
        option_type=OptionType.CALL,
        source_contract_id="provider-b-id",
    )

    snapshot = OptionChainSnapshot(
        underlying_symbol="AAPL",
        as_of=timestamp,
        quotes=(
            OptionQuote(contract_a, timestamp, bid=1.0, ask=1.1),
            OptionQuote(contract_b, timestamp, bid=1.0, ask=1.1),
        ),
    )

    assert len(snapshot.contracts) == 1


def test_models_reject_invalid_metadata_type() -> None:
    with pytest.raises(ValueError, match="metadata"):
        OptionContract(
            underlying_symbol="AAPL",
            expiration=date(2026, 9, 18),
            strike=200.0,
            option_type=OptionType.CALL,
            metadata="invalid",  # type: ignore[arg-type]
        )
