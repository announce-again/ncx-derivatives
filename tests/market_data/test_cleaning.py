from dataclasses import FrozenInstanceError
from datetime import date, datetime, timedelta, timezone

import pytest

from ncx_derivatives.market_data import (
    CarryAssumptions,
    CleaningConfig,
    CleaningDiagnostic,
    CleaningResult,
    EnrichedCleaningResult,
    ExerciseStyle,
    FlatDividendYieldCurve,
    FlatZeroRateCurve,
    OptionChainSnapshot,
    OptionContract,
    OptionQuote,
    OptionType,
    RejectedQuote,
    RejectionReason,
    ValidationSeverity,
    clean_enriched_option_quotes,
    clean_option_chain,
    enrich_option_quote,
)

UTC = timezone.utc
AS_OF = datetime(2026, 1, 2, 15, 0, tzinfo=UTC)
VALUATION_DATE = date(2026, 1, 2)


def _contract(
    option_type: OptionType = OptionType.CALL,
    strike: float = 100.0,
) -> OptionContract:
    return OptionContract(
        underlying_symbol="AAPL",
        expiration=date(2026, 7, 1),
        strike=strike,
        option_type=option_type,
        exercise_style=ExerciseStyle.EUROPEAN,
        contract_multiplier=100.0,
        currency="USD",
    )


def _quote(
    bid: float | None = 4.9,
    ask: float | None = 5.1,
    strike: float = 100.0,
    option_type: OptionType = OptionType.CALL,
    timestamp: datetime = AS_OF,
    volume: int | None = 100,
    open_interest: int | None = 500,
) -> OptionQuote:
    return OptionQuote(
        contract=_contract(option_type=option_type, strike=strike),
        quote_timestamp=timestamp,
        bid=bid,
        ask=ask,
        session_volume=volume,
        open_interest=open_interest,
        open_interest_date=date(2026, 1, 1),
    )


def _snapshot(*quotes: OptionQuote) -> OptionChainSnapshot:
    return OptionChainSnapshot(
        underlying_symbol="AAPL",
        as_of=AS_OF,
        quotes=quotes,
    )


def _carry() -> CarryAssumptions:
    return CarryAssumptions(
        risk_free_curve=FlatZeroRateCurve(0.05),
        dividend_curve=FlatDividendYieldCurve(0.02),
    )


def _enriched(quote: OptionQuote):
    return enrich_option_quote(
        quote=quote,
        valuation_timestamp=AS_OF,
        valuation_date=VALUATION_DATE,
        spot=100.0,
        carry=_carry(),
    )


def test_clean_option_chain_rejects_missing_crossed_and_zero_midpoint() -> None:
    good = _quote(bid=4.9, ask=5.1, strike=100.0)
    missing_bid = _quote(bid=None, ask=5.1, strike=95.0)
    crossed = _quote(bid=5.2, ask=5.1, strike=105.0)
    zero_midpoint = _quote(bid=0.0, ask=0.0, strike=110.0)

    result = clean_option_chain(
        _snapshot(good, missing_bid, crossed, zero_midpoint),
    )

    assert isinstance(result, CleaningResult)
    assert result.accepted == (good,)
    assert result.rejected_count == 3
    assert [item.quote for item in result.rejected] == [
        missing_bid,
        crossed,
        zero_midpoint,
    ]
    assert {diagnostic.reason for diagnostic in result.diagnostics} == {
        RejectionReason.MISSING_BID,
        RejectionReason.CROSSED_MARKET,
        RejectionReason.ZERO_MIDPOINT,
    }


def test_clean_option_chain_keeps_one_sided_and_locked_quotes_when_configured() -> None:
    missing_bid = _quote(bid=None, ask=5.1, strike=95.0)
    locked = _quote(bid=5.0, ask=5.0, strike=100.0)

    default_result = clean_option_chain(_snapshot(missing_bid, locked))
    configured_result = clean_option_chain(
        _snapshot(missing_bid, locked),
        CleaningConfig(reject_missing_bid=False, reject_locked_market=True),
    )

    assert default_result.accepted == (locked,)
    assert default_result.rejected_quotes == (missing_bid,)
    assert configured_result.accepted == (missing_bid,)
    assert configured_result.rejected_quotes == (locked,)
    assert configured_result.rejected[0].diagnostics[0].reason is (
        RejectionReason.LOCKED_MARKET
    )


def test_empty_market_is_rejected_even_if_one_sided_filters_are_disabled() -> None:
    empty = _quote(bid=None, ask=None)

    result = clean_option_chain(
        _snapshot(empty),
        CleaningConfig(reject_missing_bid=False, reject_missing_ask=False),
    )

    assert result.accepted == ()
    assert result.rejected[0].diagnostics[0].reason is RejectionReason.EMPTY_MARKET


def test_clean_option_chain_applies_liquidity_staleness_and_strike_filters() -> None:
    stale = _quote(timestamp=AS_OF - timedelta(minutes=30), strike=90.0)
    low_volume = _quote(volume=9, strike=95.0)
    low_open_interest = _quote(open_interest=49, strike=100.0)
    bad_strike = _quote(strike=200.0)
    good = _quote(strike=105.0)

    result = clean_option_chain(
        _snapshot(stale, low_volume, low_open_interest, bad_strike, good),
        CleaningConfig(
            min_volume=10,
            min_open_interest=50,
            max_quote_age=timedelta(minutes=5),
            min_strike=80.0,
            max_strike=150.0,
        ),
    )

    assert result.accepted == (good,)
    assert {diagnostic.reason for diagnostic in result.diagnostics} == {
        RejectionReason.STALE_QUOTE,
        RejectionReason.INSUFFICIENT_VOLUME,
        RejectionReason.INSUFFICIENT_OPEN_INTEREST,
        RejectionReason.STRIKE_OUT_OF_RANGE,
    }


def test_clean_option_chain_applies_canonical_spread_and_price_filters() -> None:
    wide = _quote(bid=1.0, ask=3.0, strike=95.0)
    too_low_price = _quote(bid=0.05, ask=0.05, strike=100.0)
    good = _quote(bid=4.9, ask=5.1, strike=105.0)

    result = clean_option_chain(
        _snapshot(wide, too_low_price, good),
        CleaningConfig(max_relative_spread=0.5, min_option_price=0.1),
    )

    assert result.accepted == (good,)
    assert {diagnostic.reason for diagnostic in result.diagnostics} == {
        RejectionReason.EXCESSIVE_SPREAD,
        RejectionReason.PRICE_BELOW_MINIMUM,
    }


@pytest.mark.parametrize(
    "config",
    [
        CleaningConfig(min_maturity=0.01),
        CleaningConfig(max_maturity=1.0),
        CleaningConfig(min_spot_moneyness=0.8),
        CleaningConfig(max_spot_moneyness=1.2),
        CleaningConfig(min_forward_moneyness=0.8),
        CleaningConfig(max_forward_moneyness=1.2),
    ],
)
def test_clean_option_chain_rejects_enriched_only_policies(
    config: CleaningConfig,
) -> None:
    with pytest.raises(ValueError, match="require enriched quotes"):
        clean_option_chain(_snapshot(_quote()), config)


def test_clean_enriched_option_quotes_applies_research_filters() -> None:
    wide = _enriched(_quote(bid=1.0, ask=3.0, strike=100.0))
    too_low_price = _enriched(_quote(bid=0.05, ask=0.05, strike=101.0))
    low_moneyness = _enriched(_quote(strike=150.0))
    good = _enriched(_quote(bid=4.9, ask=5.1, strike=100.0))

    result = clean_enriched_option_quotes(
        (wide, too_low_price, low_moneyness, good),
        CleaningConfig(
            max_relative_spread=0.5,
            min_option_price=0.1,
            min_spot_moneyness=0.8,
            max_spot_moneyness=1.2,
        ),
    )

    assert isinstance(result, EnrichedCleaningResult)
    assert result.accepted == (good,)
    assert result.rejected_count == 3
    assert {diagnostic.reason for diagnostic in result.diagnostics} == {
        RejectionReason.EXCESSIVE_SPREAD,
        RejectionReason.PRICE_BELOW_MINIMUM,
        RejectionReason.SPOT_MONEYNESS_OUT_OF_RANGE,
    }
    assert all(item.enriched_quote is not None for item in result.rejected)


def test_clean_enriched_option_quotes_applies_maturity_and_forward_moneyness() -> None:
    short = _enriched(
        OptionQuote(
            contract=OptionContract(
                underlying_symbol="AAPL",
                expiration=date(2026, 1, 3),
                strike=100.0,
                option_type=OptionType.CALL,
                exercise_style=ExerciseStyle.EUROPEAN,
                contract_multiplier=100.0,
                currency="USD",
            ),
            quote_timestamp=AS_OF,
            bid=1.0,
            ask=1.2,
        ),
    )
    forward_otm = _enriched(_quote(strike=200.0))
    good = _enriched(_quote(strike=100.0))

    result = clean_enriched_option_quotes(
        (short, forward_otm, good),
        CleaningConfig(
            min_maturity=0.01,
            min_forward_moneyness=0.8,
            max_forward_moneyness=1.2,
        ),
    )

    assert result.accepted == (good,)
    assert {diagnostic.reason for diagnostic in result.diagnostics} == {
        RejectionReason.MATURITY_OUT_OF_RANGE,
        RejectionReason.FORWARD_MONEYNESS_OUT_OF_RANGE,
    }


def test_cleaning_result_uses_deterministic_ordering() -> None:
    high_strike = _quote(bid=None, ask=5.1, strike=110.0)
    low_strike = _quote(bid=None, ask=5.1, strike=90.0)

    result = CleaningResult(
        accepted=(high_strike, low_strike),
        rejected=(
            RejectedQuote(
                quote=high_strike,
                diagnostics=(
                    CleaningDiagnostic(
                        severity=ValidationSeverity.WARNING,
                        reason=RejectionReason.MISSING_BID,
                        message="missing bid",
                    ),
                ),
            ),
            RejectedQuote(
                quote=low_strike,
                diagnostics=(
                    CleaningDiagnostic(
                        severity=ValidationSeverity.WARNING,
                        reason=RejectionReason.MISSING_BID,
                        message="missing bid",
                    ),
                ),
            ),
        ),
    )

    assert result.accepted == (low_strike, high_strike)
    assert result.rejected_quotes == (low_strike, high_strike)
    assert result.diagnostics == tuple(
        diagnostic
        for rejected_quote in result.rejected
        for diagnostic in rejected_quote.diagnostics
    )


def test_cleaning_result_diagnostics_are_derived_from_rejections() -> None:
    quote = _quote(bid=None, ask=5.1)
    diagnostic = CleaningDiagnostic(
        severity=ValidationSeverity.WARNING,
        reason=RejectionReason.MISSING_BID,
        message="missing bid",
    )

    result = CleaningResult(
        rejected=(RejectedQuote(quote=quote, diagnostics=(diagnostic,)),),
    )

    assert result.diagnostics == (diagnostic,)

    enriched = _enriched(quote)
    enriched_result = EnrichedCleaningResult(
        rejected=(
            RejectedQuote(
                quote=quote,
                diagnostics=(diagnostic,),
                enriched_quote=enriched,
            ),
        ),
    )

    assert enriched_result.diagnostics == (diagnostic,)


def test_cleaning_models_validate_inputs_and_are_immutable() -> None:
    with pytest.raises(ValueError, match="max_relative_spread"):
        CleaningConfig(max_relative_spread=-0.01)

    with pytest.raises(ValueError, match="min_maturity"):
        CleaningConfig(min_maturity=2.0, max_maturity=1.0)

    with pytest.raises(ValueError, match="severity"):
        CleaningDiagnostic(
            severity="warning",  # type: ignore[arg-type]
            reason=RejectionReason.MISSING_BID,
            message="missing bid",
        )

    config = CleaningConfig()
    with pytest.raises(FrozenInstanceError):
        config.reject_missing_bid = False  # type: ignore[misc]


def test_cleaning_public_functions_validate_inputs() -> None:
    with pytest.raises(ValueError, match="snapshot"):
        clean_option_chain("invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="config"):
        clean_option_chain(_snapshot(_quote()), config="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="enriched_quotes"):
        clean_enriched_option_quotes((_quote(),))  # type: ignore[arg-type]
