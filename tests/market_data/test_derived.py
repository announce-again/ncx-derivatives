from dataclasses import FrozenInstanceError
from datetime import date, datetime, timedelta, timezone
from math import exp, log
from zoneinfo import ZoneInfo

import pytest

from ncx_derivatives.market_data import (
    ActualFixedDayCount,
    CarryAssumptions,
    DayCountConvention,
    EnrichedOptionQuote,
    ExerciseStyle,
    FlatDividendYieldCurve,
    FlatZeroRateCurve,
    NoArbitrageBounds,
    OptionChainSnapshot,
    OptionContract,
    OptionQuote,
    OptionType,
    UnderlyingQuote,
    absolute_spread,
    enrich_option_chain_snapshot,
    enrich_option_quote,
    european_no_arbitrage_bounds,
    intrinsic_value,
    midpoint,
    no_arbitrage_bounds,
    relative_spread,
    year_fraction,
)

UTC = timezone.utc


def _contract(
    option_type: OptionType = OptionType.CALL,
    strike: float = 100.0,
    expiration: date = date(2027, 1, 1),
    exercise_style: ExerciseStyle | None = ExerciseStyle.EUROPEAN,
) -> OptionContract:
    return OptionContract(
        underlying_symbol="AAPL",
        expiration=expiration,
        strike=strike,
        option_type=option_type,
        exercise_style=exercise_style,
        contract_multiplier=100.0,
        currency="USD",
    )


def _quote(
    option_type: OptionType = OptionType.CALL,
    bid: float | None = 9.5,
    ask: float | None = 10.5,
    strike: float = 100.0,
    expiration: date = date(2027, 1, 1),
    exercise_style: ExerciseStyle | None = ExerciseStyle.EUROPEAN,
) -> OptionQuote:
    return OptionQuote(
        contract=_contract(
            option_type=option_type,
            strike=strike,
            expiration=expiration,
            exercise_style=exercise_style,
        ),
        quote_timestamp=datetime(2026, 1, 1, 15, 0, tzinfo=UTC),
        bid=bid,
        ask=ask,
    )


def _carry() -> CarryAssumptions:
    return CarryAssumptions(
        risk_free_curve=FlatZeroRateCurve(0.05),
        dividend_curve=FlatDividendYieldCurve(0.02),
    )


def test_year_fraction_supports_actual_fixed_conventions() -> None:
    start = date(2026, 1, 1)
    end = date(2027, 1, 1)

    assert year_fraction(start, end) == pytest.approx(1.0)
    assert year_fraction(start, end, DayCountConvention.ACT_360) == pytest.approx(
        365.0 / 360.0,
    )


def test_year_fraction_supports_intraday_datetimes() -> None:
    start = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    end = start + timedelta(hours=12)

    assert year_fraction(start, end) == pytest.approx(0.5 / 365.0)


def test_year_fraction_measures_elapsed_time_across_daylight_saving_changes() -> None:
    new_york = ZoneInfo("America/New_York")
    spring_start = datetime(2026, 3, 7, 12, tzinfo=new_york)
    spring_end = datetime(2026, 3, 8, 12, tzinfo=new_york)
    fall_start = datetime(2026, 11, 1, 1, 45, tzinfo=new_york)
    fall_end = datetime(2026, 11, 1, 1, 15, tzinfo=new_york, fold=1)

    assert year_fraction(spring_start, spring_end) == pytest.approx(23 / (24 * 365))
    assert year_fraction(fall_start, fall_end) == pytest.approx(0.5 / (24 * 365))
    with pytest.raises(ValueError, match="end"):
        year_fraction(fall_end, fall_start)


def test_year_fraction_rejects_naive_datetime_and_negative_period() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        year_fraction(
            datetime(2026, 1, 1, 12, 0),
            datetime(2026, 1, 2, 12, 0, tzinfo=UTC),
        )

    with pytest.raises(ValueError, match="end"):
        year_fraction(date(2026, 1, 2), date(2026, 1, 1))


def test_year_fraction_rejects_mixed_date_and_datetime_inputs() -> None:
    with pytest.raises(ValueError, match="both be dates or both be datetimes"):
        year_fraction(
            datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            date(2026, 1, 2),
        )


def test_actual_fixed_day_count_implements_extensible_interface() -> None:
    day_count = ActualFixedDayCount(DayCountConvention.ACT_360)

    assert day_count.convention is DayCountConvention.ACT_360
    assert day_count.year_fraction(date(2026, 1, 1), date(2026, 1, 31)) == (
        pytest.approx(30.0 / 360.0)
    )


def test_quote_price_fields_handle_missing_zero_and_crossed_quotes() -> None:
    normal = _quote(bid=9.5, ask=10.5)
    missing = _quote(bid=None, ask=10.5)
    zero = _quote(bid=0.0, ask=0.0)
    crossed = _quote(bid=10.5, ask=9.5)

    assert midpoint(normal) == 10.0
    assert absolute_spread(normal) == 1.0
    assert relative_spread(normal) == 0.1
    assert midpoint(missing) is None
    assert absolute_spread(missing) is None
    assert relative_spread(missing) is None
    assert midpoint(zero) == 0.0
    assert relative_spread(zero) is None
    assert absolute_spread(crossed) == -1.0


def test_intrinsic_value_uses_spot_and_option_type() -> None:
    assert intrinsic_value(_quote(OptionType.CALL, strike=100.0), 110.0) == 10.0
    assert intrinsic_value(_quote(OptionType.CALL, strike=100.0), 90.0) == 0.0
    assert intrinsic_value(_quote(OptionType.PUT, strike=100.0), 90.0) == 10.0
    assert intrinsic_value(_quote(OptionType.PUT, strike=100.0), 110.0) == 0.0


def test_european_no_arbitrage_bounds_use_discounted_spot_and_strike() -> None:
    call = _quote(OptionType.CALL, strike=100.0)
    put = _quote(OptionType.PUT, strike=100.0)
    spot = 105.0
    risk_free_discount = exp(-0.05)
    dividend_discount = exp(-0.02)

    call_bounds = european_no_arbitrage_bounds(
        call,
        spot,
        risk_free_discount,
        dividend_discount,
    )
    put_bounds = european_no_arbitrage_bounds(
        put,
        spot,
        risk_free_discount,
        dividend_discount,
    )

    assert call_bounds.lower_bound == pytest.approx(
        max(spot * dividend_discount - 100.0 * risk_free_discount, 0.0),
    )
    assert call_bounds.upper_bound == pytest.approx(spot * dividend_discount)
    assert put_bounds.lower_bound == pytest.approx(
        max(100.0 * risk_free_discount - spot * dividend_discount, 0.0),
    )
    assert put_bounds.upper_bound == pytest.approx(100.0 * risk_free_discount)


@pytest.mark.parametrize(
    "exercise_style",
    [ExerciseStyle.AMERICAN, ExerciseStyle.BERMUDAN, None],
)
def test_no_arbitrage_bounds_only_returns_european_bounds(
    exercise_style: ExerciseStyle | None,
) -> None:
    quote = _quote(OptionType.CALL, exercise_style=exercise_style)

    assert no_arbitrage_bounds(
        quote,
        spot=105.0,
        risk_free_discount_factor=exp(-0.05),
        dividend_discount_factor=exp(-0.02),
    ) is None


def test_enrich_option_quote_computes_research_fields_without_mutating_quote() -> None:
    quote = _quote(OptionType.CALL, bid=9.5, ask=10.5, strike=100.0)
    valuation = datetime(2026, 1, 1, 9, 30, tzinfo=UTC)
    spot = 105.0

    enriched = enrich_option_quote(
        quote=quote,
        valuation_timestamp=valuation,
        valuation_date=date(2026, 1, 1),
        spot=spot,
        carry=_carry(),
    )

    expected_maturity = 365.0 / 365.0
    expected_forward = spot * exp((0.05 - 0.02) * expected_maturity)

    assert isinstance(enriched, EnrichedOptionQuote)
    assert enriched.quote is quote
    assert enriched.valuation_date == date(2026, 1, 1)
    assert enriched.time_to_maturity == pytest.approx(expected_maturity)
    assert enriched.midpoint == 10.0
    assert enriched.absolute_spread == 1.0
    assert enriched.relative_spread == 0.1
    assert enriched.risk_free_discount_factor == pytest.approx(exp(-0.05))
    assert enriched.dividend_discount_factor == pytest.approx(exp(-0.02))
    assert enriched.forward_price == pytest.approx(expected_forward)
    assert enriched.spot_moneyness == pytest.approx(1.05)
    assert enriched.forward_moneyness == pytest.approx(expected_forward / 100.0)
    assert enriched.log_moneyness == pytest.approx(log(100.0 / expected_forward))
    assert enriched.intrinsic_value == 5.0
    assert enriched.time_value == 5.0
    assert quote.bid == 9.5

    with pytest.raises(FrozenInstanceError):
        enriched.midpoint = 11.0  # type: ignore[misc]


def test_enrich_option_quote_preserves_missing_midpoint_dependent_fields() -> None:
    enriched = enrich_option_quote(
        quote=_quote(OptionType.PUT, bid=None, ask=10.5, strike=100.0),
        valuation_timestamp=datetime(2026, 1, 1, 9, 30, tzinfo=UTC),
        valuation_date=date(2026, 1, 1),
        spot=95.0,
        carry=_carry(),
    )

    assert enriched.midpoint is None
    assert enriched.absolute_spread is None
    assert enriched.relative_spread is None
    assert enriched.intrinsic_value == 5.0
    assert enriched.time_value is None


def test_enrich_option_chain_snapshot_uses_underlying_price_and_snapshot_order() -> None:
    as_of = datetime(2026, 1, 1, 9, 30, tzinfo=UTC)
    call = _quote(OptionType.CALL, strike=100.0)
    put = _quote(OptionType.PUT, strike=100.0)
    snapshot = OptionChainSnapshot(
        underlying_symbol="AAPL",
        as_of=as_of,
        quotes=(put, call),
        underlying_quote=UnderlyingQuote(
            symbol="AAPL",
            quote_timestamp=as_of,
            price=101.0,
        ),
    )

    enriched = enrich_option_chain_snapshot(
        snapshot,
        _carry(),
        valuation_date=date(2026, 1, 1),
    )

    assert tuple(item.quote for item in enriched) == snapshot.quotes
    assert {item.spot_price for item in enriched} == {101.0}


def test_enrich_option_chain_snapshot_accepts_explicit_spot_without_underlying_quote() -> None:
    snapshot = OptionChainSnapshot(
        underlying_symbol="AAPL",
        as_of=datetime(2026, 1, 1, 9, 30, tzinfo=UTC),
        quotes=(_quote(),),
        underlying_quote=None,
    )

    enriched = enrich_option_chain_snapshot(
        snapshot,
        _carry(),
        valuation_date=date(2026, 1, 1),
        spot=99.0,
    )

    assert enriched[0].spot_price == 99.0


def test_enrich_option_chain_snapshot_requires_spot_source() -> None:
    snapshot = OptionChainSnapshot(
        underlying_symbol="AAPL",
        as_of=datetime(2026, 1, 1, 9, 30, tzinfo=UTC),
        quotes=(_quote(),),
        underlying_quote=None,
    )

    with pytest.raises(ValueError, match="spot"):
        enrich_option_chain_snapshot(
            snapshot,
            _carry(),
            valuation_date=date(2026, 1, 1),
        )


def test_enrichment_validates_inputs() -> None:
    with pytest.raises(ValueError, match="spot"):
        enrich_option_quote(
            quote=_quote(),
            valuation_timestamp=datetime(2026, 1, 1, 9, 30, tzinfo=UTC),
            valuation_date=date(2026, 1, 1),
            spot=0.0,
            carry=_carry(),
        )

    with pytest.raises(ValueError, match="carry"):
        enrich_option_quote(
            quote=_quote(),
            valuation_timestamp=datetime(2026, 1, 1, 9, 30, tzinfo=UTC),
            valuation_date=date(2026, 1, 1),
            spot=100.0,
            carry="invalid",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="NoArbitrageBounds"):
        EnrichedOptionQuote(
            quote=_quote(),
            valuation_timestamp=datetime(2026, 1, 1, 9, 30, tzinfo=UTC),
            valuation_date=date(2026, 1, 1),
            spot_price=100.0,
            time_to_maturity=1.0,
            midpoint=10.0,
            absolute_spread=1.0,
            relative_spread=0.1,
            risk_free_discount_factor=0.95,
            dividend_discount_factor=0.98,
            forward_price=103.0,
            spot_moneyness=1.0,
            forward_moneyness=1.03,
            log_moneyness=log(100.0 / 103.0),
            intrinsic_value=0.0,
            time_value=10.0,
            no_arbitrage_bounds="invalid",  # type: ignore[arg-type]
        )


def test_no_arbitrage_bounds_validate_order() -> None:
    with pytest.raises(ValueError, match="lower_bound"):
        NoArbitrageBounds(lower_bound=2.0, upper_bound=1.0)


def test_custom_day_count_result_is_validated_at_boundary() -> None:
    class InvalidDayCount:
        convention = DayCountConvention.ACT_365F

        def year_fraction(self, start, end) -> float:
            return float("nan")

    with pytest.raises(ValueError, match="time_to_maturity"):
        enrich_option_quote(
            quote=_quote(),
            valuation_timestamp=datetime(2026, 1, 1, 9, 30, tzinfo=UTC),
            valuation_date=date(2026, 1, 1),
            spot=100.0,
            carry=_carry(),
            day_count=InvalidDayCount(),
        )
