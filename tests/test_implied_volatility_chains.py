from dataclasses import replace
from datetime import date, datetime, timezone
from math import exp, inf, isinf, nan

import pytest

from ncx_derivatives.market_data import (
    CarryAssumptions,
    ExerciseStyle,
    FlatDividendYieldCurve,
    FlatZeroRateCurve,
    OptionChainSnapshot,
    OptionContract,
    OptionQuote,
    OptionType,
    UnderlyingQuote,
    enrich_option_chain_snapshot,
)
from ncx_derivatives.pricing import call_price, put_price
from ncx_derivatives.volatility import (
    ImpliedVolatilityDiagnosticFlag,
    ImpliedVolatilityFailureReason,
    ImpliedVolatilityStatus,
    build_implied_volatility_chain,
    implied_volatility_chain_to_dataframe,
    implied_volatility_chain_to_records,
)

UTC = timezone.utc
AS_OF = datetime(2026, 7, 30, 14, 30, tzinfo=UTC)
VALUATION_DATE = date(2026, 7, 30)
EXPIRATION = date(2027, 7, 30)


def _contract(
    option_type: OptionType,
    strike: float,
    expiration: date = EXPIRATION,
) -> OptionContract:
    return OptionContract(
        underlying_symbol="AAPL",
        expiration=expiration,
        strike=strike,
        option_type=option_type,
        exercise_style=ExerciseStyle.EUROPEAN,
        contract_multiplier=100.0,
        currency="USD",
        source_contract_id=f"{option_type.value}-{strike}",
    )


def _quote(
    option_type: OptionType,
    strike: float,
    bid: float | None,
    ask: float | None,
    expiration: date = EXPIRATION,
) -> OptionQuote:
    return OptionQuote(
        contract=_contract(option_type, strike, expiration),
        quote_timestamp=AS_OF,
        bid=bid,
        ask=ask,
        bid_size=10,
        ask_size=12,
    )


def _enriched_quotes(*quotes: OptionQuote, spot: float = 100.0):
    snapshot = OptionChainSnapshot(
        underlying_symbol="AAPL",
        as_of=AS_OF,
        quotes=quotes,
        underlying_quote=UnderlyingQuote(
            symbol="AAPL",
            quote_timestamp=AS_OF,
            price=spot,
        ),
    )
    return enrich_option_chain_snapshot(
        snapshot,
        carry=CarryAssumptions(
            risk_free_curve=FlatZeroRateCurve(0.05),
            dividend_curve=FlatDividendYieldCurve(0.0),
        ),
        valuation_date=VALUATION_DATE,
    )


def test_synthetic_chain_recovers_input_volatility() -> None:
    call = _quote(
        OptionType.CALL,
        100.0,
        call_price(100.0, 100.0, 1.0, 0.05, 0.18),
        call_price(100.0, 100.0, 1.0, 0.05, 0.22),
    )
    put = _quote(
        OptionType.PUT,
        100.0,
        put_price(100.0, 100.0, 1.0, 0.05, 0.18),
        put_price(100.0, 100.0, 1.0, 0.05, 0.22),
    )

    chain = build_implied_volatility_chain(_enriched_quotes(call, put))

    assert len(chain.quotes) == 2
    for quote in chain.quotes:
        assert quote.bid.implied_volatility == pytest.approx(0.18, abs=1e-8)
        assert quote.midpoint.implied_volatility == pytest.approx(0.20, abs=1e-4)
        assert quote.ask.implied_volatility == pytest.approx(0.22, abs=1e-8)
        assert quote.bid.status is ImpliedVolatilityStatus.SUCCESS
        assert quote.midpoint.status is ImpliedVolatilityStatus.SUCCESS
        assert quote.ask.status is ImpliedVolatilityStatus.SUCCESS
        assert quote.midpoint.vega is not None
        assert quote.midpoint.vega > 0.0


def test_multi_expiry_synthetic_chain_recovers_every_observation() -> None:
    expirations = (EXPIRATION, date(2028, 7, 30))
    expected_volatilities = {}
    quotes = []
    for expiration in expirations:
        maturity = (expiration - VALUATION_DATE).days / 365.0
        for option_type in (OptionType.CALL, OptionType.PUT):
            for strike in (80.0, 100.0, 120.0):
                volatility = 0.16 + strike / 2_000.0 + 0.01 * (maturity - 1.0)
                price_function = (
                    call_price if option_type is OptionType.CALL else put_price
                )
                price = price_function(
                    100.0,
                    strike,
                    maturity,
                    0.05,
                    volatility,
                )
                quotes.append(
                    _quote(
                        option_type,
                        strike,
                        price,
                        price,
                        expiration,
                    ),
                )
                expected_volatilities[(expiration, option_type, strike)] = volatility

    chain = build_implied_volatility_chain(_enriched_quotes(*quotes))

    assert len(chain.quotes) == 12
    assert chain.summary.failure_count == 0
    for quote in chain.quotes:
        contract = quote.enriched_quote.quote.contract
        expected = expected_volatilities[
            (contract.expiration, contract.option_type, contract.strike)
        ]
        for result in (quote.bid, quote.midpoint, quote.ask):
            assert result.implied_volatility == pytest.approx(expected, abs=1e-8)
            assert result.vega is not None


def test_bid_mid_ask_implied_volatility_is_ordered() -> None:
    quote = _quote(
        OptionType.CALL,
        100.0,
        call_price(100.0, 100.0, 1.0, 0.05, 0.15),
        call_price(100.0, 100.0, 1.0, 0.05, 0.30),
    )

    iv_quote = build_implied_volatility_chain(_enriched_quotes(quote)).quotes[0]

    assert iv_quote.bid.implied_volatility is not None
    assert iv_quote.midpoint.implied_volatility is not None
    assert iv_quote.ask.implied_volatility is not None
    assert (
        iv_quote.bid.implied_volatility
        <= iv_quote.midpoint.implied_volatility
        <= iv_quote.ask.implied_volatility
    )


def test_single_failed_quote_does_not_fail_chain() -> None:
    valid = _quote(
        OptionType.CALL,
        100.0,
        call_price(100.0, 100.0, 1.0, 0.05, 0.20),
        call_price(100.0, 100.0, 1.0, 0.05, 0.22),
    )
    invalid = _quote(OptionType.CALL, 100.0, 0.01, 150.0)

    chain = build_implied_volatility_chain(_enriched_quotes(valid, invalid))

    assert len(chain.quotes) == 2
    assert chain.summary.failure_count > 0
    assert chain.summary.success_count > 0
    failed = [
        result
        for quote in chain.quotes
        for result in (quote.bid, quote.midpoint, quote.ask)
        if not result.is_success
    ]
    assert failed
    assert all(result.status is ImpliedVolatilityStatus.FAILED for result in failed)
    assert all(result.failure_reason is not None for result in failed)


def test_invalid_price_side_does_not_stop_later_quotes() -> None:
    first = _quote(
        OptionType.CALL,
        90.0,
        call_price(100.0, 90.0, 1.0, 0.05, 0.20),
        call_price(100.0, 90.0, 1.0, 0.05, 0.22),
    )
    invalid = _quote(
        OptionType.CALL,
        100.0,
        call_price(100.0, 100.0, 1.0, 0.05, 0.20),
        call_price(100.0, 100.0, 1.0, 0.05, 0.22),
    )
    third = _quote(
        OptionType.CALL,
        110.0,
        call_price(100.0, 110.0, 1.0, 0.05, 0.20),
        call_price(100.0, 110.0, 1.0, 0.05, 0.22),
    )
    enriched = _enriched_quotes(first, invalid, third)
    object.__setattr__(enriched[1].quote, "bid", "1.25")

    chain = build_implied_volatility_chain(enriched)

    assert chain.quotes[0].midpoint.status is ImpliedVolatilityStatus.SUCCESS
    assert chain.quotes[1].bid.status is ImpliedVolatilityStatus.FAILED
    assert chain.quotes[1].bid.failure_reason is (
        ImpliedVolatilityFailureReason.INVALID_INPUT
    )
    assert chain.quotes[2].midpoint.status is ImpliedVolatilityStatus.SUCCESS
    assert chain.summary.invalid_input_count == 1


def test_overflowing_integer_price_does_not_stop_later_quotes() -> None:
    invalid = _quote(
        OptionType.CALL,
        90.0,
        call_price(100.0, 90.0, 1.0, 0.05, 0.20),
        call_price(100.0, 90.0, 1.0, 0.05, 0.22),
    )
    later = _quote(
        OptionType.CALL,
        110.0,
        call_price(100.0, 110.0, 1.0, 0.05, 0.20),
        call_price(100.0, 110.0, 1.0, 0.05, 0.22),
    )
    enriched = _enriched_quotes(invalid, later)
    object.__setattr__(enriched[0].quote, "bid", 10**10000)

    chain = build_implied_volatility_chain(enriched)

    assert chain.quotes[0].bid.status is ImpliedVolatilityStatus.FAILED
    assert chain.quotes[0].bid.failure_reason is (
        ImpliedVolatilityFailureReason.INVALID_INPUT
    )
    assert chain.quotes[0].bid.price is None
    assert chain.quotes[1].midpoint.status is ImpliedVolatilityStatus.SUCCESS


@pytest.mark.parametrize("bad_price", ["1.25", True, nan, inf, -inf])
def test_bad_price_values_are_invalid_input(bad_price) -> None:
    quote = _quote(
        OptionType.CALL,
        100.0,
        call_price(100.0, 100.0, 1.0, 0.05, 0.20),
        call_price(100.0, 100.0, 1.0, 0.05, 0.22),
    )
    enriched = _enriched_quotes(quote)
    object.__setattr__(enriched[0].quote, "bid", bad_price)

    result = build_implied_volatility_chain(enriched).quotes[0].bid

    assert result.status is ImpliedVolatilityStatus.FAILED
    assert result.implied_volatility is None
    assert result.failure_reason is ImpliedVolatilityFailureReason.INVALID_INPUT


def test_numpy_real_price_is_accepted() -> None:
    np = pytest.importorskip("numpy")
    price = call_price(100.0, 100.0, 1.0, 0.05, 0.20)
    quote = _quote(OptionType.CALL, 100.0, price, price)
    enriched = _enriched_quotes(quote)
    object.__setattr__(enriched[0].quote, "bid", np.float64(price))

    result = build_implied_volatility_chain(enriched).quotes[0].bid

    assert result.status is ImpliedVolatilityStatus.SUCCESS
    assert result.implied_volatility == pytest.approx(0.20, abs=1e-8)


def test_arithmetic_solver_error_is_solver_failed(monkeypatch) -> None:
    import ncx_derivatives.volatility.chains as chains

    quote = _quote(
        OptionType.CALL,
        100.0,
        call_price(100.0, 100.0, 1.0, 0.05, 0.20),
        call_price(100.0, 100.0, 1.0, 0.05, 0.22),
    )

    def fail(*args):
        raise OverflowError("overflow")

    monkeypatch.setattr(chains, "call_implied_volatility", fail)

    chain = build_implied_volatility_chain(_enriched_quotes(quote))

    assert chain.quotes[0].bid.failure_reason is (
        ImpliedVolatilityFailureReason.SOLVER_FAILED
    )
    assert chain.summary.solver_failed_count == 3


def test_unexpected_solver_value_error_is_not_hidden(monkeypatch) -> None:
    import ncx_derivatives.volatility.chains as chains

    quote = _quote(
        OptionType.CALL,
        100.0,
        call_price(100.0, 100.0, 1.0, 0.05, 0.20),
        call_price(100.0, 100.0, 1.0, 0.05, 0.22),
    )

    def fail(*args):
        raise ValueError("implementation bug")

    monkeypatch.setattr(chains, "call_implied_volatility", fail)

    with pytest.raises(ValueError, match="implementation bug"):
        build_implied_volatility_chain(_enriched_quotes(quote))


def test_missing_and_outside_bounds_are_machine_readable() -> None:
    missing = _quote(OptionType.CALL, 100.0, None, 12.0)
    outside_bounds = _quote(OptionType.PUT, 100.0, 100.0, 101.0)

    chain = build_implied_volatility_chain(
        _enriched_quotes(missing, outside_bounds),
    )

    assert chain.quotes[0].bid.failure_reason is (
        ImpliedVolatilityFailureReason.MISSING_PRICE
    )
    assert chain.quotes[0].bid.diagnostic_flags == ()
    assert chain.quotes[1].bid.failure_reason is (
        ImpliedVolatilityFailureReason.OUTSIDE_BOUNDS
    )
    assert chain.quotes[1].bid.diagnostic_flags == ()


def test_lower_and_upper_bounds_have_explicit_results() -> None:
    discounted_strike = 100.0 * exp(-0.05)
    lower_bound = max(100.0 - discounted_strike, 0.0)
    quote = _quote(OptionType.CALL, 100.0, lower_bound, 100.0)

    iv_quote = build_implied_volatility_chain(_enriched_quotes(quote)).quotes[0]

    assert iv_quote.bid.status is ImpliedVolatilityStatus.SUCCESS
    assert iv_quote.bid.implied_volatility == 0.0
    assert iv_quote.bid.vega == 0.0
    assert iv_quote.ask.status is ImpliedVolatilityStatus.SUCCESS
    assert iv_quote.ask.implied_volatility is not None
    assert isinf(iv_quote.ask.implied_volatility)
    assert ImpliedVolatilityDiagnosticFlag.UPPER_BOUND_IV in (
        iv_quote.ask.diagnostic_flags
    )


@pytest.mark.parametrize("exercise_style", [ExerciseStyle.AMERICAN, None])
@pytest.mark.parametrize(
    ("option_type", "upper_bound"),
    [(OptionType.CALL, 100.0), (OptionType.PUT, 100.0 * exp(-0.05))],
)
def test_bsm_upper_bound_is_independent_of_contract_exercise_style(
    exercise_style, option_type, upper_bound
) -> None:
    quote = _quote(option_type, 100.0, upper_bound, upper_bound)
    quote = replace(quote, contract=replace(quote.contract, exercise_style=exercise_style))
    (enriched,) = _enriched_quotes(quote)
    assert enriched.no_arbitrage_bounds is None

    result = build_implied_volatility_chain([enriched]).quotes[0].ask

    assert result.status is ImpliedVolatilityStatus.SUCCESS
    assert result.implied_volatility == inf
    assert result.diagnostic_flags == (ImpliedVolatilityDiagnosticFlag.UPPER_BOUND_IV,)


@pytest.mark.parametrize(
    ("strike", "expiration"),
    [
        (110.0, EXPIRATION),
        (100.0 * exp(0.05), EXPIRATION),
        (100.0, VALUATION_DATE),
    ],
)
def test_zero_iv_uses_explicit_zero_vega_boundary_policy(
    strike: float,
    expiration: date,
) -> None:
    quote = _quote(OptionType.CALL, strike, 0.0, 0.0, expiration)

    result = build_implied_volatility_chain(_enriched_quotes(quote)).quotes[0].bid

    assert result.status is ImpliedVolatilityStatus.SUCCESS
    assert result.implied_volatility == 0.0
    assert result.vega == 0.0


def test_scaled_upper_bound_uses_solver_tolerance_policy() -> None:
    spot = 100_000.0
    strike = 100_000.0
    upper_bound_price = spot - 5e-8
    quote = _quote(
        OptionType.CALL,
        strike,
        call_price(spot, strike, 1.0, 0.05, 0.20),
        upper_bound_price,
    )

    result = build_implied_volatility_chain(
        _enriched_quotes(quote, spot=spot),
    ).quotes[0].ask

    assert result.status is ImpliedVolatilityStatus.SUCCESS
    assert result.implied_volatility is not None
    assert isinf(result.implied_volatility)
    assert result.diagnostic_flags == (
        ImpliedVolatilityDiagnosticFlag.UPPER_BOUND_IV,
    )


def test_vega_failure_is_machine_readable(monkeypatch) -> None:
    import ncx_derivatives.volatility.chains as chains

    quote = _quote(
        OptionType.CALL,
        100.0,
        call_price(100.0, 100.0, 1.0, 0.05, 0.20),
        call_price(100.0, 100.0, 1.0, 0.05, 0.22),
    )

    def fail(*args):
        raise ValueError("vega unavailable")

    monkeypatch.setattr(chains, "black_scholes_vega", fail)

    chain = build_implied_volatility_chain(_enriched_quotes(quote))
    result = chain.quotes[0].midpoint

    assert result.status is ImpliedVolatilityStatus.SUCCESS
    assert result.implied_volatility is not None
    assert result.vega is None
    assert result.diagnostic_flags == (
        ImpliedVolatilityDiagnosticFlag.VEGA_UNAVAILABLE,
    )
    assert chain.summary.vega_unavailable_count == 3


@pytest.mark.parametrize("bad_vega", [nan, inf, -1.0])
def test_invalid_vega_result_is_machine_readable(monkeypatch, bad_vega) -> None:
    import ncx_derivatives.volatility.chains as chains

    quote = _quote(
        OptionType.CALL,
        100.0,
        call_price(100.0, 100.0, 1.0, 0.05, 0.20),
        call_price(100.0, 100.0, 1.0, 0.05, 0.22),
    )
    monkeypatch.setattr(chains, "black_scholes_vega", lambda *args: bad_vega)

    result = build_implied_volatility_chain(_enriched_quotes(quote)).quotes[0].bid

    assert result.status is ImpliedVolatilityStatus.SUCCESS
    assert result.vega is None
    assert result.diagnostic_flags == (
        ImpliedVolatilityDiagnosticFlag.VEGA_UNAVAILABLE,
    )


def test_solver_nan_is_failed_non_finite_result(monkeypatch) -> None:
    import ncx_derivatives.volatility.chains as chains

    quote = _quote(
        OptionType.CALL,
        100.0,
        call_price(100.0, 100.0, 1.0, 0.05, 0.20),
        call_price(100.0, 100.0, 1.0, 0.05, 0.22),
    )
    monkeypatch.setattr(chains, "call_implied_volatility", lambda *args: nan)

    result = build_implied_volatility_chain(_enriched_quotes(quote)).quotes[0].bid

    assert result.status is ImpliedVolatilityStatus.FAILED
    assert result.implied_volatility is None
    assert result.failure_reason is ImpliedVolatilityFailureReason.NON_FINITE_RESULT
    assert result.diagnostic_flags == ()


def test_unexpected_solver_infinity_is_failed_non_finite_result(monkeypatch) -> None:
    import ncx_derivatives.volatility.chains as chains

    quote = _quote(
        OptionType.CALL,
        100.0,
        call_price(100.0, 100.0, 1.0, 0.05, 0.20),
        call_price(100.0, 100.0, 1.0, 0.05, 0.22),
    )
    monkeypatch.setattr(chains, "call_implied_volatility", lambda *args: inf)

    result = build_implied_volatility_chain(_enriched_quotes(quote)).quotes[0].bid

    assert result.status is ImpliedVolatilityStatus.FAILED
    assert result.implied_volatility is None
    assert result.failure_reason is ImpliedVolatilityFailureReason.NON_FINITE_RESULT


def test_negative_solver_infinity_at_upper_bound_is_failed(monkeypatch) -> None:
    import ncx_derivatives.volatility.chains as chains

    quote = _quote(OptionType.CALL, 100.0, 100.0, 100.0)
    monkeypatch.setattr(chains, "call_implied_volatility", lambda *args: -inf)

    result = build_implied_volatility_chain(_enriched_quotes(quote)).quotes[0].bid

    assert result.status is ImpliedVolatilityStatus.FAILED
    assert result.implied_volatility is None
    assert result.failure_reason is ImpliedVolatilityFailureReason.NON_FINITE_RESULT
    assert result.diagnostic_flags == ()


def test_records_preserve_moneyness_vega_status_and_ordering() -> None:
    high_strike = _quote(
        OptionType.CALL,
        110.0,
        call_price(100.0, 110.0, 1.0, 0.05, 0.20),
        call_price(100.0, 110.0, 1.0, 0.05, 0.24),
    )
    low_strike = _quote(
        OptionType.PUT,
        90.0,
        put_price(100.0, 90.0, 1.0, 0.05, 0.20),
        put_price(100.0, 90.0, 1.0, 0.05, 0.24),
    )

    chain = build_implied_volatility_chain(
        _enriched_quotes(high_strike, low_strike),
    )
    records = implied_volatility_chain_to_records(chain)

    assert [record["strike"] for record in records] == [90.0, 110.0]
    assert records[0]["spot_moneyness"] == pytest.approx(100.0 / 90.0)
    assert records[0]["forward_moneyness"] > 0.0
    assert records[0]["log_forward_moneyness"] < 0.0
    assert records[0]["midpoint_vega"] is not None
    assert records[0]["midpoint_status"] == ImpliedVolatilityStatus.SUCCESS.value
    assert "midpoint_failure_reason" in records[0]
    assert "diagnostic_flags" in records[0]


def test_dataframe_export_has_stable_columns() -> None:
    pd = pytest.importorskip("pandas")
    quote = _quote(
        OptionType.CALL,
        100.0,
        call_price(100.0, 100.0, 1.0, 0.05, 0.20),
        call_price(100.0, 100.0, 1.0, 0.05, 0.22),
    )

    frame = implied_volatility_chain_to_dataframe(
        build_implied_volatility_chain(_enriched_quotes(quote)),
    )

    assert isinstance(frame, pd.DataFrame)
    assert list(frame["option_type"]) == ["call"]
    assert "bid_iv" in frame.columns
    assert "midpoint_vega" in frame.columns
    assert "forward_moneyness" in frame.columns

    record = implied_volatility_chain_to_records(
        build_implied_volatility_chain(_enriched_quotes(quote)),
    )[0]
    frame_record = frame.to_dict(orient="records")[0]
    for field in (
        "underlying_symbol",
        "expiration",
        "strike",
        "option_type",
        "midpoint_iv",
        "midpoint_vega",
        "midpoint_status",
        "midpoint_failure_reason",
        "midpoint_diagnostic_flags",
        "log_forward_moneyness",
    ):
        assert frame_record[field] == record[field]


def test_empty_chain_summary_and_dataframe_are_stable() -> None:
    pd = pytest.importorskip("pandas")

    chain = build_implied_volatility_chain(())
    frame = implied_volatility_chain_to_dataframe(chain)

    assert chain.summary.quote_count == 0
    assert chain.summary.result_count == 0
    assert chain.summary.failure_count == 0
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns)[0] == "underlying_symbol"
    assert "log_forward_moneyness" in frame.columns


def test_generator_input_is_consumed_once() -> None:
    quote = _quote(
        OptionType.CALL,
        100.0,
        call_price(100.0, 100.0, 1.0, 0.05, 0.20),
        call_price(100.0, 100.0, 1.0, 0.05, 0.22),
    )
    consumed = 0

    def quote_generator():
        nonlocal consumed
        for item in _enriched_quotes(quote):
            consumed += 1
            yield item

    chain = build_implied_volatility_chain(quote_generator())

    assert consumed == 1
    assert len(chain.quotes) == 1


def test_reversed_input_has_same_ordering() -> None:
    quotes = _enriched_quotes(
        _quote(
            OptionType.CALL,
            110.0,
            call_price(100.0, 110.0, 1.0, 0.05, 0.20),
            call_price(100.0, 110.0, 1.0, 0.05, 0.22),
        ),
        _quote(
            OptionType.PUT,
            90.0,
            put_price(100.0, 90.0, 1.0, 0.05, 0.20),
            put_price(100.0, 90.0, 1.0, 0.05, 0.22),
        ),
    )

    chain_a = build_implied_volatility_chain(quotes)
    chain_b = build_implied_volatility_chain(reversed(quotes))

    assert [quote.sort_key for quote in chain_a.quotes] == [
        quote.sort_key for quote in chain_b.quotes
    ]
