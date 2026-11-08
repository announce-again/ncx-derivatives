from math import exp

import pytest

from ncx_derivatives.pricing import call_price, put_price


def test_deep_out_of_money_call_has_positive_tail_value():
    assert call_price(100, 218, 0.25, 0, 0.2) == pytest.approx(5.99050855132617e-15, rel=1e-8, abs=0)


def test_black_scholes_benchmark_prices() -> None:
    """Match a standard Black-Scholes benchmark."""

    spot = 100.0
    strike = 100.0
    maturity = 1.0
    rate = 0.05
    volatility = 0.20

    call = call_price(spot, strike, maturity, rate, volatility)
    put = put_price(spot, strike, maturity, rate, volatility)

    assert call == pytest.approx(10.4506, abs=1e-4)
    assert put == pytest.approx(5.5735, abs=1e-4)


@pytest.mark.parametrize(
    ("spot", "strike", "maturity", "rate", "volatility"),
    [
        (100.0, 100.0, 1.0, 0.05, 0.20),
        (120.0, 100.0, 0.50, 0.03, 0.30),
        (80.0, 100.0, 2.00, 0.01, 0.40),
        (150.0, 130.0, 0.25, -0.01, 0.15),
    ],
)
def test_put_call_parity(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
) -> None:
    """European option prices satisfy put-call parity."""

    call = call_price(spot, strike, maturity, rate, volatility)
    put = put_price(spot, strike, maturity, rate, volatility)

    discounted_strike = strike * exp(-rate * maturity)

    assert call - put == pytest.approx(
        spot - discounted_strike,
        abs=1e-10,
    )


@pytest.mark.parametrize(
    ("spot", "strike", "maturity", "rate", "volatility", "dividend_yield"),
    [
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.02),
        (120.0, 100.0, 0.50, 0.03, 0.30, 0.01),
        (80.0, 100.0, 2.00, 0.01, 0.40, 0.04),
        (150.0, 130.0, 0.25, -0.01, 0.15, 0.03),
    ],
)
def test_put_call_parity_with_continuous_dividend_yield(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float,
) -> None:
    """European option prices satisfy BSM put-call parity."""

    call = call_price(
        spot,
        strike,
        maturity,
        rate,
        volatility,
        dividend_yield,
    )
    put = put_price(
        spot,
        strike,
        maturity,
        rate,
        volatility,
        dividend_yield,
    )

    discounted_spot = spot * exp(-dividend_yield * maturity)
    discounted_strike = strike * exp(-rate * maturity)

    assert call - put == pytest.approx(
        discounted_spot - discounted_strike,
        abs=1e-10,
    )


def test_call_price_increases_with_spot() -> None:
    """A call becomes more valuable when the underlying price rises."""

    low_spot_call = call_price(90.0, 100.0, 1.0, 0.05, 0.20)
    high_spot_call = call_price(110.0, 100.0, 1.0, 0.05, 0.20)

    assert high_spot_call > low_spot_call


def test_put_price_decreases_with_spot() -> None:
    """A put becomes less valuable when the underlying price rises."""

    low_spot_put = put_price(90.0, 100.0, 1.0, 0.05, 0.20)
    high_spot_put = put_price(110.0, 100.0, 1.0, 0.05, 0.20)

    assert high_spot_put < low_spot_put


def test_call_price_decreases_with_strike() -> None:
    """A call becomes less valuable when its strike rises."""

    low_strike_call = call_price(100.0, 90.0, 1.0, 0.05, 0.20)
    high_strike_call = call_price(100.0, 110.0, 1.0, 0.05, 0.20)

    assert high_strike_call < low_strike_call


def test_put_price_increases_with_strike() -> None:
    """A put becomes more valuable when its strike rises."""

    low_strike_put = put_price(100.0, 90.0, 1.0, 0.05, 0.20)
    high_strike_put = put_price(100.0, 110.0, 1.0, 0.05, 0.20)

    assert high_strike_put > low_strike_put


def test_option_prices_increase_with_volatility() -> None:
    """Both calls and puts become more valuable as volatility rises."""

    low_volatility = 0.10
    high_volatility = 0.40

    low_vol_call = call_price(
        100.0, 100.0, 1.0, 0.05, low_volatility
    )
    high_vol_call = call_price(
        100.0, 100.0, 1.0, 0.05, high_volatility
    )

    low_vol_put = put_price(
        100.0, 100.0, 1.0, 0.05, low_volatility
    )
    high_vol_put = put_price(
        100.0, 100.0, 1.0, 0.05, high_volatility
    )

    assert high_vol_call > low_vol_call
    assert high_vol_put > low_vol_put


def test_dividend_yield_keyword_affects_option_prices() -> None:
    """Continuous dividends lower calls and raise puts."""

    no_dividend_call = call_price(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
    )
    dividend_call = call_price(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        dividend_yield=0.02,
    )
    no_dividend_put = put_price(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
    )
    dividend_put = put_price(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        dividend_yield=0.02,
    )

    assert dividend_call < no_dividend_call
    assert dividend_put > no_dividend_put


@pytest.mark.parametrize(
    ("spot", "strike", "maturity", "rate", "volatility"),
    [
        (100.0, 100.0, 1.0, 0.05, 0.20),
        (140.0, 100.0, 0.50, 0.03, 0.25),
        (70.0, 100.0, 2.00, 0.01, 0.35),
        (100.0, 120.0, 0.25, -0.02, 0.15),
    ],
)
def test_option_prices_respect_no_arbitrage_bounds(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
) -> None:
    """European option prices remain within static-arbitrage bounds."""

    call = call_price(spot, strike, maturity, rate, volatility)
    put = put_price(spot, strike, maturity, rate, volatility)

    discounted_strike = strike * exp(-rate * maturity)

    call_lower_bound = max(0.0, spot - discounted_strike)
    put_lower_bound = max(0.0, discounted_strike - spot)

    assert call_lower_bound <= call <= spot
    assert put_lower_bound <= put <= discounted_strike

def test_call_and_put_return_payoff_at_maturity() -> None:
    """At expiry, option prices equal intrinsic value."""

    assert call_price(100.0, 90.0, 0.0, 0.05, 0.20) == 10.0
    assert call_price(100.0, 110.0, 0.0, 0.05, 0.20) == 0.0

    assert put_price(100.0, 110.0, 0.0, 0.05, 0.20) == 10.0
    assert put_price(100.0, 90.0, 0.0, 0.05, 0.20) == 0.0


def test_zero_volatility_returns_discounted_deterministic_payoff() -> None:
    """With zero volatility, pricing reduces to discounted intrinsic value."""

    spot = 100.0
    strike = 105.0
    maturity = 1.5
    rate = 0.04
    volatility = 0.0

    discounted_strike = strike * exp(-rate * maturity)

    assert call_price(
        spot,
        strike,
        maturity,
        rate,
        volatility,
    ) == pytest.approx(max(spot - discounted_strike, 0.0))
    assert put_price(
        spot,
        strike,
        maturity,
        rate,
        volatility,
    ) == pytest.approx(max(discounted_strike - spot, 0.0))


def test_zero_volatility_with_dividend_yield_returns_forward_payoff() -> None:
    """With dividends, zero-vol pricing uses the prepaid forward."""

    spot = 100.0
    strike = 105.0
    maturity = 1.5
    rate = 0.04
    volatility = 0.0
    dividend_yield = 0.02

    discounted_spot = spot * exp(-dividend_yield * maturity)
    discounted_strike = strike * exp(-rate * maturity)

    assert call_price(
        spot,
        strike,
        maturity,
        rate,
        volatility,
        dividend_yield,
    ) == pytest.approx(max(discounted_spot - discounted_strike, 0.0))
    assert put_price(
        spot,
        strike,
        maturity,
        rate,
        volatility,
        dividend_yield,
    ) == pytest.approx(max(discounted_strike - discounted_spot, 0.0))


@pytest.mark.parametrize(
    ("spot", "strike", "maturity", "rate", "volatility"),
    [
        (0.0, 100.0, 1.0, 0.05, 0.20),
        (-100.0, 100.0, 1.0, 0.05, 0.20),
        (100.0, 0.0, 1.0, 0.05, 0.20),
        (100.0, -100.0, 1.0, 0.05, 0.20),
        (100.0, 100.0, -1.0, 0.05, 0.20),
        (100.0, 100.0, 1.0, 0.05, -0.20),
    ],
)
def test_option_prices_reject_invalid_inputs(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
) -> None:
    """Invalid market or contract inputs raise ValueError."""

    with pytest.raises(ValueError):
        call_price(spot, strike, maturity, rate, volatility)
    with pytest.raises(ValueError):
        put_price(spot, strike, maturity, rate, volatility)
