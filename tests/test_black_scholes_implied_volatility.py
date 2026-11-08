from math import exp, isinf, nan

import pytest

from ncx_derivatives.pricing import call_price, put_price
from ncx_derivatives.volatility import (
    ImpliedVolatilityConvergenceError,
    ImpliedVolatilityInputError,
    call_implied_volatility,
    put_implied_volatility,
)


@pytest.mark.parametrize("solver", [call_implied_volatility, put_implied_volatility])
def test_iv_rejects_non_finite_price_and_rate(solver):
    with pytest.raises(ImpliedVolatilityInputError, match="finite"):
        solver(nan, 100, 100, 1, 0.05)
    with pytest.raises(ImpliedVolatilityInputError, match="finite"):
        solver(10, 100, 100, 1, nan)


@pytest.mark.parametrize(
    ("price_function", "solver"),
    [(call_price, call_implied_volatility), (put_price, put_implied_volatility)],
)
def test_iv_public_iteration_limit_raises_on_exhaustion(price_function, solver):
    price = price_function(100, 110, 1, 0.03, 0.33)
    with pytest.raises(ImpliedVolatilityConvergenceError, match="max_iterations"):
        solver(price, 100, 110, 1, 0.03, max_iterations=1)
    assert solver(price, 100, 110, 1, 0.03, max_iterations=100) == pytest.approx(0.33)


@pytest.mark.parametrize(
    ("spot", "strike", "maturity", "rate", "volatility"),
    [
        (100.0, 100.0, 1.0, 0.05, 0.20),
        (140.0, 100.0, 0.50, 0.03, 0.25),
        (70.0, 100.0, 2.00, 0.01, 0.35),
        (100.0, 120.0, 0.25, -0.02, 0.15),
        (50.0, 150.0, 1.50, 0.04, 0.55),
        (150.0, 50.0, 1.50, 0.04, 0.55),
        (100.0, 100.0, 1.0 / 365.0, 0.01, 0.30),
    ],
)
def test_call_implied_volatility_round_trip(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
) -> None:
    market_price = call_price(
        spot,
        strike,
        maturity,
        rate,
        volatility,
    )

    implied_volatility = call_implied_volatility(
        market_price,
        spot,
        strike,
        maturity,
        rate,
    )

    assert implied_volatility == pytest.approx(
        volatility,
        rel=1e-8,
        abs=1e-8,
    )


@pytest.mark.parametrize(
    ("spot", "strike", "maturity", "rate", "volatility"),
    [
        (100.0, 100.0, 1.0, 0.05, 0.20),
        (140.0, 100.0, 0.50, 0.03, 0.25),
        (70.0, 100.0, 2.00, 0.01, 0.35),
        (100.0, 120.0, 0.25, -0.02, 0.15),
        (50.0, 150.0, 1.50, 0.04, 0.55),
        (150.0, 50.0, 1.50, 0.04, 0.55),
        (100.0, 100.0, 1.0 / 365.0, 0.01, 0.30),
    ],
)
def test_put_implied_volatility_round_trip(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
) -> None:
    market_price = put_price(
        spot,
        strike,
        maturity,
        rate,
        volatility,
    )

    implied_volatility = put_implied_volatility(
        market_price,
        spot,
        strike,
        maturity,
        rate,
    )

    assert implied_volatility == pytest.approx(
        volatility,
        rel=1e-8,
        abs=1e-8,
    )


@pytest.mark.parametrize(
    ("spot", "strike", "maturity", "rate", "volatility", "dividend_yield"),
    [
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.02),
        (140.0, 100.0, 0.50, 0.03, 0.25, 0.01),
        (70.0, 100.0, 2.00, 0.01, 0.35, 0.04),
        (100.0, 120.0, 0.25, -0.02, 0.15, 0.03),
    ],
)
def test_implied_volatility_round_trip_with_dividend_yield(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float,
) -> None:
    call_market_price = call_price(
        spot,
        strike,
        maturity,
        rate,
        volatility,
        dividend_yield,
    )
    put_market_price = put_price(
        spot,
        strike,
        maturity,
        rate,
        volatility,
        dividend_yield,
    )

    assert call_implied_volatility(
        call_market_price,
        spot,
        strike,
        maturity,
        rate,
        dividend_yield,
    ) == pytest.approx(volatility, rel=1e-8, abs=1e-8)
    assert put_implied_volatility(
        put_market_price,
        spot,
        strike,
        maturity,
        rate,
        dividend_yield,
    ) == pytest.approx(volatility, rel=1e-8, abs=1e-8)


def test_implied_volatility_reprices_market_price() -> None:
    spot = 100.0
    strike = 100.0
    maturity = 1.0
    rate = 0.05
    market_price = 10.4506

    implied_volatility = call_implied_volatility(
        market_price,
        spot,
        strike,
        maturity,
        rate,
    )

    repriced = call_price(
        spot,
        strike,
        maturity,
        rate,
        implied_volatility,
    )

    assert repriced == pytest.approx(market_price, abs=1e-8)


def test_lower_bound_price_returns_zero_implied_volatility() -> None:
    spot = 100.0
    strike = 105.0
    maturity = 1.5
    rate = 0.04
    discounted_strike = strike * exp(-rate * maturity)

    assert call_implied_volatility(
        max(spot - discounted_strike, 0.0),
        spot,
        strike,
        maturity,
        rate,
    ) == 0.0
    assert put_implied_volatility(
        max(discounted_strike - spot, 0.0),
        spot,
        strike,
        maturity,
        rate,
    ) == 0.0


def test_dividend_yield_changes_implied_volatility_bounds() -> None:
    spot = 100.0
    strike = 105.0
    maturity = 1.5
    rate = 0.04
    dividend_yield = 0.02

    discounted_spot = spot * exp(-dividend_yield * maturity)
    discounted_strike = strike * exp(-rate * maturity)

    assert call_implied_volatility(
        max(discounted_spot - discounted_strike, 0.0),
        spot,
        strike,
        maturity,
        rate,
        dividend_yield,
    ) == 0.0
    assert put_implied_volatility(
        max(discounted_strike - discounted_spot, 0.0),
        spot,
        strike,
        maturity,
        rate,
        dividend_yield,
    ) == 0.0


def test_upper_bound_price_returns_infinite_implied_volatility() -> None:
    assert isinf(call_implied_volatility(100.0, 100.0, 100.0, 1.0, 0.05))
    assert isinf(
        put_implied_volatility(
            100.0 * exp(-0.05),
            100.0,
            100.0,
            1.0,
            0.05,
        )
    )


@pytest.mark.parametrize(
    ("market_price", "spot", "strike", "maturity", "rate"),
    [
        (-1.0, 100.0, 100.0, 1.0, 0.05),
        (1.0, 0.0, 100.0, 1.0, 0.05),
        (1.0, 100.0, 0.0, 1.0, 0.05),
        (1.0, 100.0, 100.0, -1.0, 0.05),
    ],
)
def test_implied_volatility_rejects_invalid_inputs(
    market_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
) -> None:
    with pytest.raises(ValueError):
        call_implied_volatility(
            market_price,
            spot,
            strike,
            maturity,
            rate,
        )
    with pytest.raises(ValueError):
        put_implied_volatility(
            market_price,
            spot,
            strike,
            maturity,
            rate,
        )


def test_implied_volatility_rejects_prices_outside_bounds() -> None:
    with pytest.raises(ValueError):
        call_implied_volatility(0.01, 100.0, 50.0, 1.0, 0.05)

    with pytest.raises(ValueError):
        call_implied_volatility(101.0, 100.0, 100.0, 1.0, 0.05)

    with pytest.raises(ValueError):
        put_implied_volatility(0.01, 50.0, 100.0, 1.0, 0.05)

    with pytest.raises(ValueError):
        put_implied_volatility(100.0, 100.0, 100.0, 1.0, 0.05)
