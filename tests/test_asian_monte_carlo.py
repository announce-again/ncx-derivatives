from math import exp

import pytest

from ncx_derivatives.monte_carlo import (
    MonteCarloResult,
    monte_carlo_asian_call_price,
    monte_carlo_asian_put_price,
    simulate_gbm_paths,
)
from ncx_derivatives.pricing import call_price


@pytest.mark.parametrize(
    ("pricer", "spot"),
    [(monte_carlo_asian_call_price, 105), (monte_carlo_asian_put_price, 95)],
)
def test_asian_price_at_expiry_is_deterministic(pricer, spot):
    assert pricer(
        spot, 100, 0, 0.05, 0.2, monitoring_dates=12, simulations=1, control_variate=True
    ) == MonteCarloResult(5, 0, (5, 5), 1)


def test_stochastic_asian_price_requires_two_independent_observations():
    with pytest.raises(ValueError, match="independent observations"):
        monte_carlo_asian_call_price(100, 100, 1, 0.05, 0.2, simulations=2, antithetic=True)


def test_gbm_paths_are_reproducible_with_seed() -> None:
    first_run = simulate_gbm_paths(
        100.0,
        1.0,
        0.05,
        0.20,
        0.02,
        monitoring_dates=12,
        simulations=5,
        seed=42,
    )
    second_run = simulate_gbm_paths(
        100.0,
        1.0,
        0.05,
        0.20,
        0.02,
        monitoring_dates=12,
        simulations=5,
        seed=42,
    )

    assert first_run == second_run
    assert len(first_run) == 5
    assert len(first_run[0]) == 12


def test_low_volatility_asian_price_matches_deterministic_average() -> None:
    spot = 100.0
    strike = 100.0
    maturity = 1.0
    rate = 0.05
    dividend_yield = 0.02
    monitoring_dates = 12

    result = monte_carlo_asian_call_price(
        spot,
        strike,
        maturity,
        rate,
        0.0,
        dividend_yield,
        monitoring_dates=monitoring_dates,
        simulations=1_000,
        seed=1,
    )

    average_path = sum(
        spot
        * exp((rate - dividend_yield) * step / monitoring_dates)
        for step in range(1, monitoring_dates + 1)
    ) / monitoring_dates
    expected_price = exp(-rate * maturity) * max(
        average_path - strike,
        0.0,
    )

    assert result.price == pytest.approx(expected_price)
    assert result.standard_error == 0.0


def test_zero_volatility_control_variate_uses_geometric_average() -> None:
    spot = 100.0
    strike = 100.0
    maturity = 1.0
    rate = 0.08
    dividend_yield = 0.01
    monitoring_dates = (0.25, 0.5, 0.75, 1.0)

    plain = monte_carlo_asian_call_price(
        spot,
        strike,
        maturity,
        rate,
        0.0,
        dividend_yield,
        monitoring_dates=monitoring_dates,
        simulations=100,
        seed=1,
    )
    controlled = monte_carlo_asian_call_price(
        spot,
        strike,
        maturity,
        rate,
        0.0,
        dividend_yield,
        monitoring_dates=monitoring_dates,
        simulations=100,
        seed=1,
        control_variate=True,
    )

    assert controlled.price == pytest.approx(plain.price)
    assert controlled.standard_error == 0.0


def test_asian_call_decreases_and_put_increases_with_strike() -> None:
    low_strike_call = monte_carlo_asian_call_price(
        100.0,
        90.0,
        1.0,
        0.05,
        0.20,
        0.02,
        simulations=20_000,
        seed=7,
        control_variate=True,
    )
    high_strike_call = monte_carlo_asian_call_price(
        100.0,
        110.0,
        1.0,
        0.05,
        0.20,
        0.02,
        simulations=20_000,
        seed=7,
        control_variate=True,
    )
    low_strike_put = monte_carlo_asian_put_price(
        100.0,
        90.0,
        1.0,
        0.05,
        0.20,
        0.02,
        simulations=20_000,
        seed=7,
        control_variate=True,
    )
    high_strike_put = monte_carlo_asian_put_price(
        100.0,
        110.0,
        1.0,
        0.05,
        0.20,
        0.02,
        simulations=20_000,
        seed=7,
        control_variate=True,
    )

    assert low_strike_call.price > high_strike_call.price
    assert low_strike_put.price < high_strike_put.price


def test_asian_option_prices_are_non_negative() -> None:
    call = monte_carlo_asian_call_price(
        80.0,
        120.0,
        2.0,
        0.03,
        0.35,
        0.01,
        simulations=10_000,
        seed=9,
    )
    put = monte_carlo_asian_put_price(
        120.0,
        80.0,
        2.0,
        0.03,
        0.35,
        0.01,
        simulations=10_000,
        seed=9,
    )

    assert call.price >= 0.0
    assert put.price >= 0.0


def test_asian_monte_carlo_result_shape() -> None:
    result = monte_carlo_asian_call_price(
        100.0,
        100.0,
        1.0,
        0.05,
        0.20,
        0.02,
        monitoring_dates=(0.25, 0.5, 0.75, 1.0),
        simulations=10_000,
        seed=11,
    )

    assert isinstance(result, MonteCarloResult)
    assert result.price > 0.0
    assert result.standard_error > 0.0
    assert result.confidence_interval[0] < result.price
    assert result.confidence_interval[1] > result.price
    assert result.simulations == 10_000


def test_asian_standard_error_decreases_with_simulation_count() -> None:
    low_count = monte_carlo_asian_call_price(
        100.0,
        100.0,
        1.0,
        0.05,
        0.20,
        simulations=2_000,
        seed=5,
    )
    high_count = monte_carlo_asian_call_price(
        100.0,
        100.0,
        1.0,
        0.05,
        0.20,
        simulations=20_000,
        seed=5,
    )

    assert high_count.standard_error < low_count.standard_error


def test_asian_antithetic_variates_reduce_standard_error() -> None:
    plain = monte_carlo_asian_call_price(
        100.0,
        100.0,
        1.0,
        0.05,
        0.20,
        0.02,
        simulations=20_000,
        seed=77,
    )
    antithetic = monte_carlo_asian_call_price(
        100.0,
        100.0,
        1.0,
        0.05,
        0.20,
        0.02,
        simulations=20_000,
        seed=77,
        antithetic=True,
    )

    assert antithetic.standard_error < plain.standard_error


def test_geometric_control_variate_reduces_standard_error() -> None:
    plain = monte_carlo_asian_call_price(
        100.0,
        100.0,
        1.0,
        0.05,
        0.20,
        0.02,
        simulations=20_000,
        seed=13,
    )
    controlled = monte_carlo_asian_call_price(
        100.0,
        100.0,
        1.0,
        0.05,
        0.20,
        0.02,
        simulations=20_000,
        seed=13,
        control_variate=True,
    )

    assert controlled.standard_error < plain.standard_error


def test_asian_call_is_below_european_call_for_positive_rate_no_dividend() -> None:
    asian_call = monte_carlo_asian_call_price(
        100.0,
        100.0,
        1.0,
        0.05,
        0.20,
        0.0,
        simulations=50_000,
        seed=23,
        control_variate=True,
    )
    european_call = call_price(100.0, 100.0, 1.0, 0.05, 0.20, 0.0)

    assert asian_call.price < european_call


@pytest.mark.parametrize(
    "monitoring_dates",
    [0, True, (), (0.5, 0.5), (0.5, 1.5), [0.5, 1.0]],
)
def test_asian_pricing_rejects_invalid_monitoring_dates(
    monitoring_dates,
) -> None:
    with pytest.raises(ValueError):
        monte_carlo_asian_call_price(
            100.0,
            100.0,
            1.0,
            0.05,
            0.20,
            monitoring_dates=monitoring_dates,
        )
