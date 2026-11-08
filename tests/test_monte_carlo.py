from math import exp, isfinite, log, nextafter

import pytest

from ncx_derivatives.monte_carlo import (
    MonteCarloResult,
    monte_carlo_call_price,
    monte_carlo_put_price,
    simulate_gbm_terminal_prices,
)
from ncx_derivatives.pricing import (
    binomial_call_price,
    call_price,
    put_price,
)


@pytest.mark.parametrize(("simulations", "antithetic"), [(1, False), (2, True)])
def test_stochastic_mc_requires_two_independent_observations(simulations, antithetic):
    with pytest.raises(ValueError, match="independent observations"):
        monte_carlo_call_price(100, 100, 1, 0.05, 0.2, simulations=simulations, antithetic=antithetic)
    result = monte_carlo_call_price(100, 100, 0, 0.05, 0.2, simulations=simulations, antithetic=antithetic)
    assert result.standard_error == 0


def test_mc_confidence_level_nearest_one_still_has_finite_interval():
    result = monte_carlo_call_price(100, 100, 1, 0.05, 0.2, simulations=10, seed=1, confidence_level=nextafter(1, 0))
    assert all(isfinite(bound) for bound in result.confidence_interval)
    assert result.confidence_interval[0] < result.price < result.confidence_interval[1]


def test_gbm_terminal_mean_matches_risk_neutral_forward() -> None:
    spot = 100.0
    maturity = 1.0
    rate = 0.05
    volatility = 0.20
    dividend_yield = 0.02

    terminal_prices = simulate_gbm_terminal_prices(
        spot,
        maturity,
        rate,
        volatility,
        dividend_yield,
        simulations=50_000,
        seed=7,
    )

    sample_mean = sum(terminal_prices) / len(terminal_prices)
    expected_mean = spot * exp((rate - dividend_yield) * maturity)

    assert sample_mean == pytest.approx(expected_mean, rel=4e-3)


def test_gbm_log_return_variance_matches_model_variance() -> None:
    spot = 100.0
    maturity = 1.5
    volatility = 0.30

    terminal_prices = simulate_gbm_terminal_prices(
        spot,
        maturity,
        0.03,
        volatility,
        0.01,
        simulations=50_000,
        seed=11,
    )

    log_returns = tuple(log(price / spot) for price in terminal_prices)
    mean_log_return = sum(log_returns) / len(log_returns)
    sample_variance = sum(
        (value - mean_log_return) ** 2 for value in log_returns
    ) / (len(log_returns) - 1)

    assert sample_variance == pytest.approx(
        volatility * volatility * maturity,
        rel=1e-2,
    )


def test_gbm_terminal_prices_are_reproducible_with_seed() -> None:
    first_run = simulate_gbm_terminal_prices(
        100.0,
        1.0,
        0.05,
        0.20,
        simulations=10,
        seed=42,
    )
    second_run = simulate_gbm_terminal_prices(
        100.0,
        1.0,
        0.05,
        0.20,
        simulations=10,
        seed=42,
    )

    assert first_run == second_run


def test_monte_carlo_result_shape() -> None:
    result = monte_carlo_call_price(
        100.0,
        100.0,
        1.0,
        0.05,
        0.20,
        0.02,
        simulations=10_000,
        seed=1,
    )

    assert isinstance(result, MonteCarloResult)
    assert result.price > 0.0
    assert result.standard_error > 0.0
    assert result.confidence_interval[0] < result.price
    assert result.confidence_interval[1] > result.price
    assert result.simulations == 10_000


@pytest.mark.parametrize(
    ("price_function", "analytic_function"),
    [
        (monte_carlo_call_price, call_price),
        (monte_carlo_put_price, put_price),
    ],
)
def test_monte_carlo_prices_converge_to_black_scholes_merton(
    price_function,
    analytic_function,
) -> None:
    result = price_function(
        100.0,
        100.0,
        1.0,
        0.05,
        0.20,
        0.02,
        simulations=80_000,
        seed=123,
        control_variate=True,
    )
    analytic_price = analytic_function(
        100.0,
        100.0,
        1.0,
        0.05,
        0.20,
        0.02,
    )

    assert result.price == pytest.approx(analytic_price, abs=3e-2)
    assert (
        result.confidence_interval[0]
        <= analytic_price
        <= result.confidence_interval[1]
    )


def test_monte_carlo_price_compares_to_binomial_tree() -> None:
    result = monte_carlo_call_price(
        100.0,
        100.0,
        1.0,
        0.05,
        0.20,
        0.02,
        simulations=80_000,
        seed=321,
        control_variate=True,
    )
    tree_price = binomial_call_price(
        100.0,
        100.0,
        1.0,
        0.05,
        0.20,
        0.02,
        steps=1200,
    )

    assert result.price == pytest.approx(tree_price, abs=5e-2)


def test_standard_error_decreases_with_simulation_count() -> None:
    low_count = monte_carlo_call_price(
        100.0,
        100.0,
        1.0,
        0.05,
        0.20,
        simulations=2_000,
        seed=5,
    )
    high_count = monte_carlo_call_price(
        100.0,
        100.0,
        1.0,
        0.05,
        0.20,
        simulations=20_000,
        seed=5,
    )

    assert high_count.standard_error < low_count.standard_error


def test_zero_volatility_monte_carlo_is_deterministic() -> None:
    result = monte_carlo_put_price(
        100.0,
        105.0,
        1.5,
        0.04,
        0.0,
        0.02,
        simulations=1_000,
        seed=1,
    )
    expected_price = put_price(100.0, 105.0, 1.5, 0.04, 0.0, 0.02)

    assert result.price == pytest.approx(expected_price)
    assert result.standard_error == 0.0
    assert result.confidence_interval == pytest.approx(
        (expected_price, expected_price)
    )


def test_antithetic_variates_reduce_terminal_mean_error() -> None:
    spot = 100.0
    expected_mean = spot * exp((0.05 - 0.02) * 1.0)

    plain_prices = simulate_gbm_terminal_prices(
        spot,
        1.0,
        0.05,
        0.20,
        0.02,
        simulations=4_000,
        seed=99,
    )
    antithetic_prices = simulate_gbm_terminal_prices(
        spot,
        1.0,
        0.05,
        0.20,
        0.02,
        simulations=4_000,
        seed=99,
        antithetic=True,
    )

    plain_error = abs(
        sum(plain_prices) / len(plain_prices) - expected_mean
    )
    antithetic_error = abs(
        sum(antithetic_prices) / len(antithetic_prices) - expected_mean
    )

    assert antithetic_error < plain_error


def test_control_variate_reduces_standard_error() -> None:
    plain = monte_carlo_call_price(
        100.0,
        100.0,
        1.0,
        0.05,
        0.20,
        0.02,
        simulations=20_000,
        seed=77,
    )
    controlled = monte_carlo_call_price(
        100.0,
        100.0,
        1.0,
        0.05,
        0.20,
        0.02,
        simulations=20_000,
        seed=77,
        control_variate=True,
    )

    assert controlled.standard_error < plain.standard_error


@pytest.mark.parametrize(
    ("spot", "strike", "maturity", "volatility", "simulations"),
    [
        (0.0, 100.0, 1.0, 0.20, 100),
        (100.0, 0.0, 1.0, 0.20, 100),
        (100.0, 100.0, -1.0, 0.20, 100),
        (100.0, 100.0, 1.0, -0.20, 100),
        (100.0, 100.0, 1.0, 0.20, 0),
        (100.0, 100.0, 1.0, 0.20, True),
    ],
)
def test_monte_carlo_rejects_invalid_inputs(
    spot: float,
    strike: float,
    maturity: float,
    volatility: float,
    simulations: int,
) -> None:
    with pytest.raises(ValueError):
        monte_carlo_call_price(
            spot,
            strike,
            maturity,
            0.05,
            volatility,
            simulations=simulations,
        )


@pytest.mark.parametrize("confidence_level", [0.0, 1.0, -0.1])
def test_monte_carlo_rejects_invalid_confidence_level(
    confidence_level: float,
) -> None:
    with pytest.raises(ValueError):
        monte_carlo_call_price(
            100.0,
            100.0,
            1.0,
            0.05,
            0.20,
            confidence_level=confidence_level,
        )
