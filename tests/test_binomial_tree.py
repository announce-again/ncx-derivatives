import pytest

from ncx_derivatives.pricing import (
    BinomialTreeResult,
    binomial_call_price,
    binomial_put_price,
    call_price,
    put_price,
)
from ncx_derivatives.pricing.binomial_tree import _payoff


def test_crr_handles_indistinguishable_up_and_down_factors():
    assert binomial_call_price(100, 90, 1, 0, 1e-16, steps=100) == 10


@pytest.mark.parametrize(
    ("spot", "strike", "maturity", "rate", "volatility", "dividend_yield"),
    [
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.02),
        (120.0, 100.0, 0.50, 0.03, 0.30, 0.01),
        (80.0, 100.0, 2.00, 0.01, 0.35, 0.04),
        (100.0, 120.0, 0.25, -0.02, 0.15, 0.03),
    ],
)
def test_european_tree_converges_to_black_scholes_merton(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float,
) -> None:
    steps = 1200

    tree_call = binomial_call_price(
        spot,
        strike,
        maturity,
        rate,
        volatility,
        dividend_yield,
        steps=steps,
    )
    tree_put = binomial_put_price(
        spot,
        strike,
        maturity,
        rate,
        volatility,
        dividend_yield,
        steps=steps,
    )

    analytic_call = call_price(
        spot,
        strike,
        maturity,
        rate,
        volatility,
        dividend_yield,
    )
    analytic_put = put_price(
        spot,
        strike,
        maturity,
        rate,
        volatility,
        dividend_yield,
    )

    assert tree_call == pytest.approx(analytic_call, abs=2e-2)
    assert tree_put == pytest.approx(analytic_put, abs=2e-2)


@pytest.mark.parametrize(
    ("spot", "strike", "maturity", "rate", "volatility", "dividend_yield"),
    [
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.00),
        (90.0, 100.0, 1.0, 0.05, 0.25, 0.00),
        (110.0, 100.0, 2.0, 0.03, 0.30, 0.02),
    ],
)
def test_american_options_are_at_least_european_value(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float,
) -> None:
    european_call = binomial_call_price(
        spot,
        strike,
        maturity,
        rate,
        volatility,
        dividend_yield,
        steps=400,
    )
    american_call = binomial_call_price(
        spot,
        strike,
        maturity,
        rate,
        volatility,
        dividend_yield,
        steps=400,
        american=True,
    )
    european_put = binomial_put_price(
        spot,
        strike,
        maturity,
        rate,
        volatility,
        dividend_yield,
        steps=400,
    )
    american_put = binomial_put_price(
        spot,
        strike,
        maturity,
        rate,
        volatility,
        dividend_yield,
        steps=400,
        american=True,
    )

    assert american_call >= european_call
    assert american_put >= european_put


def test_non_dividend_american_call_equals_european_call() -> None:
    european_call = binomial_call_price(
        100.0,
        100.0,
        1.0,
        0.05,
        0.20,
        steps=500,
    )
    american_call = binomial_call_price(
        100.0,
        100.0,
        1.0,
        0.05,
        0.20,
        steps=500,
        american=True,
    )

    assert american_call == pytest.approx(european_call, abs=1e-12)


def test_tree_error_decreases_with_depth() -> None:
    analytic_call = call_price(100.0, 100.0, 1.0, 0.05, 0.20, 0.02)

    coarse_error = abs(
        binomial_call_price(
            100.0,
            100.0,
            1.0,
            0.05,
            0.20,
            0.02,
            steps=25,
        )
        - analytic_call
    )
    fine_error = abs(
        binomial_call_price(
            100.0,
            100.0,
            1.0,
            0.05,
            0.20,
            0.02,
            steps=400,
        )
        - analytic_call
    )

    assert fine_error < coarse_error


@pytest.mark.parametrize(
    ("price_function", "intrinsic_value"),
    [
        (binomial_call_price, 20.0),
        (binomial_put_price, 20.0),
    ],
)
def test_american_prices_respect_intrinsic_value_lower_bound(
    price_function,
    intrinsic_value: float,
) -> None:
    spot = 120.0 if price_function is binomial_call_price else 80.0

    price = price_function(
        spot,
        100.0,
        1.0,
        0.05,
        0.20,
        steps=300,
        american=True,
    )

    assert price >= intrinsic_value


def test_deep_in_the_money_american_put_has_early_exercise_boundary() -> None:
    result = binomial_put_price(
        spot=50.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        steps=200,
        american=True,
        return_exercise_boundary=True,
    )

    assert isinstance(result, BinomialTreeResult)
    assert result.price > put_price(50.0, 100.0, 1.0, 0.05, 0.20)
    assert any(boundary is not None for boundary in result.exercise_boundary)


def test_european_tree_does_not_report_exercise_boundary() -> None:
    result = binomial_put_price(
        spot=50.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        steps=100,
        return_exercise_boundary=True,
    )

    assert isinstance(result, BinomialTreeResult)
    assert all(boundary is None for boundary in result.exercise_boundary)


def test_expiry_tree_result_does_not_report_exercise_boundary() -> None:
    result = binomial_put_price(
        spot=50.0,
        strike=100.0,
        maturity=0.0,
        rate=0.05,
        volatility=0.20,
        steps=100,
        american=True,
        return_exercise_boundary=True,
    )

    assert isinstance(result, BinomialTreeResult)
    assert result.price == 50.0
    assert result.exercise_boundary == (None,)


@pytest.mark.parametrize(
    ("spot", "strike", "maturity", "volatility"),
    [
        (60.0, 100.0, 0.10, 0.50),
        (100.0, 100.0, 1.00, 0.20),
        (160.0, 100.0, 3.00, 0.35),
    ],
)
def test_tree_is_stable_across_moneyness_and_maturities(
    spot: float,
    strike: float,
    maturity: float,
    volatility: float,
) -> None:
    call = binomial_call_price(
        spot,
        strike,
        maturity,
        0.03,
        volatility,
        0.01,
        steps=500,
    )
    put = binomial_put_price(
        spot,
        strike,
        maturity,
        0.03,
        volatility,
        0.01,
        steps=500,
    )

    assert call >= 0.0
    assert put >= 0.0


@pytest.mark.parametrize(
    ("spot", "strike", "maturity", "rate", "volatility", "steps"),
    [
        (0.0, 100.0, 1.0, 0.05, 0.20, 100),
        (100.0, 0.0, 1.0, 0.05, 0.20, 100),
        (100.0, 100.0, -1.0, 0.05, 0.20, 100),
        (100.0, 100.0, 1.0, 0.05, -0.20, 100),
        (100.0, 100.0, 1.0, 0.05, 0.20, 0),
    ],
)
def test_binomial_tree_rejects_invalid_inputs(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    steps: int,
) -> None:
    with pytest.raises(ValueError):
        binomial_call_price(
            spot,
            strike,
            maturity,
            rate,
            volatility,
            steps=steps,
        )
    with pytest.raises(ValueError):
        binomial_put_price(
            spot,
            strike,
            maturity,
            rate,
            volatility,
            steps=steps,
        )


@pytest.mark.parametrize("steps", [1.5, "100", True])
def test_binomial_tree_rejects_non_integer_steps(steps) -> None:
    with pytest.raises(ValueError):
        binomial_call_price(
            100.0,
            100.0,
            1.0,
            0.05,
            0.20,
            steps=steps,
        )


def test_payoff_rejects_invalid_option_type() -> None:
    with pytest.raises(ValueError):
        _payoff("straddle", 100.0, 100.0)
