import pytest

from ncx_derivatives.greeks import (
    call_delta,
    call_rho,
    call_theta,
    gamma,
    put_delta,
    put_rho,
    put_theta,
    vega,
)
from ncx_derivatives.pricing import call_price, put_price

SPOT = 100.0
STRIKE = 100.0
MATURITY = 1.0
RATE = 0.05
VOLATILITY = 0.20


def test_call_delta_benchmark() -> None:
    result = call_delta(
        SPOT,
        STRIKE,
        MATURITY,
        RATE,
        VOLATILITY,
    )

    assert result == pytest.approx(0.636831, abs=1e-6)


def test_put_delta_benchmark() -> None:
    result = put_delta(
        SPOT,
        STRIKE,
        MATURITY,
        RATE,
        VOLATILITY,
    )

    assert result == pytest.approx(-0.363169, abs=1e-6)


def test_gamma_benchmark() -> None:
    result = gamma(
        SPOT,
        STRIKE,
        MATURITY,
        RATE,
        VOLATILITY,
    )

    assert result == pytest.approx(0.018762, abs=1e-6)


def test_vega_benchmark() -> None:
    result = vega(
        SPOT,
        STRIKE,
        MATURITY,
        RATE,
        VOLATILITY,
    )

    assert result == pytest.approx(37.524035, abs=1e-6)


def test_call_theta_benchmark() -> None:
    result = call_theta(
        SPOT,
        STRIKE,
        MATURITY,
        RATE,
        VOLATILITY,
    )

    assert result == pytest.approx(-6.414028, abs=1e-6)


def test_put_theta_benchmark() -> None:
    result = put_theta(
        SPOT,
        STRIKE,
        MATURITY,
        RATE,
        VOLATILITY,
    )

    assert result == pytest.approx(-1.657880, abs=1e-6)


def test_call_rho_benchmark() -> None:
    result = call_rho(
        SPOT,
        STRIKE,
        MATURITY,
        RATE,
        VOLATILITY,
    )

    assert result == pytest.approx(53.232482, abs=1e-6)


def test_put_rho_benchmark() -> None:
    result = put_rho(
        SPOT,
        STRIKE,
        MATURITY,
        RATE,
        VOLATILITY,
    )

    assert result == pytest.approx(-41.890461, abs=1e-6)

@pytest.mark.parametrize(
    ("spot", "strike", "maturity", "rate", "volatility"),
    [
        (100.0, 100.0, 1.0, 0.05, 0.20),
        (120.0, 100.0, 0.50, 0.03, 0.30),
        (80.0, 100.0, 2.00, -0.01, 0.40),
    ],
)
def test_call_delta_matches_central_difference(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
) -> None:
    step = 1e-4 * spot

    numerical_delta = (
        call_price(
            spot + step,
            strike,
            maturity,
            rate,
            volatility,
        )
        - call_price(
            spot - step,
            strike,
            maturity,
            rate,
            volatility,
        )
    ) / (2.0 * step)

    analytical_delta = call_delta(
        spot,
        strike,
        maturity,
        rate,
        volatility,
    )

    assert analytical_delta == pytest.approx(
        numerical_delta,
        rel=1e-6,
        abs=1e-8,
    )


@pytest.mark.parametrize(
    ("spot", "strike", "maturity", "rate", "volatility"),
    [
        (100.0, 100.0, 1.0, 0.05, 0.20),
        (120.0, 100.0, 0.50, 0.03, 0.30),
        (80.0, 100.0, 2.00, -0.01, 0.40),
    ],
)
def test_put_delta_matches_central_difference(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
) -> None:
    step = 1e-4 * spot

    numerical_delta = (
        put_price(
            spot + step,
            strike,
            maturity,
            rate,
            volatility,
        )
        - put_price(
            spot - step,
            strike,
            maturity,
            rate,
            volatility,
        )
    ) / (2.0 * step)

    analytical_delta = put_delta(
        spot,
        strike,
        maturity,
        rate,
        volatility,
    )

    assert analytical_delta == pytest.approx(
        numerical_delta,
        rel=1e-6,
        abs=1e-8,
    )


@pytest.mark.parametrize(
    ("spot", "strike", "maturity", "rate", "volatility"),
    [
        (100.0, 100.0, 1.0, 0.05, 0.20),
        (120.0, 100.0, 0.50, 0.03, 0.30),
        (80.0, 100.0, 2.00, -0.01, 0.40),
    ],
)
def test_gamma_matches_second_central_difference(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
) -> None:
    step = 1e-3 * spot

    price_up = call_price(
        spot + step,
        strike,
        maturity,
        rate,
        volatility,
    )
    price_mid = call_price(
        spot,
        strike,
        maturity,
        rate,
        volatility,
    )
    price_down = call_price(
        spot - step,
        strike,
        maturity,
        rate,
        volatility,
    )

    numerical_gamma = (
        price_up - 2.0 * price_mid + price_down
    ) / (step * step)

    analytical_gamma = gamma(
        spot,
        strike,
        maturity,
        rate,
        volatility,
    )

    assert analytical_gamma == pytest.approx(
        numerical_gamma,
        rel=1e-5,
        abs=1e-8,
    )


@pytest.mark.parametrize(
    ("spot", "strike", "maturity", "rate", "volatility"),
    [
        (100.0, 100.0, 1.0, 0.05, 0.20),
        (120.0, 100.0, 0.50, 0.03, 0.30),
        (80.0, 100.0, 2.00, -0.01, 0.40),
    ],
)
def test_vega_matches_central_difference(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
) -> None:
    step = 1e-5

    numerical_vega = (
        call_price(
            spot,
            strike,
            maturity,
            rate,
            volatility + step,
        )
        - call_price(
            spot,
            strike,
            maturity,
            rate,
            volatility - step,
        )
    ) / (2.0 * step)

    analytical_vega = vega(
        spot,
        strike,
        maturity,
        rate,
        volatility,
    )

    assert analytical_vega == pytest.approx(
        numerical_vega,
        rel=1e-6,
        abs=1e-7,
    )


@pytest.mark.parametrize(
    ("spot", "strike", "maturity", "rate", "volatility"),
    [
        (100.0, 100.0, 1.0, 0.05, 0.20),
        (120.0, 100.0, 0.50, 0.03, 0.30),
        (80.0, 100.0, 2.00, -0.01, 0.40),
    ],
)
def test_call_rho_matches_central_difference(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
) -> None:
    step = 1e-5

    numerical_rho = (
        call_price(
            spot,
            strike,
            maturity,
            rate + step,
            volatility,
        )
        - call_price(
            spot,
            strike,
            maturity,
            rate - step,
            volatility,
        )
    ) / (2.0 * step)

    analytical_rho = call_rho(
        spot,
        strike,
        maturity,
        rate,
        volatility,
    )

    assert analytical_rho == pytest.approx(
        numerical_rho,
        rel=1e-6,
        abs=1e-7,
    )


@pytest.mark.parametrize(
    ("spot", "strike", "maturity", "rate", "volatility"),
    [
        (100.0, 100.0, 1.0, 0.05, 0.20),
        (120.0, 100.0, 0.50, 0.03, 0.30),
        (80.0, 100.0, 2.00, -0.01, 0.40),
    ],
)
def test_put_rho_matches_central_difference(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
) -> None:
    step = 1e-5

    numerical_rho = (
        put_price(
            spot,
            strike,
            maturity,
            rate + step,
            volatility,
        )
        - put_price(
            spot,
            strike,
            maturity,
            rate - step,
            volatility,
        )
    ) / (2.0 * step)

    analytical_rho = put_rho(
        spot,
        strike,
        maturity,
        rate,
        volatility,
    )

    assert analytical_rho == pytest.approx(
        numerical_rho,
        rel=1e-6,
        abs=1e-7,
    )


def test_greeks_match_finite_differences_with_dividend_yield() -> None:
    spot = 100.0
    strike = 100.0
    maturity = 1.0
    rate = 0.05
    volatility = 0.20
    dividend_yield = 0.02

    spot_step = 1e-4 * spot
    gamma_step = 1e-3 * spot
    volatility_step = 1e-5
    rate_step = 1e-5
    maturity_step = 1e-5

    numerical_call_delta = (
        call_price(
            spot + spot_step,
            strike,
            maturity,
            rate,
            volatility,
            dividend_yield,
        )
        - call_price(
            spot - spot_step,
            strike,
            maturity,
            rate,
            volatility,
            dividend_yield,
        )
    ) / (2.0 * spot_step)

    numerical_put_delta = (
        put_price(
            spot + spot_step,
            strike,
            maturity,
            rate,
            volatility,
            dividend_yield,
        )
        - put_price(
            spot - spot_step,
            strike,
            maturity,
            rate,
            volatility,
            dividend_yield,
        )
    ) / (2.0 * spot_step)

    numerical_vega = (
        call_price(
            spot,
            strike,
            maturity,
            rate,
            volatility + volatility_step,
            dividend_yield,
        )
        - call_price(
            spot,
            strike,
            maturity,
            rate,
            volatility - volatility_step,
            dividend_yield,
        )
    ) / (2.0 * volatility_step)

    numerical_gamma = (
        call_price(
            spot + gamma_step,
            strike,
            maturity,
            rate,
            volatility,
            dividend_yield,
        )
        - 2.0
        * call_price(
            spot,
            strike,
            maturity,
            rate,
            volatility,
            dividend_yield,
        )
        + call_price(
            spot - gamma_step,
            strike,
            maturity,
            rate,
            volatility,
            dividend_yield,
        )
    ) / (gamma_step * gamma_step)

    numerical_call_rho = (
        call_price(
            spot,
            strike,
            maturity,
            rate + rate_step,
            volatility,
            dividend_yield,
        )
        - call_price(
            spot,
            strike,
            maturity,
            rate - rate_step,
            volatility,
            dividend_yield,
        )
    ) / (2.0 * rate_step)

    numerical_put_rho = (
        put_price(
            spot,
            strike,
            maturity,
            rate + rate_step,
            volatility,
            dividend_yield,
        )
        - put_price(
            spot,
            strike,
            maturity,
            rate - rate_step,
            volatility,
            dividend_yield,
        )
    ) / (2.0 * rate_step)

    call_maturity_derivative = (
        call_price(
            spot,
            strike,
            maturity + maturity_step,
            rate,
            volatility,
            dividend_yield,
        )
        - call_price(
            spot,
            strike,
            maturity - maturity_step,
            rate,
            volatility,
            dividend_yield,
        )
    ) / (2.0 * maturity_step)

    put_maturity_derivative = (
        put_price(
            spot,
            strike,
            maturity + maturity_step,
            rate,
            volatility,
            dividend_yield,
        )
        - put_price(
            spot,
            strike,
            maturity - maturity_step,
            rate,
            volatility,
            dividend_yield,
        )
    ) / (2.0 * maturity_step)

    assert call_delta(
        spot,
        strike,
        maturity,
        rate,
        volatility,
        dividend_yield,
    ) == pytest.approx(numerical_call_delta, rel=1e-6, abs=1e-8)
    assert put_delta(
        spot,
        strike,
        maturity,
        rate,
        volatility,
        dividend_yield,
    ) == pytest.approx(numerical_put_delta, rel=1e-6, abs=1e-8)
    assert vega(
        spot,
        strike,
        maturity,
        rate,
        volatility,
        dividend_yield,
    ) == pytest.approx(numerical_vega, rel=1e-6, abs=1e-7)
    assert gamma(
        spot,
        strike,
        maturity,
        rate,
        volatility,
        dividend_yield,
    ) == pytest.approx(numerical_gamma, rel=1e-5, abs=1e-8)
    assert call_theta(
        spot,
        strike,
        maturity,
        rate,
        volatility,
        dividend_yield,
    ) == pytest.approx(-call_maturity_derivative, rel=1e-6, abs=1e-7)
    assert put_theta(
        spot,
        strike,
        maturity,
        rate,
        volatility,
        dividend_yield,
    ) == pytest.approx(-put_maturity_derivative, rel=1e-6, abs=1e-7)
    assert call_rho(
        spot,
        strike,
        maturity,
        rate,
        volatility,
        dividend_yield,
    ) == pytest.approx(numerical_call_rho, rel=1e-6, abs=1e-7)
    assert put_rho(
        spot,
        strike,
        maturity,
        rate,
        volatility,
        dividend_yield,
    ) == pytest.approx(numerical_put_rho, rel=1e-6, abs=1e-7)

@pytest.mark.parametrize(
    ("spot", "strike", "maturity", "rate", "volatility"),
    [
        (100.0, 100.0, 1.0, 0.05, 0.20),
        (120.0, 100.0, 0.50, 0.03, 0.30),
        (80.0, 100.0, 2.00, -0.01, 0.40),
    ],
)
def test_call_theta_matches_maturity_difference(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
) -> None:
    step = 1e-5

    maturity_derivative = (
        call_price(
            spot,
            strike,
            maturity + step,
            rate,
            volatility,
        )
        - call_price(
            spot,
            strike,
            maturity - step,
            rate,
            volatility,
        )
    ) / (2.0 * step)

    analytical_theta = call_theta(
        spot,
        strike,
        maturity,
        rate,
        volatility,
    )

    assert analytical_theta == pytest.approx(
        -maturity_derivative,
        rel=1e-6,
        abs=1e-7,
    )


@pytest.mark.parametrize(
    ("spot", "strike", "maturity", "rate", "volatility"),
    [
        (100.0, 100.0, 1.0, 0.05, 0.20),
        (120.0, 100.0, 0.50, 0.03, 0.30),
        (80.0, 100.0, 2.00, -0.01, 0.40),
    ],
)
def test_put_theta_matches_maturity_difference(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
) -> None:
    step = 1e-5

    maturity_derivative = (
        put_price(
            spot,
            strike,
            maturity + step,
            rate,
            volatility,
        )
        - put_price(
            spot,
            strike,
            maturity - step,
            rate,
            volatility,
        )
    ) / (2.0 * step)

    analytical_theta = put_theta(
        spot,
        strike,
        maturity,
        rate,
        volatility,
    )

    assert analytical_theta == pytest.approx(
        -maturity_derivative,
        rel=1e-6,
        abs=1e-7,
    )

def test_call_and_put_delta_differ_by_one() -> None:
    call = call_delta(100.0, 100.0, 1.0, 0.05, 0.20)
    put = put_delta(100.0, 100.0, 1.0, 0.05, 0.20)

    assert call - put == pytest.approx(1.0, abs=1e-12)


def test_gamma_is_positive() -> None:
    result = gamma(100.0, 100.0, 1.0, 0.05, 0.20)

    assert result > 0.0


def test_vega_is_positive() -> None:
    result = vega(100.0, 100.0, 1.0, 0.05, 0.20)

    assert result > 0.0
