from collections.abc import Callable
from math import exp, inf, isclose, isfinite, pi, sqrt

from ncx_derivatives.greeks import vega
from ncx_derivatives.pricing import call_price, put_price

PriceFunction = Callable[[float, float, float, float, float, float], float]

IMPLIED_VOLATILITY_PRICE_ABSOLUTE_TOLERANCE = 1e-10
IMPLIED_VOLATILITY_PRICE_RELATIVE_TOLERANCE = 1e-12


class ImpliedVolatilityError(ValueError):
    """Base error for implied-volatility inversion failures."""


class ImpliedVolatilityInputError(ImpliedVolatilityError):
    """Raised when solver inputs are outside the supported domain."""


class ImpliedVolatilityBoundsError(ImpliedVolatilityError):
    """Raised when a price violates no-arbitrage bounds."""


class ImpliedVolatilityConvergenceError(ImpliedVolatilityError):
    """Raised when the solver cannot bracket or converge to an IV."""


def _validate_inputs(
    option_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
) -> None:
    if not all(isfinite(value) for value in (option_price, spot, strike, maturity, rate, dividend_yield)):
        raise ImpliedVolatilityInputError("price and market inputs must be finite")
    if option_price < 0.0:
        raise ImpliedVolatilityInputError("option price must be non-negative")
    if spot <= 0.0:
        raise ImpliedVolatilityInputError("spot must be positive")
    if strike <= 0.0:
        raise ImpliedVolatilityInputError("strike must be positive")
    if maturity < 0.0:
        raise ImpliedVolatilityInputError("maturity must be non-negative")


def _no_arbitrage_bounds(
    option_type: str,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
) -> tuple[float, float]:
    if maturity == 0.0:
        payoff = (
            max(spot - strike, 0.0)
            if option_type == "call"
            else max(strike - spot, 0.0)
        )
        return payoff, payoff

    discounted_strike = strike * exp(-rate * maturity)
    discounted_spot = spot * exp(-dividend_yield * maturity)

    if option_type == "call":
        return max(discounted_spot - discounted_strike, 0.0), discounted_spot

    return max(discounted_strike - discounted_spot, 0.0), discounted_strike


def _validate_price_bounds(
    option_price: float,
    lower_bound: float,
    upper_bound: float,
    tolerance: float,
) -> None:
    if option_price < lower_bound and not _is_close_to_price_bound(
        option_price,
        lower_bound,
        absolute_tolerance=tolerance,
    ):
        raise ImpliedVolatilityBoundsError(
            "option price is below no-arbitrage lower bound",
        )
    if option_price > upper_bound and not _is_close_to_price_bound(
        option_price,
        upper_bound,
        absolute_tolerance=tolerance,
    ):
        raise ImpliedVolatilityBoundsError(
            "option price is above no-arbitrage upper bound",
        )


def _is_close_to_price_bound(
    price: float,
    bound: float,
    *,
    absolute_tolerance: float = IMPLIED_VOLATILITY_PRICE_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = IMPLIED_VOLATILITY_PRICE_RELATIVE_TOLERANCE,
) -> bool:
    return isclose(
        price,
        bound,
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    )


def _initial_volatility_guess(
    option_price: float,
    lower_bound: float,
    spot: float,
    maturity: float,
) -> float:
    extrinsic_value = max(option_price - lower_bound, 0.0)
    if extrinsic_value == 0.0:
        return 0.20

    guess = sqrt(2.0 * pi / maturity) * (extrinsic_value / spot)
    return min(max(guess, 1e-6), 5.0)


def _implied_volatility(
    option_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    price_function: PriceFunction,
    option_type: str,
    *,
    price_tolerance: float = IMPLIED_VOLATILITY_PRICE_ABSOLUTE_TOLERANCE,
    volatility_tolerance: float = 1e-10,
    max_iterations: int = 100,
    max_volatility: float = 10.0,
    use_newton: bool = True,
) -> float:
    _validate_inputs(option_price, spot, strike, maturity, rate, dividend_yield)
    for value in (price_tolerance, volatility_tolerance, max_volatility):
        if isinstance(value, bool) or not isfinite(value) or value <= 0.0:
            raise ImpliedVolatilityInputError("solver tolerances and max volatility must be finite and positive")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ImpliedVolatilityInputError("max_iterations must be a positive integer")

    lower_bound, upper_bound = _no_arbitrage_bounds(
        option_type,
        spot,
        strike,
        maturity,
        rate,
        dividend_yield,
    )
    _validate_price_bounds(
        option_price,
        lower_bound,
        upper_bound,
        price_tolerance,
    )

    if option_price <= lower_bound or _is_close_to_price_bound(
        option_price,
        lower_bound,
        absolute_tolerance=price_tolerance,
    ):
        return 0.0
    if maturity == 0.0:
        return 0.0
    if option_price >= upper_bound or _is_close_to_price_bound(
        option_price,
        upper_bound,
        absolute_tolerance=price_tolerance,
    ):
        return inf

    low = 0.0
    high = 1.0

    while high < max_volatility:
        if (
            price_function(
                spot,
                strike,
                maturity,
                rate,
                high,
                dividend_yield,
            )
            >= option_price
        ):
            break
        high *= 2.0
    else:
        high = max_volatility

    if (
        price_function(
            spot,
            strike,
            maturity,
            rate,
            high,
            dividend_yield,
        )
        < option_price
    ):
        raise ImpliedVolatilityConvergenceError(
            "implied volatility exceeds solver upper bound",
        )

    sigma = _initial_volatility_guess(
        option_price,
        lower_bound,
        spot,
        maturity,
    )
    sigma = min(max(sigma, low), high)

    for _ in range(max_iterations):
        model_price = price_function(
            spot,
            strike,
            maturity,
            rate,
            sigma,
            dividend_yield,
        )
        price_error = model_price - option_price

        if abs(price_error) <= price_tolerance:
            return sigma

        if price_error > 0.0:
            high = sigma
        else:
            low = sigma

        if high - low <= volatility_tolerance:
            return 0.5 * (low + high)

        next_sigma = 0.5 * (low + high)

        if use_newton and sigma > 0.0:
            option_vega = vega(
                spot,
                strike,
                maturity,
                rate,
                sigma,
                dividend_yield,
            )
            if option_vega > 1e-12:
                newton_sigma = sigma - price_error / option_vega
                if low < newton_sigma < high:
                    next_sigma = newton_sigma

        sigma = next_sigma

    raise ImpliedVolatilityConvergenceError("implied volatility solver exhausted max_iterations")


def call_implied_volatility(
    option_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float = 0.0,
    *,
    price_tolerance: float = IMPLIED_VOLATILITY_PRICE_ABSOLUTE_TOLERANCE,
    volatility_tolerance: float = 1e-10,
    max_iterations: int = 100,
    max_volatility: float = 10.0,
) -> float:
    """Return Black-Scholes-Merton implied volatility for a European call."""

    return _implied_volatility(
        option_price,
        spot,
        strike,
        maturity,
        rate,
        dividend_yield,
        call_price,
        "call",
        price_tolerance=price_tolerance,
        volatility_tolerance=volatility_tolerance,
        max_iterations=max_iterations,
        max_volatility=max_volatility,
    )


def put_implied_volatility(
    option_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float = 0.0,
    *,
    price_tolerance: float = IMPLIED_VOLATILITY_PRICE_ABSOLUTE_TOLERANCE,
    volatility_tolerance: float = 1e-10,
    max_iterations: int = 100,
    max_volatility: float = 10.0,
) -> float:
    """Return Black-Scholes-Merton implied volatility for a European put."""

    return _implied_volatility(
        option_price,
        spot,
        strike,
        maturity,
        rate,
        dividend_yield,
        put_price,
        "put",
        price_tolerance=price_tolerance,
        volatility_tolerance=volatility_tolerance,
        max_iterations=max_iterations,
        max_volatility=max_volatility,
    )
