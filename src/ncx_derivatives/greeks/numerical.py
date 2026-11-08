from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

DifferenceMethod = Literal["central", "forward", "backward"]
BumpParameter = Literal[
    "spot",
    "strike",
    "maturity",
    "rate",
    "volatility",
    "dividend_yield",
]
PriceFunction = Callable[..., Any]
PriceKwargs = dict[str, Any]


@dataclass(frozen=True)
class NumericalGreekResult:
    """Numerical Greek estimate with bump-and-revalue diagnostics."""

    value: float
    parameter: str
    bump: float
    method: DifferenceMethod
    evaluations: int
    prices: tuple[float, ...]


def _extract_price(value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError("price function must not return bool")

    if isinstance(value, int | float):
        return float(value)

    price = getattr(value, "price", None)
    if isinstance(price, bool):
        raise TypeError("price attribute must not be bool")

    if isinstance(price, int | float):
        return float(price)

    raise TypeError("price function must return a float or object with price")


def _validate_method(method: DifferenceMethod) -> None:
    if method not in {"central", "forward", "backward"}:
        raise ValueError("method must be central, forward, or backward")


def _adaptive_bump(value: float, relative_bump: float) -> float:
    if relative_bump <= 0.0:
        raise ValueError("relative bump must be positive")
    return relative_bump * max(abs(value), 1.0)


def _validate_bump(bump: float) -> None:
    if bump <= 0.0:
        raise ValueError("bump must be positive")


def _validate_bumped_parameter(
    parameter: BumpParameter,
    value: float,
) -> None:
    if parameter in {"spot", "strike"} and value <= 0.0:
        raise ValueError(f"{parameter} bump crosses zero")
    if parameter in {"maturity", "volatility"} and value < 0.0:
        raise ValueError(f"{parameter} bump crosses below zero")


def _price(
    price_function: PriceFunction,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float,
    price_kwargs: PriceKwargs,
) -> float:
    return _extract_price(
        price_function(
            spot,
            strike,
            maturity,
            rate,
            volatility,
            dividend_yield,
            **price_kwargs,
        )
    )


def _first_derivative(
    price_function: PriceFunction,
    parameter: BumpParameter,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float,
    *,
    bump: float | None,
    relative_bump: float,
    method: DifferenceMethod,
    sign: float,
    return_diagnostics: bool,
    price_kwargs: PriceKwargs,
) -> float | NumericalGreekResult:
    _validate_method(method)

    base_parameters = {
        "spot": spot,
        "strike": strike,
        "maturity": maturity,
        "rate": rate,
        "volatility": volatility,
        "dividend_yield": dividend_yield,
    }
    parameter_value = base_parameters[parameter]
    step = bump if bump is not None else _adaptive_bump(
        parameter_value,
        relative_bump,
    )
    _validate_bump(step)

    def evaluate(updated_value: float) -> float:
        _validate_bumped_parameter(parameter, updated_value)

        parameters = dict(base_parameters)
        parameters[parameter] = updated_value
        return _price(
            price_function,
            parameters["spot"],
            parameters["strike"],
            parameters["maturity"],
            parameters["rate"],
            parameters["volatility"],
            parameters["dividend_yield"],
            price_kwargs,
        )

    if method == "central":
        down = evaluate(parameter_value - step)
        up = evaluate(parameter_value + step)
        derivative = (up - down) / (2.0 * step)
        prices = (down, up)
    elif method == "forward":
        base = evaluate(parameter_value)
        up = evaluate(parameter_value + step)
        derivative = (up - base) / step
        prices = (base, up)
    else:
        down = evaluate(parameter_value - step)
        base = evaluate(parameter_value)
        derivative = (base - down) / step
        prices = (down, base)

    value = sign * derivative

    if return_diagnostics:
        return NumericalGreekResult(
            value=value,
            parameter=parameter,
            bump=step,
            method=method,
            evaluations=len(prices),
            prices=prices,
        )
    return value


def numerical_delta(
    price_function: PriceFunction,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    *,
    bump: float | None = None,
    relative_bump: float = 1e-4,
    method: DifferenceMethod = "central",
    return_diagnostics: bool = False,
    **price_kwargs: Any,
) -> float | NumericalGreekResult:
    """Estimate Delta by bumping spot and revaluing."""

    return _first_derivative(
        price_function,
        "spot",
        spot,
        strike,
        maturity,
        rate,
        volatility,
        dividend_yield,
        bump=bump,
        relative_bump=relative_bump,
        method=method,
        sign=1.0,
        return_diagnostics=return_diagnostics,
        price_kwargs=price_kwargs,
    )


def numerical_vega(
    price_function: PriceFunction,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    *,
    bump: float | None = None,
    relative_bump: float = 1e-4,
    method: DifferenceMethod = "central",
    return_diagnostics: bool = False,
    **price_kwargs: Any,
) -> float | NumericalGreekResult:
    """Estimate Vega by bumping volatility and revaluing."""

    return _first_derivative(
        price_function,
        "volatility",
        spot,
        strike,
        maturity,
        rate,
        volatility,
        dividend_yield,
        bump=bump,
        relative_bump=relative_bump,
        method=method,
        sign=1.0,
        return_diagnostics=return_diagnostics,
        price_kwargs=price_kwargs,
    )


def numerical_rho(
    price_function: PriceFunction,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    *,
    bump: float | None = None,
    relative_bump: float = 1e-4,
    method: DifferenceMethod = "central",
    return_diagnostics: bool = False,
    **price_kwargs: Any,
) -> float | NumericalGreekResult:
    """Estimate Rho by bumping the interest rate and revaluing."""

    return _first_derivative(
        price_function,
        "rate",
        spot,
        strike,
        maturity,
        rate,
        volatility,
        dividend_yield,
        bump=bump,
        relative_bump=relative_bump,
        method=method,
        sign=1.0,
        return_diagnostics=return_diagnostics,
        price_kwargs=price_kwargs,
    )


def numerical_theta(
    price_function: PriceFunction,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    *,
    bump: float | None = None,
    relative_bump: float = 1e-4,
    method: DifferenceMethod = "central",
    return_diagnostics: bool = False,
    **price_kwargs: Any,
) -> float | NumericalGreekResult:
    """Estimate market theta as negative derivative with respect to maturity."""

    return _first_derivative(
        price_function,
        "maturity",
        spot,
        strike,
        maturity,
        rate,
        volatility,
        dividend_yield,
        bump=bump,
        relative_bump=relative_bump,
        method=method,
        sign=-1.0,
        return_diagnostics=return_diagnostics,
        price_kwargs=price_kwargs,
    )


def numerical_gamma(
    price_function: PriceFunction,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    *,
    bump: float | None = None,
    relative_bump: float = 1e-3,
    method: DifferenceMethod = "central",
    return_diagnostics: bool = False,
    **price_kwargs: Any,
) -> float | NumericalGreekResult:
    """Estimate Gamma by second-order bump-and-revalue on spot."""

    _validate_method(method)
    step = bump if bump is not None else _adaptive_bump(
        spot,
        relative_bump,
    )
    _validate_bump(step)

    def evaluate(updated_spot: float) -> float:
        _validate_bumped_parameter("spot", updated_spot)
        return _price(
            price_function,
            updated_spot,
            strike,
            maturity,
            rate,
            volatility,
            dividend_yield,
            price_kwargs,
        )

    if method == "central":
        down = evaluate(spot - step)
        base = evaluate(spot)
        up = evaluate(spot + step)
        value = (up - 2.0 * base + down) / (step * step)
        prices = (down, base, up)
    elif method == "forward":
        base = evaluate(spot)
        up = evaluate(spot + step)
        up_two = evaluate(spot + 2.0 * step)
        value = (up_two - 2.0 * up + base) / (step * step)
        prices = (base, up, up_two)
    else:
        down_two = evaluate(spot - 2.0 * step)
        down = evaluate(spot - step)
        base = evaluate(spot)
        value = (base - 2.0 * down + down_two) / (step * step)
        prices = (down_two, down, base)

    if return_diagnostics:
        return NumericalGreekResult(
            value=value,
            parameter="spot",
            bump=step,
            method=method,
            evaluations=len(prices),
            prices=prices,
        )
    return value
