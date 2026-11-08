from dataclasses import dataclass
from math import exp, sqrt
from typing import Literal

OptionType = Literal["call", "put"]
EXERCISE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class BinomialTreeResult:
    """Price plus optional early-exercise boundary by time step."""

    price: float
    exercise_boundary: tuple[float | None, ...]


def _validate_inputs(
    spot: float,
    strike: float,
    maturity: float,
    volatility: float,
    steps: int,
) -> None:
    if spot <= 0.0:
        raise ValueError("spot must be positive")
    if strike <= 0.0:
        raise ValueError("strike must be positive")
    if maturity < 0.0:
        raise ValueError("maturity must be non-negative")
    if volatility < 0.0:
        raise ValueError("volatility must be non-negative")
    if not isinstance(steps, int) or isinstance(steps, bool):
        raise ValueError("steps must be an integer")
    if steps <= 0:
        raise ValueError("steps must be positive")


def _payoff(
    option_type: OptionType,
    spot: float,
    strike: float,
) -> float:
    if option_type == "call":
        return max(spot - strike, 0.0)
    if option_type == "put":
        return max(strike - spot, 0.0)
    raise ValueError("option type must be 'call' or 'put'")


def _should_exercise(
    exercise_value: float,
    continuation_value: float,
) -> bool:
    scale = max(abs(exercise_value), abs(continuation_value), 1.0)
    return (
        exercise_value
        > continuation_value + EXERCISE_TOLERANCE * scale
    )


def _zero_volatility_price(
    option_type: OptionType,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    american: bool,
    steps: int,
    return_exercise_boundary: bool,
) -> float | BinomialTreeResult:
    if maturity == 0.0:
        price = _payoff(option_type, spot, strike)
        if return_exercise_boundary:
            return BinomialTreeResult(price, (None,))
        return price

    dt = maturity / steps
    discount = exp(-rate * dt)
    values = [
        _payoff(
            option_type,
            spot * exp((rate - dividend_yield) * maturity),
            strike,
        )
    ]
    boundaries: list[float | None] = [None] * (steps + 1)

    for step in range(steps - 1, -1, -1):
        node_spot = spot * exp((rate - dividend_yield) * step * dt)
        continuation_value = discount * values[0]
        exercise_value = _payoff(option_type, node_spot, strike)

        if american and _should_exercise(
            exercise_value,
            continuation_value,
        ):
            values[0] = exercise_value
            boundaries[step] = node_spot
        else:
            values[0] = continuation_value

    if return_exercise_boundary:
        return BinomialTreeResult(values[0], tuple(boundaries))
    return values[0]


def _binomial_price(
    option_type: OptionType,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float,
    steps: int,
    american: bool,
    return_exercise_boundary: bool,
) -> float | BinomialTreeResult:
    _validate_inputs(spot, strike, maturity, volatility, steps)

    dt = maturity / steps
    up = exp(volatility * sqrt(dt))
    if up == 1.0:
        return _zero_volatility_price(
            option_type,
            spot,
            strike,
            maturity,
            rate,
            dividend_yield,
            american,
            steps,
            return_exercise_boundary,
        )

    down = 1.0 / up
    growth = exp((rate - dividend_yield) * dt)
    probability = (growth - down) / (up - down)

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "risk-neutral probability is outside [0, 1]; increase steps"
        )

    discount = exp(-rate * dt)
    down_probability = 1.0 - probability

    node_spot = spot * (down ** steps)
    spot_ratio = up / down
    values = []
    for _ in range(steps + 1):
        values.append(_payoff(option_type, node_spot, strike))
        node_spot *= spot_ratio

    boundaries: list[float | None] | None = (
        [None] * (steps + 1) if return_exercise_boundary else None
    )
    if not american:
        for step in range(steps - 1, -1, -1):
            for node in range(step + 1):
                values[node] = discount * (
                    probability * values[node + 1]
                    + down_probability * values[node]
                )
        if boundaries is not None:
            return BinomialTreeResult(values[0], tuple(boundaries))
        return values[0]

    for step in range(steps - 1, -1, -1):
        node_spot = spot * (down ** step)

        for node in range(step + 1):
            continuation_value = discount * (
                probability * values[node + 1]
                + down_probability * values[node]
            )

            exercise_value = _payoff(option_type, node_spot, strike)

            if _should_exercise(
                exercise_value,
                continuation_value,
            ):
                values[node] = exercise_value
                # Spots rise with the node index. Calls use the first
                # exercising node and puts use the last.
                if boundaries is not None and (
                    option_type == "put" or boundaries[step] is None
                ):
                    boundaries[step] = node_spot
            else:
                values[node] = continuation_value

            node_spot *= spot_ratio

    if boundaries is not None:
        return BinomialTreeResult(values[0], tuple(boundaries))
    return values[0]


def binomial_call_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    *,
    steps: int = 100,
    american: bool = False,
    return_exercise_boundary: bool = False,
) -> float | BinomialTreeResult:
    """
    Price a call with a Cox-Ross-Rubinstein binomial tree.

    The implementation uses memory-efficient backward induction, storing
    only one layer of option values. Runtime is O(steps^2) and memory is
    O(steps). Very coarse trees can have visible discretisation error, and
    extreme carry assumptions may require more steps to keep the CRR
    risk-neutral probability inside [0, 1].

    The default return value is a float. If return_exercise_boundary=True,
    the function returns BinomialTreeResult. European exercise boundaries
    are reported as all None because early exercise is disabled.
    """

    return _binomial_price(
        "call",
        spot,
        strike,
        maturity,
        rate,
        volatility,
        dividend_yield,
        steps,
        american,
        return_exercise_boundary,
    )


def binomial_put_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    *,
    steps: int = 100,
    american: bool = False,
    return_exercise_boundary: bool = False,
) -> float | BinomialTreeResult:
    """
    Price a put with a Cox-Ross-Rubinstein binomial tree.

    The implementation uses memory-efficient backward induction, storing
    only one layer of option values. Runtime is O(steps^2) and memory is
    O(steps). Very coarse trees can have visible discretisation error, and
    extreme carry assumptions may require more steps to keep the CRR
    risk-neutral probability inside [0, 1].

    The default return value is a float. If return_exercise_boundary=True,
    the function returns BinomialTreeResult. European exercise boundaries
    are reported as all None because early exercise is disabled.
    """

    return _binomial_price(
        "put",
        spot,
        strike,
        maturity,
        rate,
        volatility,
        dividend_yield,
        steps,
        american,
        return_exercise_boundary,
    )
