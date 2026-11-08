from dataclasses import dataclass
from math import exp, log, sqrt
from random import Random
from statistics import NormalDist
from typing import Literal

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class MonteCarloResult:
    """Monte Carlo price estimate and sampling uncertainty."""

    price: float
    standard_error: float
    confidence_interval: tuple[float, float]
    simulations: int


def _validate_market_inputs(
    spot: float,
    strike: float,
    maturity: float,
    volatility: float,
) -> None:
    if spot <= 0.0:
        raise ValueError("spot must be positive")
    if strike <= 0.0:
        raise ValueError("strike must be positive")
    if maturity < 0.0:
        raise ValueError("maturity must be non-negative")
    if volatility < 0.0:
        raise ValueError("volatility must be non-negative")


def _validate_simulation_inputs(
    simulations: int,
    confidence_level: float,
) -> None:
    if (
        not isinstance(simulations, int)
        or isinstance(simulations, bool)
    ):
        raise ValueError("simulations must be an integer")
    if simulations <= 0:
        raise ValueError("simulations must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence level must be between 0 and 1")


def _payoff(
    option_type: OptionType,
    terminal_spot: float,
    strike: float,
) -> float:
    if option_type == "call":
        return max(terminal_spot - strike, 0.0)
    if option_type == "put":
        return max(strike - terminal_spot, 0.0)
    raise ValueError("option type must be 'call' or 'put'")


def simulate_gbm_terminal_prices(
    spot: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    *,
    simulations: int = 10_000,
    seed: int | None = None,
    antithetic: bool = False,
) -> tuple[float, ...]:
    """Simulate terminal prices under risk-neutral GBM."""

    if spot <= 0.0:
        raise ValueError("spot must be positive")
    if maturity < 0.0:
        raise ValueError("maturity must be non-negative")
    if volatility < 0.0:
        raise ValueError("volatility must be non-negative")
    _validate_simulation_inputs(simulations, 0.95)

    if maturity == 0.0:
        return tuple(spot for _ in range(simulations))

    if volatility == 0.0:
        terminal_spot = spot * exp((rate - dividend_yield) * maturity)
        return tuple(terminal_spot for _ in range(simulations))

    rng = Random(seed)
    drift = (
        rate
        - dividend_yield
        - 0.5 * volatility * volatility
    ) * maturity
    diffusion = volatility * sqrt(maturity)

    prices: list[float] = []

    if antithetic:
        pairs = simulations // 2
        for _ in range(pairs):
            shock = rng.gauss(0.0, 1.0)
            prices.append(spot * exp(drift + diffusion * shock))
            prices.append(spot * exp(drift - diffusion * shock))

        if simulations % 2:
            shock = rng.gauss(0.0, 1.0)
            prices.append(spot * exp(drift + diffusion * shock))

        return tuple(prices)

    for _ in range(simulations):
        shock = rng.gauss(0.0, 1.0)
        prices.append(spot * exp(drift + diffusion * shock))

    return tuple(prices)


def _monitoring_times(
    maturity: float,
    monitoring_dates: int | tuple[float, ...],
) -> tuple[float, ...]:
    if isinstance(monitoring_dates, int) and not isinstance(
        monitoring_dates,
        bool,
    ):
        if monitoring_dates <= 0:
            raise ValueError("monitoring dates must be positive")
        return tuple(
            maturity * step / monitoring_dates
            for step in range(1, monitoring_dates + 1)
        )

    if not isinstance(monitoring_dates, tuple):
        raise ValueError("monitoring dates must be an integer or tuple")
    if len(monitoring_dates) == 0:
        raise ValueError("monitoring dates must be non-empty")
    if any(time <= 0.0 for time in monitoring_dates):
        raise ValueError("monitoring dates must be positive")
    if any(time > maturity for time in monitoring_dates):
        raise ValueError("monitoring dates cannot exceed maturity")
    if any(
        earlier >= later
        for earlier, later in zip(
            monitoring_dates,
            monitoring_dates[1:],
        )
    ):
        raise ValueError("monitoring dates must be strictly increasing")
    return monitoring_dates


def _gbm_path_from_shocks(
    spot: float,
    rate: float,
    volatility: float,
    dividend_yield: float,
    monitoring_times: tuple[float, ...],
    shocks: tuple[float, ...],
) -> tuple[float, ...]:
    path = []
    previous_time = 0.0
    current_spot = spot

    for time, shock in zip(monitoring_times, shocks):
        dt = time - previous_time
        drift = (
            rate
            - dividend_yield
            - 0.5 * volatility * volatility
        ) * dt
        diffusion = volatility * sqrt(dt)
        current_spot *= exp(drift + diffusion * shock)
        path.append(current_spot)
        previous_time = time

    return tuple(path)


def simulate_gbm_paths(
    spot: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    *,
    monitoring_dates: int | tuple[float, ...] = 12,
    simulations: int = 10_000,
    seed: int | None = None,
    antithetic: bool = False,
) -> tuple[tuple[float, ...], ...]:
    """Simulate full GBM paths on discrete monitoring dates."""

    if spot <= 0.0:
        raise ValueError("spot must be positive")
    if maturity < 0.0:
        raise ValueError("maturity must be non-negative")
    if volatility < 0.0:
        raise ValueError("volatility must be non-negative")
    _validate_simulation_inputs(simulations, 0.95)

    times = _monitoring_times(maturity, monitoring_dates)

    if maturity == 0.0:
        return tuple(tuple(spot for _ in times) for _ in range(simulations))

    if volatility == 0.0:
        path = tuple(
            spot * exp((rate - dividend_yield) * time)
            for time in times
        )
        return tuple(path for _ in range(simulations))

    rng = Random(seed)
    paths: list[tuple[float, ...]] = []

    if antithetic:
        pairs = simulations // 2
        for _ in range(pairs):
            shocks = tuple(rng.gauss(0.0, 1.0) for _ in times)
            paths.append(
                _gbm_path_from_shocks(
                    spot,
                    rate,
                    volatility,
                    dividend_yield,
                    times,
                    shocks,
                )
            )
            paths.append(
                _gbm_path_from_shocks(
                    spot,
                    rate,
                    volatility,
                    dividend_yield,
                    times,
                    tuple(-shock for shock in shocks),
                )
            )

        if simulations % 2:
            shocks = tuple(rng.gauss(0.0, 1.0) for _ in times)
            paths.append(
                _gbm_path_from_shocks(
                    spot,
                    rate,
                    volatility,
                    dividend_yield,
                    times,
                    shocks,
                )
            )

        return tuple(paths)

    for _ in range(simulations):
        shocks = tuple(rng.gauss(0.0, 1.0) for _ in times)
        paths.append(
            _gbm_path_from_shocks(
                spot,
                rate,
                volatility,
                dividend_yield,
                times,
                shocks,
            )
        )

    return tuple(paths)


def _confidence_interval(
    price: float,
    standard_error: float,
    confidence_level: float,
) -> tuple[float, float]:
    tail_probability = (1.0 - confidence_level) / 2.0
    z_score = -NormalDist().inv_cdf(tail_probability)
    width = z_score * standard_error
    return price - width, price + width


def _standard_error(values: tuple[float, ...]) -> float:
    count = len(values)
    if count <= 1:
        return 0.0

    mean_value = sum(values) / count
    variance = sum(
        (value - mean_value) ** 2 for value in values
    ) / (count - 1)
    return sqrt(variance / count)


def _validate_sampling_error(
    maturity: float, volatility: float, simulations: int, antithetic: bool
) -> None:
    independent_observations = (simulations + 1) // 2 if antithetic else simulations
    if maturity > 0.0 and volatility > 0.0 and independent_observations < 2:
        raise ValueError("at least two independent observations are required to estimate sampling error")


def _estimator_observations(
    discounted_payoffs: tuple[float, ...],
    antithetic: bool,
) -> tuple[float, ...]:
    if not antithetic:
        return discounted_payoffs

    observations = []
    pairs = len(discounted_payoffs) // 2
    for index in range(pairs):
        first = discounted_payoffs[2 * index]
        second = discounted_payoffs[2 * index + 1]
        observations.append(0.5 * (first + second))

    if len(discounted_payoffs) % 2:
        observations.append(discounted_payoffs[-1])

    return tuple(observations)


def _apply_control_variate(
    discounted_payoffs: tuple[float, ...],
    discounted_terminal_spots: tuple[float, ...],
    expected_discounted_terminal_spot: float,
) -> tuple[float, ...]:
    payoff_mean = sum(discounted_payoffs) / len(discounted_payoffs)
    control_mean = (
        sum(discounted_terminal_spots)
        / len(discounted_terminal_spots)
    )
    control_variance = sum(
        (value - control_mean) ** 2
        for value in discounted_terminal_spots
    )

    if control_variance == 0.0:
        return discounted_payoffs

    covariance = sum(
        (payoff - payoff_mean) * (control - control_mean)
        for payoff, control in zip(
            discounted_payoffs,
            discounted_terminal_spots,
        )
    )
    beta = covariance / control_variance

    return tuple(
        payoff
        - beta
        * (control - expected_discounted_terminal_spot)
        for payoff, control in zip(
            discounted_payoffs,
            discounted_terminal_spots,
        )
    )


def _monte_carlo_price(
    option_type: OptionType,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float,
    simulations: int,
    seed: int | None,
    confidence_level: float,
    antithetic: bool,
    control_variate: bool,
) -> MonteCarloResult:
    _validate_market_inputs(spot, strike, maturity, volatility)
    _validate_simulation_inputs(simulations, confidence_level)
    _validate_sampling_error(maturity, volatility, simulations, antithetic)

    terminal_prices = simulate_gbm_terminal_prices(
        spot,
        maturity,
        rate,
        volatility,
        dividend_yield,
        simulations=simulations,
        seed=seed,
        antithetic=antithetic,
    )
    discount_factor = exp(-rate * maturity)
    discounted_payoffs = tuple(
        discount_factor * _payoff(option_type, terminal_spot, strike)
        for terminal_spot in terminal_prices
    )

    if control_variate:
        discounted_terminal_spots = tuple(
            discount_factor * terminal_spot
            for terminal_spot in terminal_prices
        )
        expected_discounted_terminal_spot = (
            spot * exp(-dividend_yield * maturity)
        )
        discounted_payoffs = _apply_control_variate(
            discounted_payoffs,
            discounted_terminal_spots,
            expected_discounted_terminal_spot,
        )

    estimator_observations = _estimator_observations(
        discounted_payoffs,
        antithetic,
    )
    price = sum(estimator_observations) / len(estimator_observations)
    standard_error = _standard_error(estimator_observations)

    return MonteCarloResult(
        price=price,
        standard_error=standard_error,
        confidence_interval=_confidence_interval(
            price,
            standard_error,
            confidence_level,
        ),
        simulations=simulations,
    )


def _geometric_average(values: tuple[float, ...]) -> float:
    return exp(sum(log(value) for value in values) / len(values))


def _geometric_asian_price(
    option_type: OptionType,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float,
    monitoring_times: tuple[float, ...],
) -> float:
    if maturity == 0.0 or volatility == 0.0:
        average_time = sum(monitoring_times) / len(monitoring_times)
        deterministic_geometric_average = spot * exp(
            (rate - dividend_yield) * average_time
        )
        payoff = _payoff(
            option_type,
            deterministic_geometric_average,
            strike,
        )
        return exp(-rate * maturity) * payoff

    count = len(monitoring_times)
    average_time = sum(monitoring_times) / count
    variance = volatility * volatility * sum(
        min(first, second)
        for first in monitoring_times
        for second in monitoring_times
    ) / (count * count)
    standard_deviation = sqrt(variance)
    mean_log = (
        log(spot)
        + (
            rate
            - dividend_yield
            - 0.5 * volatility * volatility
        )
        * average_time
    )
    forward_geometric_average = exp(mean_log + 0.5 * variance)
    d_1 = (mean_log - log(strike) + variance) / standard_deviation
    d_2 = d_1 - standard_deviation
    normal = NormalDist()
    discount_factor = exp(-rate * maturity)

    if option_type == "call":
        return discount_factor * (
            forward_geometric_average * normal.cdf(d_1)
            - strike * normal.cdf(d_2)
        )
    if option_type == "put":
        return discount_factor * (
            strike * normal.cdf(-d_2)
            - forward_geometric_average * normal.cdf(-d_1)
        )
    raise ValueError("option type must be 'call' or 'put'")


def _monte_carlo_asian_price(
    option_type: OptionType,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float,
    monitoring_dates: int | tuple[float, ...],
    simulations: int,
    seed: int | None,
    confidence_level: float,
    antithetic: bool,
    control_variate: bool,
) -> MonteCarloResult:
    _validate_market_inputs(spot, strike, maturity, volatility)
    _validate_simulation_inputs(simulations, confidence_level)
    _validate_sampling_error(maturity, volatility, simulations, antithetic)
    times = _monitoring_times(maturity, monitoring_dates)
    if maturity == 0.0:
        price = _payoff(option_type, spot, strike)
        return MonteCarloResult(price, 0.0, (price, price), simulations)

    paths = simulate_gbm_paths(
        spot,
        maturity,
        rate,
        volatility,
        dividend_yield,
        monitoring_dates=times,
        simulations=simulations,
        seed=seed,
        antithetic=antithetic,
    )
    discount_factor = exp(-rate * maturity)
    discounted_payoffs = tuple(
        discount_factor
        * _payoff(
            option_type,
            sum(path) / len(path),
            strike,
        )
        for path in paths
    )

    if control_variate:
        discounted_geometric_payoffs = tuple(
            discount_factor
            * _payoff(
                option_type,
                _geometric_average(path),
                strike,
            )
            for path in paths
        )
        expected_geometric_price = _geometric_asian_price(
            option_type,
            spot,
            strike,
            maturity,
            rate,
            volatility,
            dividend_yield,
            times,
        )
        discounted_payoffs = _apply_control_variate(
            discounted_payoffs,
            discounted_geometric_payoffs,
            expected_geometric_price,
        )

    estimator_observations = _estimator_observations(
        discounted_payoffs,
        antithetic,
    )
    price = sum(estimator_observations) / len(estimator_observations)
    standard_error = _standard_error(estimator_observations)

    return MonteCarloResult(
        price=price,
        standard_error=standard_error,
        confidence_interval=_confidence_interval(
            price,
            standard_error,
            confidence_level,
        ),
        simulations=simulations,
    )


def monte_carlo_call_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    *,
    simulations: int = 10_000,
    seed: int | None = None,
    confidence_level: float = 0.95,
    antithetic: bool = False,
    control_variate: bool = False,
) -> MonteCarloResult:
    """Estimate a European call price under Black-Scholes-Merton."""

    return _monte_carlo_price(
        "call",
        spot,
        strike,
        maturity,
        rate,
        volatility,
        dividend_yield,
        simulations,
        seed,
        confidence_level,
        antithetic,
        control_variate,
    )


def monte_carlo_put_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    *,
    simulations: int = 10_000,
    seed: int | None = None,
    confidence_level: float = 0.95,
    antithetic: bool = False,
    control_variate: bool = False,
) -> MonteCarloResult:
    """Estimate a European put price under Black-Scholes-Merton."""

    return _monte_carlo_price(
        "put",
        spot,
        strike,
        maturity,
        rate,
        volatility,
        dividend_yield,
        simulations,
        seed,
        confidence_level,
        antithetic,
        control_variate,
    )


def monte_carlo_asian_call_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    *,
    monitoring_dates: int | tuple[float, ...] = 12,
    simulations: int = 10_000,
    seed: int | None = None,
    confidence_level: float = 0.95,
    antithetic: bool = False,
    control_variate: bool = False,
) -> MonteCarloResult:
    """Estimate a discrete arithmetic-average Asian call price."""

    return _monte_carlo_asian_price(
        "call",
        spot,
        strike,
        maturity,
        rate,
        volatility,
        dividend_yield,
        monitoring_dates,
        simulations,
        seed,
        confidence_level,
        antithetic,
        control_variate,
    )


def monte_carlo_asian_put_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    *,
    monitoring_dates: int | tuple[float, ...] = 12,
    simulations: int = 10_000,
    seed: int | None = None,
    confidence_level: float = 0.95,
    antithetic: bool = False,
    control_variate: bool = False,
) -> MonteCarloResult:
    """Estimate a discrete arithmetic-average Asian put price."""

    return _monte_carlo_asian_price(
        "put",
        spot,
        strike,
        maturity,
        rate,
        volatility,
        dividend_yield,
        monitoring_dates,
        simulations,
        seed,
        confidence_level,
        antithetic,
        control_variate,
    )
