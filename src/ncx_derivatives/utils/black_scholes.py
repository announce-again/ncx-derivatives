from math import erf, erfc, exp, log, pi, sqrt


def norm_cdf(value: float) -> float:
    """Standard normal cumulative distribution function."""

    if value < -5.0:
        return 0.5 * erfc(-value / sqrt(2.0))
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def norm_pdf(value: float) -> float:
    """Standard normal probability density function."""

    return exp(-0.5 * value * value) / sqrt(2.0 * pi)


def d1(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """Calculate the Black-Scholes d1 term."""

    numerator = (
        log(spot / strike)
        + (
            rate
            - dividend_yield
            + 0.5 * volatility * volatility
        )
        * maturity
    )
    denominator = volatility * sqrt(maturity)

    return numerator / denominator


def d2(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """Calculate the Black-Scholes d2 term."""

    return d1(
        spot,
        strike,
        maturity,
        rate,
        volatility,
        dividend_yield,
    ) - volatility * sqrt(maturity)
