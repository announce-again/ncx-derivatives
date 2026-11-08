# Asian Option Monte Carlo Pricing

The Asian Monte Carlo module prices discrete arithmetic-average Asian calls
and puts under Black-Scholes-Merton GBM dynamics.

## Public API

```python
from ncx_derivatives.monte_carlo import (
    monte_carlo_asian_call_price,
    monte_carlo_asian_put_price,
)

result = monte_carlo_asian_call_price(
    spot=100.0,
    strike=100.0,
    maturity=1.0,
    rate=0.05,
    volatility=0.20,
    dividend_yield=0.02,
    monitoring_dates=12,
    simulations=50_000,
    seed=42,
    antithetic=True,
    control_variate=True,
)
```

The return type is `MonteCarloResult`, matching the European Monte Carlo API.
At zero maturity, an integer monitoring schedule collapses to the current spot
and the pricer returns the intrinsic payoff with zero sampling error.

## Capabilities

- Full GBM path simulation on discrete monitoring dates
- Arithmetic-average Asian call and put pricing
- Integer or explicit tuple monitoring schedules
- Reproducible random seeds
- Standard errors and confidence intervals
- Antithetic variates
- Geometric Asian closed-form control variate

## Validation Notes

- Low-volatility prices should approach the discounted deterministic average
  path payoff.
- Calls decrease with strike; puts increase with strike.
- Prices are non-negative.
- Standard errors should generally shrink at approximately `1 / sqrt(N)`.
- Antithetic variates and the geometric control variate should reduce variance
  in the standard at-the-money test cases.
- Arithmetic Asian calls are often below same-parameter European calls for
  common positive-rate, no-dividend cases, but this is not used as an
  unconditional theorem.
