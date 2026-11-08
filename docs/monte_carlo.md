# Monte Carlo Pricing

The Monte Carlo module estimates European option prices under the
Black-Scholes-Merton risk-neutral GBM model.

Monte Carlo prices are estimates, not exact model values. Public pricing
functions therefore return `MonteCarloResult`:

```python
MonteCarloResult(
    price=...,
    standard_error=...,
    confidence_interval=...,
    simulations=...,
)
```

## Supported Methods

- Terminal GBM simulation
- European call and put pricing
- Arithmetic-average Asian call and put pricing
- Continuous dividend yield through drift `r - q`
- Reproducible seeds
- Standard error estimation
- Configurable confidence intervals
- Antithetic terminal simulation
- Discounted-underlying control variate
- Geometric Asian closed-form control variate

## Numerical Notes

- Runtime is `O(simulations)`.
- Memory is `O(simulations)` because terminal prices are currently retained as
  tuples for inspection and reproducibility.
- Confidence intervals describe Monte Carlo sampling error, not model risk.
- Antithetic variates and control variates reduce variance in many common
  cases, but no variance-reduction method should be assumed to improve every
  payoff under every parameter set.
- When antithetic variates are enabled, standard errors are estimated from
  pair averages. An odd final path is a separate observation.
- Stochastic pricing requires at least two independent observations to estimate
  the standard error. A single path is sufficient at expiry or at zero volatility.
- The current implementation uses only the Python standard library. A future
  NumPy backend can improve throughput without changing the public result
  object.
