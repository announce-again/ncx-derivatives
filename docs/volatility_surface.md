# Non-parametric volatility surfaces

`VolatilitySurface` accepts Stage 3.2 `VolatilitySmile` objects for one underlying, valuation timestamp, and IV side. Supply strictly positive, distinct maturities and distinct expirations. Build separate surfaces for bid, midpoint, and ask when those estimates are needed.

```python
from ncx_derivatives.volatility import VolatilitySurface, diagnose_surface_arbitrage

surface = VolatilitySurface(selection.smiles)
result = surface.query(log_forward_moneyness=0.0, maturity=30 / 365)
if result.failure_reason is None:
    print(result.implied_volatility, result.lower_maturity, result.upper_maturity)

held_out = surface.residuals(leave_one_out=True)
diagnostics = diagnose_surface_arbitrage(surface)
```

The strike coordinate is `k = log(K / F(T))` and maturity is a year fraction. Each smile is piecewise linear in `w = IV² × T` at fixed maturity. For a target between expiries, the surface evaluates both immediate neighbors at the same `k`, then interpolates their total variances linearly in `T`. The returned IV is `sqrt(w / T)`. An exact maturity uses only its own smile. An exact observed strike recovers its node. A single-strike slice supports only that strike coordinate. A query beyond the time range or outside either neighbor's strike support returns a failure reason and available maturity endpoints. An interior sparse slice can therefore limit coverage on both sides of its expiry.

`interpolate_total_variance` provides the time interpolation separately for inputs that already contain variance at a fixed coordinate. Research 001A uses this entry point with its published Stage 3.2 ATM nodes. See the [fixed-tenor follow-up](../research/real_data/research_001_atm_stability/constant_maturity_followup.md).

In-sample residuals are fitted minus observed total variance and are zero by construction. The leave-one-out version uses the withheld strike's two remaining neighbors within the same expiry. Endpoint holdouts have missing residuals because the reduced slice cannot bracket them. The synthetic tests verify exact recovery, interpolation on a ragged grid, sparse-slice coverage, and a nonzero held-out residual on a curved smile.

## Static arbitrage diagnostics

- Calendar checks compare expiry pairs at the union of strike knots within their shared `k` support. The difference in total variance is piecewise linear there, so checking all these knots captures decreases between the two slices. Comparing nonadjacent expiries also catches decreases across a gap in strike coverage. The result includes the number of comparisons and identifies pairs with no overlap.
- Strike checks use forward-normalized European call prices at `K/F = exp(k)`. Call-spread slopes should lie in `[-1, 0]` and slopes across consecutive intervals should be nondecreasing. The slope difference divided by the distance between interval centers estimates the normalized risk-neutral density. A negative estimate flags a sampled butterfly violation.
- The diagnostic evaluates the original strike grids separately, then adds interior strike and maturity samples. Each violation reports magnitude, coordinates, and whether it uses an interpolated value. Coverage includes the observed support, the number of sampled strikes, and the left and right boundary slopes. Strike checks are grid-based, and the boundary slopes describe the finite supported wings.

The default grid adds one sample per strike interval and one per maturity interval. Increase `strike_subdivisions` or `maturity_subdivisions` to inspect the support more finely. `tolerance` applies to total-variance decreases and normalized slope violations. The slope checks also allow for floating-point cancellation when strike intervals are narrow. Diagnostics leave surface values intact.
