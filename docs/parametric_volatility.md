# Stage 3.5 parametric volatility models

The calibration API consumes Stage 3.2 `VolatilitySmile` objects. Install the optional optimizer with `python -m pip install -e ".[calibration]"`.

```python
from ncx_derivatives.volatility import (
    CalibrationWeighting,
    calibrate_ssvi,
    calibrate_svi,
    compare_parametric_holdouts,
)

svi = calibrate_svi(smiles[0], weighting=CalibrationWeighting.VEGA)
ssvi = calibrate_ssvi(smiles)
if ssvi.converged:
    atm_iv = ssvi.model.implied_volatility(k=0.0, maturity=30 / 365)
holdouts = compare_parametric_holdouts(smiles)
```

Supply one underlying, valuation timestamp, and IV quote side to surface calibration. Raw SVI requires five eligible strikes per expiry. SSVI requires at least two expiries and three eligible strikes per expiry. Each fit minimizes weighted squared total-variance errors with bounded nonlinear least squares and deterministic starts. The optimizer selects the lowest-cost converged fit when available. If every start fails to converge, it returns the lowest-cost attempt with `converged=False`. The result exposes the selected optimizer status, evaluation count across starts, omitted count, normalized weights, point residuals, weighted RMSE, and minimum sampled butterfly factor.

| Weighting | Relative weight in the total-variance objective |
| --- | --- |
| `UNIFORM` | Equal across eligible quotes |
| `INVERSE_SPREAD` | Inverse squared bid–ask IV spread, floored at `1e-4` IV by default |
| `VEGA` | Linear Black–Scholes Vega |

Zero-variance boundary points and quotes without a valid requested weight are omitted and counted. Raw SVI uses `w(k) = a + b (rho (k − m) + sqrt((k − m)² + sigma²))`. The fit constrains minimum total variance to be positive and both wing slopes below two. The density factor `g(k)` is evaluated at observed strikes and on a grid extending at least to `k = ±2`. A negative sampled value identifies butterfly arbitrage. A positive sampled minimum describes the grid, while inter-expiry calendar consistency is outside an individual raw-SVI fit. [Gatheral and Jacquier](https://arxiv.org/abs/1204.0646)

SSVI fits shared `rho` and `eta` and jointly estimates nondecreasing ATM total-variance nodes. The maturity query linearly interpolates those nodes within the observed expiry range. The shape uses `phi(theta) = eta / sqrt(theta (1 + theta))` with `eta (1 + abs(rho)) <= 2`. This is the sufficient static-arbitrage-free family described in the paper for the supported maturity range under the model's forward-moneyness convention. Strike queries use the parametric wings. `slice_parameters()` exposes the equivalent raw SVI slice for inspecting wing slopes and density factors. [Gatheral and Jacquier, equation 4.5](https://arxiv.org/pdf/1204.0646)

The holdout comparison removes one interior quote per expiry and fits all three models on the same eligible training quotes. It returns the observed, raw SVI, SSVI, and Stage 3.3 total variances at each withheld coordinate. At least six eligible strikes per expiry are required. Tests recover known raw SVI and SSVI curves from synthetic option quotes, then compare the three models on the same held-out quotes. Run the end-to-end example on a separate synthetic dataset with the standard CSV, IV, and smile pipeline.

```powershell
python examples/market_data/12_stage_3_5_parametric_volatility.py
```

The uniform-weight example converges on all five expiries. On its five withheld quotes, mean absolute total-variance errors are `2.08e-6` for the non-parametric surface, `1.02e-4` for raw SVI, and `3.72e-4` for SSVI. Three raw SVI slices have negative sampled butterfly factors. All three weighting modes run on the same synthetic CSV dataset. These results describe this particular quote generator and holdout rule.
