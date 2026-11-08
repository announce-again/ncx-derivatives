# Stage 3.2 ATM and Local Smile Metrics

The metric layer consumes validated `VolatilitySmile` objects. It is implemented in
`volatility/smile_metrics.py` and does not read raw quotes, repeat IV inversion, or
modify Stage 3.2 selection policy.

## Boundary

```text
VolatilitySmile
-> ATM total-variance interpolation
-> local total-variance fit
-> independent ATM, skew, and curvature results
-> deterministic records / DataFrame
```

ATM, skew, and curvature have separate result objects because one metric can succeed
while another fails. `SmileMetricResult` retains the source smile and complete
`SmileMetricConfig`.

## Coordinates

All v1 metrics use log-forward-moneyness and total variance:

```text
k = log(K / F)
w(k, T) = implied_volatility(k, T)^2 * T
```

The reported local skew is `dw/dk` at `k=0`. The reported local curvature is
`d2w/dk2` at `k=0`. They are not implied-volatility derivatives.

## ATM Policy

If `observed_atm_point` exists and observed ATM use is enabled, the metric uses that
point directly with method `OBSERVED`.

Otherwise, the nearest points satisfying `k_left < 0 < k_right` bracket zero. ATM
total variance is interpolated linearly:

```text
weight = -k_left / (k_right - k_left)
w_atm = w_left + weight * (w_right - w_left)
atm_volatility = sqrt(w_atm / T)
```

Single-sided extrapolation is not allowed. A nearest-forward point is never used as
a substitute for ATM.

## Local Fit

The default fit selects up to two nearest observed points on each side of zero and an
observed near-zero point when present. It fits the unweighted quadratic:

```text
w(k) = intercept + linear_coefficient * k + quadratic_coefficient * k^2
```

Therefore:

```text
total_variance_skew_slope = linear_coefficient
total_variance_curvature = 2 * quadratic_coefficient
```

Coordinates are scaled before solving the least-squares normal equations. The source
points, fitted coefficients, fit window, weighting policy, and coordinate tolerances
are retained in results and exports.

With exactly two bracketing points, skew can succeed using `BRACKET_SECANT`, while
curvature fails with `INSUFFICIENT_POINTS`. Local fitting requires observations on
both sides of zero and never performs single-sided extrapolation.

## Failure Reasons

```text
EMPTY_SMILE
NON_POSITIVE_MATURITY
INSUFFICIENT_POINTS
ATM_NOT_BRACKETED
DEGENERATE_COORDINATES
NEGATIVE_TOTAL_VARIANCE
NON_FINITE_RESULT
LOCAL_FIT_FAILED
```

Every input smile produces one `SmileMetricResult`; failures remain in the output
with source-point context rather than being dropped.

## Export

`smile_metrics_to_records` and `smile_metrics_to_dataframe` produce one row per
smile. The stable schema contains:

- smile identity, maturity, and point count;
- ATM level, total variance, method, status, failure reason, and bracket points;
- total-variance slope and curvature with independent status and failure reason;
- local-fit source range and coefficients;
- interpolation, fit-window, weighting, and coordinate-tolerance policy.

## Run

```powershell
.\.venv\Scripts\python.exe examples\market_data\11_stage_3_2_volatility_smiles.py --rows 50000
```

The example writes `smile_metrics.csv` with the other Stage 3.2 analysis exports.
Pipeline, selection, local metric, delta metric, term assembly, and full
generation-to-export throughput are reported separately.

## Observed 50,000-Row Smoke

Observed on 2026-08-02 using the command above:

```text
500 input smiles
500 ATM successes
500 skew successes
500 curvature successes
6,274 local metric results / second
```

The metric rate is local-metric throughput measured over already constructed smiles.
It is not pipeline throughput and is not generation-to-export throughput. The value
is a development-machine observation rather than a service-level guarantee. See
`stage_3_2_summary.md` for the complete benchmark boundaries and export hashes.

## Downstream Stage 3.2 Work

Delta-target interpolation, risk reversals, and butterflies are implemented in
`volatility/delta_metrics.py`. Multi-expiry representation is implemented in
`volatility/term_structure.py`. Both consume these local results directly.
See [Stage 3.3 volatility surfaces](volatility_surface.md) for interpolation
across strikes and maturities.
