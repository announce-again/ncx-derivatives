# Research workflows

- [Research 001 — ATM volatility stability](real_data/research_001_atm_stability/README.md) covers historical SPY and cross-underlying comparisons.
- [Stage 3.6 — SPY volatility dynamics](real_data/volatility_dynamics.md) covers horizon-matched realized variance, smile dynamics, regimes, and FOMC event windows.
- Smile-metric robustness uses synthetic quotes to measure errors against a known smile.

## Smile-metric robustness

`smile_metric_robustness.py` is a standalone, deterministic research harness.
It does not alter or duplicate the production metric implementations. Every
synthetic quote passes through the public Stage 3.1 CSV pipeline and the public
Stage 3.2 smile-selection, local-metric, and delta-metric functions.

The truth smile is quadratic in log-forward-moneyness:

```text
w(k) = level + slope * k + 0.5 * curvature * k^2
```

ATM is reported in annualized volatility. Skew and curvature are derivatives of
total variance. RR25 and BF25 are annualized-volatility structures evaluated at
the exact signed 25-delta coordinates of the continuous truth smile.

Run the default 900 experiments (100 replications for each noise/density pair):

```powershell
.\.venv\Scripts\python.exe -m research.smile_metric_robustness
```

Run a smaller custom grid:

```powershell
.\.venv\Scripts\python.exe -m research.smile_metric_robustness `
  --repetitions 20 `
  --noise 0,0.01,0.05 `
  --densities 9,17,33 `
  --output-dir .tmp/research/smile_metric_robustness
```

`--noise` is the Gaussian standard deviation in option-price currency units.
`--densities` is the number of evenly spaced strikes across the configured
log-forward-moneyness range; values must be odd so the grid includes ATM.

Outputs:

- `config.json`: exact configuration, continuous metric truth, and units.
- `stage_3_1_inputs/*.csv`: every exact input passed to Stage 3.1.
- `quote_records.csv`: quote-level truth, fair price, noise draw, and noisy quote.
- `experiment_records.csv`: every recovered value, error, failure reason, and
  pipeline/selection count for every replicate.
- `summary.csv`: bias, MAE, RMSE, success rate, and selected-point counts by
  price-noise level, strike density, and metric.

Bias, MAE, and RMSE are computed over successful recoveries. Success rate uses
all experiments in the condition as its denominator, so metric failures remain
visible rather than silently disappearing from the error distribution.

