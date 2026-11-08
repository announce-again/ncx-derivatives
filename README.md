# NCX Derivatives

A from-scratch quantitative derivatives pricing, market-data, volatility, risk,
and market-making research platform.

## Current capabilities

### Pricing and Greeks

- Black-Scholes-Merton European call and put pricing
- Continuous dividend yield support
- Analytical Delta, Gamma, Vega, Theta, and Rho
- Implied volatility inversion with a hybrid Newton-bisection solver
- No-arbitrage bounds and input validation
- Finite-difference validation of analytical Greeks

### Monte Carlo and American pricing

- Cox-Ross-Rubinstein binomial tree pricing
- European and American option support
- Optional American exercise-boundary extraction
- Monte Carlo European option pricing
- Asian option Monte Carlo pricing
- Standard errors and confidence intervals
- Antithetic variates and control variates

### Reproducible option market-data pipeline

- Canonical option market-data models
- Provider-aware CSV ingestion and normalisation
- Rates, forwards, enrichment, and configurable cleaning
- Static-arbitrage diagnostics
- pandas interoperability
- Deterministic dataset serialisation with manifests and hashes

### Implied-volatility research data

- Chain-wide bid, midpoint, and ask implied volatility
- Per-side solver failures and diagnostic flags
- Vega, spot moneyness, forward moneyness, and log-forward-moneyness
- Deterministic records, DataFrame, and CSV exports
- Executable 50,000-row ingestion-to-IV pipeline
- Expiry-level OTM smile representation built directly from IV chains
- Configurable quote-side, quality, liquidity, duplicate, and ATM selection
- Machine-readable point, quote-selection, and group diagnostics
- Separate nearest-forward and observed-ATM semantics
- Total-variance ATM interpolation without single-sided extrapolation
- Local total-variance skew slope and curvature with partial-result diagnostics
- Signed-delta IV interpolation without extrapolation
- Call-minus-put risk reversals and symmetric delta butterflies
- Deterministic expiry term structures and aggregate Stage 3.2 exports

### Engineering

- Comprehensive automated test coverage with pytest

See [Stage 2 summary](docs/stage_2_summary.md) for the full market-data
infrastructure overview and the
[Stage 3.1 volatility pipeline](docs/volatility_pipeline.md) for the large
end-to-end workflow and benchmark, and
[Stage 3.2 smile selection](docs/volatility_smiles.md) for the research-smile
boundary and default policy, and
[Stage 3.2 smile metrics](docs/smile_metrics.md) for ATM interpolation and local
total-variance metrics, [delta smile metrics](docs/delta_smile_metrics.md) for
signed-delta/RR/BF conventions, and
[volatility term structures](docs/volatility_term_structure.md) for expiry-level
representation. The [Stage 3.2 completion summary](docs/stage_3_2_summary.md)
documents the 50,000-row conservation and separate throughput boundaries.
See [volatility surfaces and arbitrage diagnostics](docs/volatility_surface.md),
[SVI and SSVI calibration](docs/parametric_volatility.md), and the
[SPY volatility dynamics study](research/real_data/volatility_dynamics.md) for
Stages 3.3–3.6.

## Installation

```bash
python -m pip install -e .
```

Install the development or research dependencies from the repository root:

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[dev,research]"
```

## Repository layout

| Directory | Contents |
| --- | --- |
| [src/ncx_derivatives](src/ncx_derivatives) | Installable pricing, Greeks, market data, and volatility library |
| [examples](examples/README.md) | Runnable pricing and data workflows |
| [tests](tests) | Unit tests, pipeline smoke tests, and small provider fixtures |
| [docs](docs/roadmap.md) | Model conventions, pipeline documentation, and roadmap |
| [research](research/README.md) | Research scripts, methodology, and published outputs |
| [data](data) | Local raw inputs, intermediate checkpoints, and processed datasets |

## Quick start

```python
from ncx_derivatives.greeks import call_delta, gamma, vega
from ncx_derivatives.greeks import numerical_delta
from ncx_derivatives.monte_carlo import (
    monte_carlo_asian_call_price,
    monte_carlo_call_price,
)
from ncx_derivatives.pricing import binomial_put_price, call_price
from ncx_derivatives.volatility import call_implied_volatility

price = call_price(
    spot=100.0,
    strike=100.0,
    maturity=1.0,
    rate=0.05,
    volatility=0.20,
    dividend_yield=0.02,
)

delta = call_delta(
    spot=100.0,
    strike=100.0,
    maturity=1.0,
    rate=0.05,
    volatility=0.20,
    dividend_yield=0.02,
)

option_gamma = gamma(
    spot=100.0,
    strike=100.0,
    maturity=1.0,
    rate=0.05,
    volatility=0.20,
    dividend_yield=0.02,
)

option_vega = vega(
    spot=100.0,
    strike=100.0,
    maturity=1.0,
    rate=0.05,
    volatility=0.20,
    dividend_yield=0.02,
)

implied_volatility = call_implied_volatility(
    option_price=price,
    spot=100.0,
    strike=100.0,
    maturity=1.0,
    rate=0.05,
    dividend_yield=0.02,
)

american_put = binomial_put_price(
    spot=100.0,
    strike=100.0,
    maturity=1.0,
    rate=0.05,
    volatility=0.20,
    dividend_yield=0.02,
    steps=500,
    american=True,
)

monte_carlo_result = monte_carlo_call_price(
    spot=100.0,
    strike=100.0,
    maturity=1.0,
    rate=0.05,
    volatility=0.20,
    dividend_yield=0.02,
    simulations=50_000,
    seed=42,
    control_variate=True,
)

asian_result = monte_carlo_asian_call_price(
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

mc_delta = numerical_delta(
    monte_carlo_call_price,
    spot=100.0,
    strike=100.0,
    maturity=1.0,
    rate=0.05,
    volatility=0.20,
    dividend_yield=0.02,
    bump=0.10,
    simulations=50_000,
    seed=42,
)
```

## Roadmap

- [x] Stage 1.1: Black-Scholes pricing
- [x] Stage 1.2: Analytical Greeks
- [x] Stage 1.3: Implied volatility solver
- [x] Stage 1.4: Continuous dividend yield
- [x] Stage 1.5: Binomial tree pricing
- [x] Stage 1.6a: GBM terminal simulation
- [x] Stage 1.6b: European Monte Carlo pricing
- [x] Stage 1.6c: Monte Carlo standard errors and confidence intervals
- [x] Stage 1.6d: Monte Carlo variance reduction
- [x] Stage 1.7: Asian option Monte Carlo pricing
- [x] Stage 1.8: Numerical Greeks
- [x] Stage 2: Market data infrastructure
- [x] Stage 3: Volatility research
- [ ] Stage 4: Strategy research
- [ ] Stage 5: Portfolio risk
- [ ] Stage 6: Market-making simulation

## Testing

Run these commands from the repository root after installing the development
dependencies. The default suite uses checked-in fixtures and synthetic inputs.

```bash
python -m pytest
python -m pytest -m large tests/test_volatility_pipeline_large.py -vv
```

Run the static checks with `ruff check src tests research examples`.
