# Stage 3.1 End-to-End Volatility Pipeline

The pipeline carries provider-style CSV records through the existing ingestion,
enrichment, cleaning, and implied-volatility-chain APIs. It deliberately stops at
Stage 3.1 and does not select smile points or calculate Stage 3.2 analytics.

## Data Flow

```text
deterministic synthetic CSV
-> CsvIngestionResult
-> EnrichedOptionQuote records
-> EnrichedCleaningResult
-> ImpliedVolatilityChain
-> deterministic IV-chain CSV
```

`VolatilityPipelineResult` retains the native result object from each stage:

- ingestion errors retain CSV row numbers and raw records;
- cleaning rejections retain the quote and all structured diagnostics;
- IV observations retain independent bid, midpoint, and ask status, failure
  reason, Vega, and diagnostic flags.

The stage counts are summaries over those objects, not replacements for their
diagnostics.

The chain reports Black–Scholes–Merton implied volatility for every exercise
style. Price inversion and the `UPPER_BOUND_IV` flag use the same BSM model
bounds. For American, Bermudan, or unspecified exercise styles, the result is
the volatility implied by a European pricing model.

## Synthetic Dataset

The default dataset contains 50,000 rows across 10 underlyings, 10 snapshots per
underlying, five expiries, 50 strikes, and both calls and puts. Prices are generated
deterministically from the existing Black-Scholes-Merton pricing functions with
symbol, maturity, and moneyness variation.

The generator injects deterministic isolated failures every 10,000 rows:

- one invalid strike for ingestion diagnostics;
- one missing bid and one crossed market for cleaning diagnostics;
- one quote above its no-arbitrage upper bound for Stage 3.1 diagnostics.

The CSV row order uses a deterministic permutation. No random market data or
network access is involved.

## Run

From the repository root:

```powershell
.\.venv\Scripts\python.exe examples\market_data\10_stage_3_1_volatility_pipeline.py --rows 50000 --output-dir .tmp\volatility_pipeline_benchmark
```

The command writes:

- `synthetic_option_quotes.csv`, the generated provider-style input;
- `implied_volatility_chain.csv`, one row per accepted quote with bid, midpoint,
  and ask IV diagnostics.

## Observed Benchmark

Observed on 2026-08-01 using Python 3.12.13 on the local Windows development
environment:

| Stage | Count | Seconds |
| --- | ---: | ---: |
| Synthetic generation | 50,000 rows | 1.703 |
| CSV ingestion | 50,000 raw / 49,995 successful | 3.264 |
| Enrichment | 49,995 quotes | 1.188 |
| Cleaning | 49,985 accepted / 10 rejected | 0.252 |
| Stage 3.1 IV | 149,955 side results | 5.135 |
| CSV export | 49,985 rows / 17,089,747 bytes | 1.727 |
| Pipeline core | 50,000 input rows | 9.839 |

Pipeline-core throughput was 5,082 input rows/second. Including deterministic
generation and CSV export, observed end-to-end throughput was approximately 3,768
rows/second. These figures are development-machine observations, not hard service
level guarantees.

Observed diagnostics were:

```text
Ingestion: 5 ROW_PARSE_ERROR rows
Cleaning: 5 MISSING_BID + 5 CROSSED_MARKET rows
Stage 3.1: 15 OUTSIDE_BOUNDS side failures across 5 quotes
Stage 3.1 successes: 149,940 side results
```

The observed output SHA-256 was
`b9a79dc889b12564fd851ea8a5be085d370f57255ee33b8249233a664064ed07`.

## Tests

The normal suite includes a deterministic 2,000-row integration test. The 50,000
row smoke/performance test is marked `large` and excluded from the default suite.

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pytest -m large tests\test_volatility_pipeline_large.py -vv
```
