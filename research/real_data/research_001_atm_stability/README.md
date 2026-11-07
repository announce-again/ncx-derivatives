# Research 001 — ATM volatility stability

This package implements the reproducible boundaries for:

- Part A: SPY time-series and maturity stability, 2010–2023.
- Part B: SPY/QQQ/AAPL/NVDA/TSLA comparison, 2020–2022.

Raw Kaggle files are inputs, never outputs. Keep them under `data/raw/kaggle`,
freeze each directory with `dataset_manifest.json`, and write all derived files
under `data/interim`, `data/processed`, or this research package's `outputs`
directory.

## Implemented flow

```text
frozen OptionsDX/Kaggle wide files
  -> call/put long adapter
  -> non-destructive data audit
  -> daily Stage 2 + Stage 3.1 NCX reconstruction
  -> Stage 3.2 bid/mid/ask smiles
  -> expiry-level ATM panel
  -> nearest-tenor panel and daily stability labels
  -> summaries, regressions, plots, report, hashes
```

The [fixed-tenor follow-up](constant_maturity_followup.md) applies Stage 3.3
total-variance interpolation to the frozen Research 001A expiry panels. It
compares fixed-expiry, nearest-tenor, and constant-maturity changes on matching
SPY sessions under both historical carry specifications.

The canonical English Research 001A report is available at
`outputs/spy_2010_2023_carry_comparison/research_001A_full_report_en.md`.

## Freeze provenance

After downloading a Kaggle dataset into its final raw directory:

```powershell
.\.venv\Scripts\python.exe -B -m `
  research.real_data.research_001_atm_stability.download_manifest `
  data\raw\kaggle\spy_options_2010_2023 `
  --slug UPLOADER/DATASET `
  --title "SPY Options EOD Data (2010-2023)" `
  --uploader UPLOADER `
  --version VERSION `
  --claimed-source OptionsDX `
  --license LICENSE `
  --download-timestamp 2026-08-05T12:00:00+00:00 `
  --date-start 2010-01-01 `
  --date-end 2023-12-31
```

The version, uploader, license, source claim, and page/README snapshot must be
copied from the Kaggle page; the code does not guess missing provenance.

## Audit a pilot

```powershell
.\.venv\Scripts\python.exe -B -m `
  research.real_data.research_001_atm_stability.audit `
  --symbol SPY `
  --output-dir data\interim\research_001\spy_pilot `
  data\raw\kaggle\common_uploader\spy_2020_2022\*.parquet
```

The audit writes `standardized_contracts.csv`, `audit_summary.csv`,
`audit_by_date.csv`, `audit_by_expiry.csv`, `audit_failures.csv`,
`schema_report.json`, and `data_quality_report.md`. It does not drop or repair
rows.

## Carry policy

`run_daily_ncx_pipeline` requires an explicit `carry_for_date(date)` callback.
This prevents the pilot zero-rate/zero-dividend diagnostic from being confused
with the baseline rate/dividend enrichment. The callback and its source hashes
belong in `run_manifest.json`.

## Historical carry and full-history reconstruction

Research 001A now freezes and enriches the 2010–2023 SPY sample with:

- linearly interpolated DGS1MO/DGS3MO/DGS6MO/DGS1/DGS2 Treasury proxies and
  continuously compounded discount factors;
- State Street SPY distribution ex-dates with the latest cash amount known on
  each quote date (baseline);
- a trailing-365-day cash dividend yield alternative;
- flat-DGS3MO and available-period flat-SOFR diagnostics; and
- a put-call-parity option-implied forward diagnostic that is never used as a
  formal carry input.

Build carry inputs with `build_historical_carry`, then run
`run_spy_full_ncx` once for each formal specification:

```powershell
.\.venv\Scripts\python.exe -B -m `
  research.real_data.research_001_atm_stability.run_spy_full_ncx `
  --options-database data\interim\research_001\spy_full_vendor\spy_full_vendor.duckdb `
  --dataset-manifest data\raw\kaggle\spy_options_2010_2023\dataset_manifest.json `
  --rate-panel data\processed\research_001\carry\historical_rate_panel.parquet `
  --distribution-panel data\processed\research_001\carry\spy_distribution_history.parquet `
  --carry-panel data\processed\research_001\carry\carry_expiry_panel.parquet `
  --option-implied-panel data\processed\research_001\carry\option_implied_forward_diagnostic.parquet `
  --audit-source-dir research\real_data\research_001_atm_stability\outputs\spy_2010_2023_vendor_iv_replication `
  --interim-dir data\interim\research_001\spy_full_ncx_baseline `
  --output-dir research\real_data\research_001_atm_stability\outputs\spy_2010_2023_ncx_baseline `
  --specification treasury_projected_dividend_schedule `
  --workers 4
```

Daily expiry partitions are checkpoints. A rerun reuses completed dates and
records dates with no two-sided quotes in `daily_exclusions.csv`; it never
silently drops them. Use `treasury_trailing_dividend_yield` with separate
interim/output directories for the formal alternative.

Run `compare_carry_specifications` after vendor, baseline, and alternative
panels exist. It uses a common underlying-date-tenor sample and requires the
same finite previous observation for three-way daily-change comparisons. The
comparison output includes stability tables, paired differences, carry
coverage, numerical validation, a figure, source hashes, and output hashes.

Treasury yields remain financing proxies rather than option OIS curves. SOFR
is represented only as a flat available-period diagnostic, and option-implied
forwards remain diagnostic because SPY options are American.

Install optional research dependencies with:

```powershell
python -m pip install -e ".[research]"
```

Parquet equality is checked on sorted logical records. CSV is deterministic at
the byte level. Raw input hashes, config SHA-256, and output hashes are separate
artifacts.
