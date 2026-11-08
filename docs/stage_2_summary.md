# Stage 2 Summary: Market Data Infrastructure

Stage 2 establishes the project's option market-data foundation. The result is
a reproducible pipeline that turns provider-style option-chain data into
canonical domain objects, validates and enriches them, applies configurable
cleaning, reports static-arbitrage issues, and writes deterministic dataset
snapshots.

```text
provider fixtures
    -> canonical models
    -> validation
    -> CSV ingestion
    -> normalisation
    -> rates, dividends, and forwards
    -> derived fields
    -> cleaning
    -> static-arbitrage diagnostics
    -> pandas boundary
    -> dataset serialisation
```

## Data Reconnaissance and Fixtures

Stage 2 began with provider-format reconnaissance before implementation. The
survey compared structurally different option-chain representations:

- Cboe-style flat interval tables
- Massive/Polygon-style nested option-chain snapshots
- Databento-style separated definitions, quotes, and statistics

The fixture set captures important market-data edge cases, including multiple
observation timestamps, calls and puts, multiple strikes and expirations,
missing bid or ask values, crossed and locked quotes, zero volume, missing
optional fields, pagination metadata, definition-to-quote joins, open-interest
reference semantics, and missing/null book prices.

## Canonical Domain Layer

The canonical model layer defines what option market data is, independently of
CSV, pandas, or provider-specific conventions. It includes:

- `OptionType` and `ExerciseStyle`
- `OptionContract`
- `OptionQuote` and `OptionTrade`
- `UnderlyingQuote`
- `OptionChainSnapshot`
- source metadata
- contract identity, hashing, deterministic sorting, and call-put pairing keys

The model boundary preserves provider identifiers while keeping canonical
contract identity separate from display symbols and raw source metadata.

## Validation

The validation framework reports structured issues instead of collapsing every
problem into a direct exception. It supports:

- `ValidationSeverity.ERROR`
- `ValidationSeverity.WARNING`
- `ValidationSeverity.INFO`
- structured issue codes, locations, and context
- field, record, and snapshot validation
- duplicate detection
- timestamp consistency checks
- validation reports suitable for later export

Constructor-level invariants remain in the domain models; validation focuses on
quality and cross-record checks that remain possible for valid canonical
objects.

## Ingestion and Normalisation

CSV ingestion reads external rows and produces canonical snapshots rather than
DataFrames. It supports explicit schema and column mappings, required and
optional columns, numeric/date/datetime parsing, missing-value interpretation,
row-level ingestion errors, raw-row preservation, and deterministic ordering.
Duplicate headers are reported as schema errors. Naive local timestamps that
are ambiguous or nonexistent at a clock transition produce row errors. An
explicit UTC offset identifies the intended instant.

Normalisation handles provider representation differences, including option
type values, exercise styles, UTC timestamps, expirations, symbols, exchanges,
numeric values, missing-value tokens, contract multipliers, and provider-specific
column conventions.

## Rates, Forwards, and Enrichment

Stage 2 introduced the carry assumptions required by downstream research:

- discount-factor curves
- zero-rate curves
- dividend-yield curves
- flat continuously compounded implementations
- forward-price estimation

Derived-field enrichment computes research fields without mutating the original
quotes. The enriched quote layer includes midpoint, spreads, time to maturity,
discount factors, forward price, spot and forward moneyness, log-moneyness,
intrinsic value, time value, and no-arbitrage bounds.

## Cleaning

Cleaning policies are configurable and never silently shrink a list. Each
rejected quote retains machine-readable diagnostics.

Supported policies include missing bid/ask, crossed markets, locked markets,
zero midpoint, maximum relative spread, minimum volume, minimum open interest,
stale quotes, maturity and strike ranges, spot and forward moneyness ranges, and
minimum option price.

## Static-Arbitrage Diagnostics

Static-arbitrage diagnostics detect, report, and quantify issues without
repairing prices. The first version supports European single-contract bounds,
call-put parity, same-expiry monotonicity, convexity, vertical-spread and
butterfly diagnostics, plus an explicitly heuristic calendar check that is
disabled by default.

Exercise-style applicability is handled conservatively. European-only
relationships are not applied to American or unknown-style options.

## pandas Boundary

pandas support is intentionally a boundary-layer feature. The domain layer does
not depend on pandas. Interoperability supports:

- snapshots to records and DataFrames
- records and DataFrames back to snapshots, including trade rows and exact
  nullable integer counts
- enriched quotes to records and DataFrames
- validation, cleaning, and static-arbitrage reports to records and DataFrames

## Dataset Serialisation

Stage 2 ends with deterministic dataset snapshots. The serialisation layer
supports canonical JSON, canonical CSV, accepted and rejected quote CSVs,
diagnostic JSON files, and a manifest with schema version, source information,
timestamps, configuration, assumptions, counts, hashes, and a deterministic
dataset ID.

The target layout is:

```text
dataset/
    raw/
        source.csv
    processed/
        canonical_option_chain.csv
        canonical_option_chain.json
        accepted_quotes.csv
    rejected/
        rejected_quotes.csv
    diagnostics/
        validation.json
        cleaning.json
        arbitrage.json
    manifest.json
```

Dataset identity is based on canonical logical content, including the canonical
snapshot, accepted and rejected partitions, diagnostics, raw input hash, and
semantic configuration. Audit-only metadata such as ingestion timestamp is
preserved in the manifest but excluded from the dataset ID.

Canonical JSON retains trades and the underlying quote of an empty quote
snapshot. Trade content and previously omitted metadata contribute to the
dataset ID. CSV output contains quote rows. Dataset rewrites stage files before
publication, withdraw the old manifest, and publish the replacement manifest
after all its files are in place.

## Version Meaning

The Stage 2 completion milestone corresponds to:

```text
v0.1.0  Pricing foundation
v0.2.0  Market data infrastructure
v0.3.0  Volatility research
```
