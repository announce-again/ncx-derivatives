# NCX Derivatives Roadmap

NCX Derivatives is a from-scratch quantitative derivatives research and market-making platform.

The project is designed to progress from mathematically transparent pricing models to a modular research, risk, and electronic market-making system. Each stage should produce a usable, tested component rather than a collection of disconnected notebooks.

---

## Engineering Principles

Development follows several core principles:

- Mathematical correctness before optimisation
- Explicit assumptions and unit conventions
- Stable, documented public APIs
- Analytical results validated against numerical methods
- Numerical methods validated through convergence and invariants
- Input validation and well-defined boundary behaviour
- Reproducible research workflows
- Automated testing for every production-facing component
- Separation between reusable library code and exploratory research
- Incremental commits organised around complete milestones

The standard development cycle is:

```text
Design
→ Implement
→ Test
→ Validate
→ Document
→ Commit
```

---

# Stage 0 — Engineering Foundation

## Objective

Establish a maintainable Python project suitable for long-term quantitative research and software development.

## Deliverables

- `src/`-based Python package structure
- Editable package installation
- `pyproject.toml` configuration
- Automated testing with `pytest`
- Consistent module boundaries
- Git and GitHub integration
- Project documentation
- Ignore rules for generated files and local environments

## Completion Criteria

- Package imports correctly after `pip install -e .`
- Tests run from the repository root
- Generated files are excluded from version control
- Repository naming and package naming are consistent
- Initial documentation accurately describes the project

## Status

**Complete**

---

# Stage 1 — Pricing and Volatility Foundation

## Objective

Build a reliable analytical and numerical foundation for derivatives pricing.

---

## Stage 1.1 — Black–Scholes Pricing

### Scope

Implement European call and put pricing under the no-dividend
Black–Scholes model.

### Capabilities

- European call pricing
- European put pricing
- Negative interest-rate support
- Expiry payoff handling
- Zero-volatility deterministic limits
- Input validation
- Static no-arbitrage bounds

### Validation

- Published benchmark prices
- No-dividend put-call parity
- Price monotonicity
- Volatility monotonicity
- Static-arbitrage bounds
- Boundary and invalid-input tests

### Status

**Complete**

---

## Stage 1.2 — Analytical Greeks

### Scope

Implement first- and second-order sensitivities for European options.

### Capabilities

- Call and put Delta
- Gamma
- Vega
- Call and put Theta
- Call and put Rho
- Black–Scholes analytical formulas
- Explicit unit conventions

### Validation

- Published benchmark values
- Central finite-difference Delta
- Second central-difference Gamma
- Central finite-difference Vega
- Maturity-difference Theta
- Central finite-difference Rho
- Structural Greek relationships

### Status

**Complete**

---

## Stage 1.3 — Implied Volatility Solver

### Scope

Invert Black–Scholes prices to recover implied volatility.

### Capabilities

- Call implied volatility
- Put implied volatility
- Hybrid Newton and bisection solver
- Bracket-preserving Newton steps
- Bisection fallback
- Low-Vega protection
- No-arbitrage price validation
- Lower-bound mapping to zero volatility
- Upper-bound mapping to infinite volatility
- Configurable convergence tolerance and iteration limits

Both public solvers accept keyword-only `price_tolerance`, `volatility_tolerance`,
`max_iterations`, and `max_volatility`. Non-finite inputs raise
`ImpliedVolatilityInputError`. An unbracketed price or an exhausted iteration
budget raises `ImpliedVolatilityConvergenceError`.

### Validation

- Price-to-volatility round trips
- Deep in-the-money options
- Deep out-of-the-money options
- Near-expiry options
- Negative interest rates
- Boundary prices
- Invalid market prices

### Status

**Complete**

---

## Stage 1.4 — Continuous Dividend Yield

### Scope

Extend the analytical pricing, Greeks, and implied-volatility stack from Black–Scholes to Black–Scholes–Merton with continuous dividend yield \(q\).

### Capabilities

- Discounted spot term \(S e^{-qT}\)
- Cost-of-carry term \(r-q\)
- Dividend-aware pricing
- Dividend-aware Greeks
- Dividend-aware implied-volatility inversion
- Backwards-compatible `dividend_yield=0.0` API

### Validation

- Black–Scholes–Merton put-call parity
- Dividend-aware finite-difference Greeks
- Dividend-aware implied-volatility round trips
- Zero-volatility forward payoff
- Dividend-aware no-arbitrage bounds

### Status

**Complete**

---

## Stage 1.5 — Binomial Tree Pricing

### Scope

Introduce the first general-purpose numerical pricing method using the Cox–Ross–Rubinstein tree.

### Capabilities

- European call and put pricing
- American call and put pricing
- Continuous dividend yield
- Early-exercise detection
- Configurable tree depth
- Memory-efficient backward induction
- Optional exercise-boundary extraction

### Validation

- European tree convergence to Black–Scholes–Merton
- American option value greater than or equal to European value
- Non-dividend-paying American call equivalence
- Intrinsic-value lower bounds
- Early-exercise behaviour for American puts
- Stability across moneyness and maturity
- Convergence behaviour across tree depths

### Completion Criteria

- European prices converge within documented tolerances
- American exercise logic is independently tested
- Complexity and numerical limitations are documented
- Public API remains consistent with the analytical pricing layer

### Status

**Complete**

---

## Stage 1.6 — Monte Carlo Pricing

### Scope

Develop a reusable simulation framework for pricing and sensitivity estimation.

Monte Carlo prices are statistical estimates rather than exact values. The
first public API should return a result object instead of a bare float:

```python
MonteCarloResult(
    price=...,
    standard_error=...,
    confidence_interval=...,
    simulations=...,
)
```

### Status

**Complete**

---

## Stage 1.6a — GBM Terminal Simulation

### Scope

Implement reproducible terminal-price simulation under risk-neutral geometric
Brownian motion.

### Capabilities

- Terminal stock-price simulation
- Continuous dividend yield through drift \(r-q\)
- Reproducible random seeds
- Batch terminal simulation
- Input validation

### Validation

- Simulated terminal mean close to risk-neutral forward
- Simulated log-return variance close to \(\sigma^2 T\)
- Reproducibility tests with fixed seeds
- Stability across moneyness, maturities, rates, and dividend yields

### Status

**Complete**

---

## Stage 1.6b — European Monte Carlo Pricing

### Scope

Price European call and put options from simulated terminal payoffs.

### Capabilities

- European call pricing
- European put pricing
- Discounted payoff estimation
- `MonteCarloResult` return object
- Configurable simulation count
- Batch-ready pricing structure

### Validation

- Convergence to Black–Scholes–Merton prices
- Comparison against binomial tree prices
- Intrinsic-value and non-negativity checks
- Reproducibility tests

### Status

**Complete**

---

## Stage 1.6c — Standard Error and Confidence Intervals

### Scope

Expose uncertainty estimates for Monte Carlo prices.

### Capabilities

- Standard-error estimation
- Configurable confidence level
- Confidence intervals
- Simulation count reporting
- Payoff variance diagnostics

### Validation

- Confidence-interval coverage checks
- Standard error decreases at approximately \(1/\sqrt{N}\)
- Deterministic zero-volatility behaviour
- Invalid confidence-level handling

### Status

**Complete**

---

## Stage 1.6d — Variance Reduction

### Scope

Add variance-reduction methods for European option Monte Carlo pricing.

### Capabilities

- Antithetic variates
- Control variates
- Black–Scholes–Merton control for European options
- Error-reduction diagnostics
- Variance-reduction configuration

### Validation

- Antithetic estimator reduces or maintains variance
- Control variate improves convergence to analytical prices
- Reproducibility with variance-reduction enabled
- Comparison against plain Monte Carlo at equal random draw budgets

### Status

**Complete**

---

## Stage 1.7 — Asian Option Monte Carlo Pricing

### Scope

Use Monte Carlo path simulation to price options without direct vanilla
closed-form coverage.

### Capabilities

- Arithmetic-average Asian call pricing
- Arithmetic-average Asian put pricing
- Discrete monitoring dates
- Full GBM path simulation
- Reproducible random seeds
- `MonteCarloResult` return object
- Standard errors and confidence intervals
- Antithetic variates
- Geometric Asian closed-form control variate

### Validation

- Low-volatility convergence to deterministic average-path payoff
- Strike monotonicity
- Non-negative call and put prices
- Standard-error reduction with larger simulation counts
- Antithetic variance reduction in standard test cases
- Geometric-control-variate variance reduction
- Carefully scoped comparison against same-parameter European calls

### Status

**Complete**

---

## Stage 1.8 — Numerical Greeks

### Scope

Provide model-independent Greek estimation for numerical pricers.

### Capabilities

- Bump-and-revalue Delta
- Gamma
- Vega
- Theta
- Rho
- Adaptive bump sizing
- Forward, backward, and central differences
- Common-random-number Monte Carlo Greeks
- Error diagnostics

### Validation

- Comparison against analytical Black–Scholes–Merton Greeks
- Sensitivity to bump size
- Numerical stability across moneyness and maturity
- Monte Carlo estimator variance analysis

### Status

**Complete**

---

# Stage 2 — Market Data Infrastructure

## Objective

Create a clean and reproducible pipeline for transforming raw option-chain data into canonical, validated, research-ready inputs.

---

## Stage 2.0 — Data Source Reconnaissance and Fixtures

### Scope

Collect and study a small set of real option-chain samples before committing
to the canonical schema or ingestion APIs.

### Planned Capabilities

- Identify 2-4 option-chain data formats
- Compare fields, missing values, and abnormal quote cases
- Save deterministic fixtures for tests and design validation
- Include at least one normal complete chain
- Include data with missing bid or ask values
- Include data with crossed quotes, zero volume, or similar quality problems
- Include a complete chain with multiple expiries, calls, and puts
- Include at least one provider format for cross-source comparison
- Infer the canonical schema from observed source differences
- Avoid any dependency on live APIs at this stage

### Fields to Investigate

- Underlying symbol
- Option symbol
- Expiration
- Strike
- Call or put type
- Bid
- Ask
- Last
- Volume
- Open interest
- Implied volatility
- Quote timestamp
- Trade timestamp
- Underlying spot
- Exchange
- Contract multiplier
- Exercise style
- Currency

### Source Differences to Record

- Whether expiration is represented as a date or datetime
- Whether timestamps include timezone information
- Whether implied volatility is represented as `0.25` or `25`
- Whether option type is represented as `C`/`P` or `call`/`put`
- Whether missing values use empty strings, `null`, `0`, or `NaN`
- Whether zero bid or ask means a real quote or missing data
- Whether null order-book prices use provider-specific sentinel values
- Whether spot is repeated on each row or supplied separately
- Whether contract symbols can be parsed reliably
- Whether volume is interval, session, previous-session, or otherwise
  aggregated
- Whether exchange-like fields mean listing exchange, quote venue, trade venue,
  publisher, or dataset

### Status Progression

Stage 2.0 should not jump directly from `Planned` to `Completed`. It should
move through explicit reconnaissance checkpoints:

```text
Planned
 ↓
In Progress
 ↓
Fixtures Collected
 ↓
Schema Requirements Finalised
 ↓
Completed
```

### Status

**Complete**

Current checkpoints:

- Source Selection Complete
- Field Survey Complete
- Fixtures Collected
- Schema Requirements Finalised
- Fixture Integrity Checks Passing

---

## Stage 2.0a — Source Selection and Field Survey

### Scope

Select representative option-chain source formats and document their field
structure before writing formal ingestion code.

### Planned Capabilities

- Select 2-4 materially different option-chain formats
- Prefer structural diversity over provider count alone
- Include at least one flat interval format with one row per contract per
  observation time
- Include at least one nested or separated format, where available
- Compare field names, types, units, and missing-value conventions
- Record timestamp, timezone, symbol, IV, and bid-ask conventions
- Identify which fields are canonical, optional canonical, source metadata, derived, or ignored
- Capture unresolved schema decisions for Stage 2.1
- Avoid production ingestion APIs, live downloads, credentials, and provider SDK dependencies

### Deliverables

- Source selection notes
- Source-to-canonical field survey table
- Missing-value and unit convention notes
- Initial list of fixture candidates
- Open schema questions for Stage 2.1

### Status

**Complete**

---

## Stage 2.0b — Fixture Construction

### Scope

Construct a small deterministic fixture set from the selected source formats
before implementing canonical models or ingestion code.

### Planned Capabilities

- Cboe-style interval CSV fixtures
- Massive/Polygon-style nested snapshot JSON fixtures
- Databento-style separated definitions, BBO, and statistics CSV fixtures
- Normal complete chain examples
- Synthetic quality-case examples
- Missing optional nested-field examples
- Pagination metadata examples
- Separated-schema join examples using `instrument_id`
- Databento-style null-price sentinel and converted-`NaN` examples
- Explicit labels distinguishing synthetic mutations from observed source
  structure

### Target Fixture Layout

```text
tests/fixtures/market_data/
    cboe_intervals/
        normal.csv
        synthetic_quality_cases.csv
    massive_snapshot/
        normal.json
        missing_optional_fields.json
    databento_separated/
        definitions.csv
        bbo.csv
        statistics.csv
```

### Status

**Complete**

---

## Stage 2.0c — Fixture Provenance

### Scope

Document source inspiration, reconstruction choices, synthetic mutations, and
licensing constraints for every fixture.

### Planned Capabilities

- Fixture provenance notes
- Source documentation references
- Synthetic mutation labels
- Redistribution and licensing notes
- Distinction between observed source structure and fabricated market values

### Status

**Complete**

---

## Stage 2.0d — Canonical Schema Requirements

### Scope

Convert the source survey and fixture observations into explicit requirements
for Stage 2.1 canonical market-data models.

### Planned Capabilities

- Required canonical fields
- Optional canonical fields
- Source metadata fields
- Derived fields to exclude from raw canonical models
- Volume aggregation semantics
- Open-interest reference-date semantics
- Exchange and venue semantics
- Contract identity requirements
- Remaining unresolved design decisions for Stage 2.1

### Status

**Complete**

---

## Stage 2.0e — Fixture Integrity Checks

### Scope

Protect the Stage 2.0b fixtures from accidental structural drift without
implementing production ingestion.

### Planned Capabilities

- CSV header and row-shape checks
- JSON structural checks
- Cboe multi-observation-time checks
- Massive optional-field and pagination checks
- Databento `instrument_id` join checks
- Databento null-price sentinel checks
- Open-interest reference-date checks

### Status

**Complete**

---

## Stage 2.1 — Canonical Market Data Models

### Scope

Establish the shared domain language for option-market data. This stage
defines what the data is; it does not handle CSV parsing, cleaning, or pandas.

### Planned Capabilities

- `OptionType`
- `ExerciseStyle`
- `OptionContract`
- `OptionQuote`
- `OptionTrade`
- Underlying quote representation
- Option-chain snapshot representation
- Timestamp, exchange, and data-source metadata
- Contract identity, deterministic sorting, and hashing
- Canonical keys needed for call-put pairing
- Public exports and focused model tests

### Status

**Complete**

---

## Stage 2.1a — Enums and Shared Primitives

### Scope

Define the shared primitive language used by canonical market-data models.

### Capabilities

- `OptionType`
- `ExerciseStyle`
- `SourceMetadata`
- `ContractPairingKey`
- Timezone-aware timestamp validation
- Required and optional text validation
- Finite numeric validation
- Non-negative integer validation for sizes and counts

### Status

**Complete**

---

## Stage 2.1b — OptionContract

### Scope

Represent option contract identity without relying solely on provider symbol
parsing.

### Capabilities

- Explicit underlying, expiration, strike, and option type
- Optional exercise style
- Optional contract multiplier
- Optional currency
- Optional source contract identifier
- Optional display symbol
- Optional listing exchange
- Contract hashing
- Deterministic contract sorting
- Stable call-put pairing key

### Status

**Complete**

---

## Stage 2.1c — OptionQuote and OptionTrade

### Scope

Represent quote and trade observations while keeping quote time, trade time,
quote venues, and trade venue separate.

### Capabilities

- Independently missing bid and ask
- Structurally representable zero bid and ask
- Optional bid and ask sizes
- Optional bid and ask venues
- Optional session volume with explicit session semantics
- Optional open interest with separate reference date
- `OptionTrade` with trade price, trade size, trade timestamp, and trade venue
- No cleaning, midpoint calculation, provider-IV authority, or bar aggregation

### Status

**Complete**

---

## Stage 2.1d — UnderlyingQuote

### Scope

Represent underlying-market observations without requiring synchronized spot
data for every option quote.

### Capabilities

- Underlying symbol
- Timezone-aware quote timestamp
- Optional spot price
- Optional bid and ask
- Optional bid and ask venues
- Optional source metadata

### Status

**Complete**

---

## Stage 2.1e — OptionChainSnapshot

### Scope

Represent a logical option-chain snapshot independently of any source file,
API page, or provider-specific ingestion format.

### Capabilities

- Snapshot underlying symbol
- Timezone-aware snapshot timestamp
- Optional underlying quote
- Tuple-based immutable quote and trade collections
- Deterministic quote and trade ordering
- Contract collection derived from quote and trade observations
- Detection of obvious underlying-symbol inconsistency

### Status

**Complete**

---

## Stage 2.1f — Identity, Ordering, Hashing, and Pairing Keys

### Scope

Make contract identity stable enough for deterministic processing and call-put
pairing.

### Capabilities

- Frozen model objects
- Hashable contracts
- Source identifier separated from display symbol
- Deterministic `sort_key` values
- Stable `ContractPairingKey` independent of call or put side

### Status

**Complete**

---

## Stage 2.1g — Public Exports and Model Tests

### Scope

Expose canonical models through the public market-data package and verify the
Stage 2.1 model boundary.

### Capabilities

- `ncx_derivatives.market_data` public exports
- Immutability tests
- Hashing tests
- Timezone-aware timestamp tests
- Missing bid/ask tests
- Zero-price representation tests
- Open-interest reference-date tests
- Snapshot consistency tests
- Full Stage 1 and fixture test compatibility

### Status

**Complete**

---

## Stage 2.2 — Validation Framework

### Scope

Build structured, reusable validation that separates fatal structural errors
from data-quality warnings.

Model invariants are enforced by Stage 2.1 constructors. Stage 2.2 validates
data-quality and cross-record issues that can still exist in valid canonical
objects.

### Planned Capabilities

- Field-level validation
- Record-level validation
- Snapshot-level validation
- `ValidationSeverity.ERROR`
- `ValidationSeverity.WARNING`
- `ValidationSeverity.INFO`
- Structured validation issues
- Validation reports
- Duplicate detection
- Inconsistent underlying detection
- Timestamp consistency checks
- Avoid representing every issue as a direct `ValueError`
- Future-relative-to-snapshot timestamp checks
- One quote per canonical contract per snapshot
- Built-in validation code constants

### Status

**Complete**

---

## Stage 2.2a — Validation Severity and Issues

### Scope

Define structured validation primitives that callers can inspect without
catching exceptions.

### Capabilities

- `ValidationSeverity.ERROR`
- `ValidationSeverity.WARNING`
- `ValidationSeverity.INFO`
- `ValidationIssue`
- `BuiltinValidationCode`
- Machine-readable issue codes
- Location paths
- Optional structured context
- Stable `ValueError` for invalid validation primitive inputs

### Status

**Complete**

---

## Stage 2.2b — Validation Reports

### Scope

Aggregate validation issues into reusable reports.

### Capabilities

- `ValidationReport`
- Severity-filtered issue access
- `has_errors`
- `has_warnings`
- `is_valid`
- `by_code`
- `at_location_prefix`
- Report combination
- Location-prefixing for nested validation

### Status

**Complete**

---

## Stage 2.2c — Record-Level Validation

### Scope

Validate canonical contracts, quotes, trades, and underlying observations
without mutating or cleaning them.

### Capabilities

- Contract optional-identity field diagnostics
- Missing bid and ask diagnostics
- Empty market diagnostics
- Crossed and locked quote diagnostics
- Zero bid and ask informational issues
- Open-interest reference-date diagnostics
- Trade-size diagnostics
- Underlying bid-ask diagnostics

### Status

**Complete**

---

## Stage 2.2d — Snapshot-Level Validation

### Scope

Validate a logical option-chain snapshot as a collection of canonical records.

### Capabilities

- Empty snapshot diagnostics
- Missing underlying quote diagnostics
- Duplicate quote detection by contract within a snapshot
- Quote timestamp after snapshot checks
- Underlying timestamp after snapshot checks
- Trade timestamp consistency checks
- Nested issue location paths

### Status

**Complete**

---

## Stage 2.3 — CSV Ingestion

### Scope

Convert external CSV files into canonical market-data snapshots.

```text
CSV
 ↓
raw records
 ↓
parsed canonical objects
```

### Planned Capabilities

- CSV schema definition
- Column mapping
- Required and optional columns
- Numeric parsing
- Date and datetime parsing
- Missing-value interpretation
- Row-level ingestion errors
- Raw-row preservation
- Deterministic ordering
- Canonical snapshot output rather than DataFrame output

### Status

**Complete**

---

## Stage 2.3a — CSV Schema and Column Mapping

### Scope

Define an explicit CSV boundary schema without hard-coding provider-specific
adapters.

### Capabilities

- `CsvColumnMapping`
- Required option quote columns
- Optional contract columns
- Optional quote-size and volume columns
- Optional open-interest columns
- Optional underlying quote columns
- Explicit value maps for source-specific enum representations
- Header validation for every configured mapped column
- Exact value mapping only; case folding and provider normalization remain in Stage 2.4

### Status

**Complete**

---

## Stage 2.3b — Raw Records and Row-Level Errors

### Scope

Preserve raw CSV rows and report row-level ingestion failures without stopping
the whole file.

### Capabilities

- `CsvRawRecord`
- `CsvIngestionError`
- Row number preservation
- Column-level parse error reporting
- Missing required column reporting
- Missing mapped column reporting
- Continue parsing valid rows after row-level failures
- Rows with row-level ingestion errors are excluded from canonical output

### Status

**Complete**

---

## Stage 2.3c — Typed Parsing

### Scope

Parse CSV text into canonical model primitives.

### Capabilities

- Numeric parsing
- Integer parsing
- ISO date parsing
- ISO datetime parsing
- Timezone assumption for naive datetimes
- General `tzinfo` support for assumed timezone
- Explicit behavior when no snapshot timestamp column is mapped: quote timestamp
  is used as the grouping timestamp
- Missing-value interpretation
- Option type parsing
- Exercise style parsing

### Status

**Complete**

---

## Stage 2.3d — Snapshot Construction

### Scope

Construct canonical option-chain snapshots rather than DataFrames.

### Capabilities

- `ingest_option_chain_csv`
- `ingest_option_chain_csv_file`
- `CsvIngestionResult`
- Built-in CSV ingestion error codes
- Immutable config mappings and raw records
- Successful and failed row counts
- Schema error access
- Group rows into logical snapshots by underlying and snapshot timestamp
- Build `OptionContract`
- Build `OptionQuote`
- Build optional `UnderlyingQuote`
- Deterministic snapshot output ordering

### Status

**Complete**

---

## Stage 2.4 — Normalisation

### Scope

Handle semantic representation differences across data sources. CSV ingestion
reads the file; normalisation unifies the meaning.

Stage 2.4 provides reusable source-value normalizers and provider convention
configurations used before canonical object construction. It does not
post-process canonical snapshots.

### Planned Capabilities

- Convert `"C"`, `"CALL"`, and `"call"` to `OptionType.CALL`
- Convert `"P"`, `"PUT"`, and `"put"` to `OptionType.PUT`
- Normalize timestamps to UTC
- Normalize expirations to dates
- Convert numeric strings to numeric values
- Convert empty strings, `"NA"`, and `"null"` to `None`
- Normalize exchange and symbol values
- Normalize contract multipliers
- Deterministic sorting
- Provider-specific column conventions
- Source-value normalisers shared by CSV ingestion
- Provider convention helpers separated from scalar normalisation

### Status

**Complete**

---

## Stage 2.4a — Value Normalisation

### Scope

Normalize source scalar representations into canonical primitive values.

### Capabilities

- Missing-value interpretation
- Numeric string parsing
- Integer string parsing
- Contract multiplier normalization
- Boolean values rejected as numeric inputs
- Strict integer parsing without truncating decimal strings
- Defensive value-map helpers

### Status

**Complete**

---

## Stage 2.4b — Enum and Text Normalisation

### Scope

Normalize common option-market text representations.

### Capabilities

- `"C"`, `"CALL"`, and `"call"` to `OptionType.CALL`
- `"P"`, `"PUT"`, and `"put"` to `OptionType.PUT`
- Trim and case-fold before enum alias lookup
- Standard exercise-style representations
- Symbol normalization
- Exchange normalization
- Symbol and exchange normalization are syntactic trim-and-uppercase only, not
  identity resolution

### Status

**Complete**

---

## Stage 2.4c — Date and Timestamp Normalisation

### Scope

Normalize source date and timestamp representations.

### Capabilities

- ISO date parsing
- Datetime inputs rejected by date normalisation unless handled explicitly by a
  datetime normaliser
- Expiration date normalization without silent datetime truncation
- ISO datetime parsing
- `Z` suffix handling
- Naive datetime timezone assumption
- UTC timestamp output

### Status

**Complete**

---

## Stage 2.4d — CSV Convention Helpers

### Scope

Use normalisation helpers to define explicit provider-style CSV conventions
without moving parsing logic into provider adapters.

### Capabilities

- Standard option-type value map
- Standard exercise-style value map
- Cboe-style option interval CSV config helper
- Optional open-interest and calculated-underlying field variants
- Integration with Stage 2.3 CSV ingestion

### Status

**Complete**

---

## Stage 2.5 — Rates, Dividends, and Forwards

### Scope

Introduce rate, dividend, and forward-price assumptions before derived fields,
research cleaning, and arbitrage diagnostics depend on them.

### Planned Capabilities

- `DiscountFactorCurve`
- `ZeroRateCurve`
- `DividendYieldCurve`
- Flat-rate implementations
- Interpolation policy
- Discount-factor lookup
- Forward-price estimation
- Carry representation
- Initial support for flat continuously compounded curves, with term structures left as an extension

### Core Relationships

\[
D_r(T)=e^{-rT}
\]

\[
D_q(T)=e^{-qT}
\]

\[
F(T)=S\frac{D_q(T)}{D_r(T)}
\]

For flat continuously compounded rates:

\[
F(T)=Se^{(r-q)T}
\]

### Status

**Complete**

---

## Stage 2.5a — Curve Interfaces and Policies

### Scope

Define the minimal curve language required by later enrichment and diagnostics.

### Capabilities

- `DiscountFactorCurve`
- `ZeroRateCurve`
- `DividendYieldCurve`
- `InterpolationPolicy.FLAT`
- Discount-factor lookup interface

### Status

**Complete**

---

## Stage 2.5b — Flat Curve Implementations

### Scope

Support initial flat continuously compounded assumptions before term
structures are introduced.

### Capabilities

- `FlatZeroRateCurve`
- `FlatDividendYieldCurve`
- Negative zero-rate support
- Input validation for maturities and finite values
- Discount factors satisfy \(D(0)=1\)
- Stage 2.5 only records the flat policy; interpolation across curve nodes is
  not yet implemented

### Curve Semantics

`DiscountFactorCurve` is a generic positive maturity-factor interface. For a
`DividendYieldCurve`, `discount_factor(T)` means \(e^{-qT}\), not a risk-free
present-value discount factor.

### Status

**Complete**

---

## Stage 2.5c — Carry and Forward Price Helpers

### Scope

Compute forward prices from spot, risk-free discount factors, and dividend
discount factors.

### Capabilities

- `CarryAssumptions`
- `discount_factor_from_zero_rate`
- `forward_price`
- `flat_forward_price`
- Relationship \(F(T)=S D_q(T) / D_r(T)\)
- Flat continuously compounded relationship \(F(T)=Se^{(r-q)T}\)

### Status

**Complete**

---

## Stage 2.6 — Derived Fields and Year Fractions

### Scope

Compute research fields from canonical quotes without mutating the original
quote objects.

### Planned Capabilities

- `ACT/365F` year fractions
- `ACT/360` year fractions
- Extensible day-count interface
- Midpoint
- Absolute spread
- Relative spread
- Time to maturity
- Discount factor
- Forward price
- Spot moneyness
- Forward moneyness
- Log-moneyness
- Intrinsic value
- Time value
- No-arbitrage lower and upper bounds
- Independent enriched quote object, for example `EnrichedOptionQuote`

### Implemented Capabilities

- `DayCountConvention`
- `DayCount`
- `ActualFixedDayCount`
- `year_fraction`
- `midpoint`
- `absolute_spread`
- `relative_spread`
- `intrinsic_value`
- `NoArbitrageBounds`
- `no_arbitrage_bounds`
- `european_no_arbitrage_bounds`
- `EnrichedOptionQuote`
- `enrich_option_quote`
- `enrich_option_chain_snapshot`

### Boundary Notes

Stage 2.6 computes derived research fields from valid canonical objects. It
does not reject quotes for quality reasons and does not emit diagnostics.
Crossed quotes can therefore produce negative spreads, which Stage 2.7
cleaning policies will decide how to reject or retain.

The lower and upper bounds are single-contract discounted reference bounds
computed from spot, risk-free discount factors, and dividend discount factors.
Stage 2.6 currently provides European no-arbitrage bounds only. American,
Bermudan, and unknown exercise styles do not receive generic bounds in the
enriched quote. Stage 2.8 remains responsible for reporting arbitrage
violations.

Time to maturity for option contracts with date-only expirations is computed
from an explicit `valuation_date`, not by silently truncating the UTC
`valuation_timestamp`. Callers are responsible for choosing the correct market
calendar date before enrichment.

### Core Relationships

\[
M_{\text{spot}}=\frac{S}{K}
\]

\[
M_{\text{forward}}=\frac{F}{K}
\]

\[
k=\ln\left(\frac{K}{F}\right)
\]

### Status

**Complete**

---

## Stage 2.7 — Cleaning Policies

### Scope

Apply configurable cleaning policies to canonical and enriched data while
preserving machine-readable diagnostics for every rejection.

### Planned Capabilities

- Missing bid or ask handling
- Negative quote detection
- Crossed-market detection
- Locked-market detection
- Zero-midpoint detection
- Maximum relative-spread filtering
- Minimum volume filtering
- Minimum open-interest filtering
- Stale-quote filtering
- Maturity-range filtering
- Strike-range filtering
- Spot-moneyness filtering
- Forward-moneyness filtering
- Minimum option-price filtering
- Configurable policy composition
- `CleaningResult` with accepted quotes, rejected quotes, and diagnostics
- Machine-readable rejection reasons such as `CROSSED_MARKET`, `EXCESSIVE_SPREAD`, and `MISSING_BID`
- No silent list shrinking

### Implemented Capabilities

- `RejectionReason`
- `CleaningConfig`
- `CleaningDiagnostic`
- `RejectedQuote`
- `CleaningResult`
- `EnrichedCleaningResult`
- `clean_option_chain`
- `clean_enriched_option_quotes`
- Missing bid and ask rejection
- Empty market rejection
- Crossed and locked market policies
- Zero-midpoint rejection
- Maximum relative-spread filtering
- Canonical midpoint and relative-spread filtering without enrichment
- Minimum volume and open-interest filtering
- Stale quote filtering
- Strike-range filtering
- Maturity-range filtering for enriched quotes
- Spot- and forward-moneyness filtering for enriched quotes
- Minimum option-price filtering
- Deterministic accepted and rejected quote ordering
- Machine-readable diagnostics for every rejected quote

### Boundary Notes

Stage 2.7 operates on valid canonical objects and enriched quotes. Negative
quote values and invalid strikes remain constructor-level model invariants from
Stage 2.1, so they are not normally reachable inside cleaning policies.

Rules that depend on derived fields, such as relative spread, maturity,
moneyness, and minimum midpoint, are applied through `EnrichedOptionQuote`.
Rules that only require canonical quotes, such as one-sided markets, crossed or
locked markets, liquidity thresholds, staleness, and strike ranges, are applied
directly to `OptionChainSnapshot`.

Canonical cleaning also applies midpoint, relative-spread, and minimum-price
rules because they can be computed directly from bid and ask. Filters requiring
maturity or moneyness must use enriched quotes; passing those policies to
`clean_option_chain` is rejected rather than silently ignored.

Cleaning assumes model validation has already been run. Future-relative
timestamps are validation issues rather than staleness rejections.

### Structural Cleaning Note

Some basic structural cleaning may happen before rates and enrichment, including
negative bid or ask values, `bid > ask`, missing contract identity, and invalid
strike. Development order should still follow Stage 2.1 through Stage 2.10 to
avoid splitting the roadmap too early.

### Status

**Complete**

---

## Stage 2.8 — Static-Arbitrage Diagnostics

### Scope

Detect, report, and quantify static-arbitrage issues after canonical data,
rates, forwards, and derived fields exist. This stage should not automatically
repair prices.

### Planned Capabilities

- Single-contract lower and upper bound checks
- Call-put parity diagnostics
- Same-expiry monotonicity across strikes
- Same-expiry convexity across strikes
- Vertical-spread bound checks
- Butterfly-arbitrage diagnostics
- Calendar consistency checks
- Defer total-variance consistency and surface repair to later volatility stages

### Implemented Capabilities

- `StaticArbitrageCode`
- `StaticArbitrageConfig`
- `StaticArbitrageDiagnostic`
- `StaticArbitrageReport`
- `diagnose_static_arbitrage`
- Single-contract European lower and upper bound diagnostics
- Call-put parity diagnostics using \(C-P=D_r(F-K)\)
- Same-expiry call and put monotonicity diagnostics
- Vertical-spread lower and upper bound diagnostics
- Strike convexity diagnostics
- Butterfly arbitrage diagnostics
- Optional heuristic raw-price calendar consistency diagnostics
- Per-rule tolerances and enable flags
- Violation amount quantification
- No price repair or mutation

### Boundary Notes

Stage 2.8 consumes `EnrichedOptionQuote` objects and assumes validation,
enrichment, and any configured cleaning have already been run. Quotes without a
midpoint are skipped because there is no observable price to diagnose.

The single-contract bounds use the European bounds carried by
`EnrichedOptionQuote`. American, Bermudan, or unknown exercise styles do not get
generic bound diagnostics unless future stages add style-specific theory.
Call-put parity diagnostics are likewise restricted to European call-put pairs.
Discounted strike-gap vertical-spread bounds are restricted to European options;
American, Bermudan, and unknown exercise styles are skipped for that rule.

Cross-contract diagnostics group quotes by the full option family:
underlying symbol, option type, exercise style, contract multiplier, and
currency. This avoids comparing different economic contract classes in
monotonicity, vertical-spread, convexity, butterfly, or calendar checks.

All enriched quotes passed to the diagnostic function must share valuation
timestamp, valuation date, and spot price. Same-expiry groups must also share
time to maturity, risk-free discount factor, dividend discount factor, and
forward price. Mixed enrichment assumptions are rejected instead of being
silently compared.

Raw-price calendar monotonicity is not generally valid under arbitrary rates,
dividends, forwards, and exercise styles. The calendar rule is therefore named
as a heuristic and disabled by default. It does not attempt surface repair or
total-variance consistency.

One underlying strike-convexity violation may generate both a structural
convexity diagnostic and its tradable weighted-butterfly interpretation.

### Core Relationships

European call bounds:

\[
\max(SD_q-KD_r,0)\le C\le SD_q
\]

European put bounds:

\[
\max(KD_r-SD_q,0)\le P\le KD_r
\]

Call-put parity:

\[
C-P=SD_q-KD_r
\]

Equivalently:

\[
C-P=D_r(F-K)
\]

### Status

**Complete**

---

## Stage 2.9 — pandas Interoperability

### Scope

Add pandas only after the domain layer is stable. pandas is a boundary-layer
tool, not the internal market-data model.

### Planned Capabilities

- Snapshot to records
- Records to snapshot
- Snapshot to DataFrame
- DataFrame to snapshot
- Enriched quotes to DataFrame
- Diagnostics to DataFrame
- Cleaning report to DataFrame
- APIs such as `option_chain_to_dataframe(snapshot)` and `option_chain_from_dataframe(frame)`

### Implemented Capabilities

- `option_chain_to_records`
- `option_chain_from_records`
- `option_chain_to_dataframe`
- `option_chain_from_dataframe`
- `enriched_quotes_to_records`
- `enriched_quotes_to_dataframe`
- `validation_report_to_records`
- `validation_report_to_dataframe`
- `cleaning_result_to_records`
- `cleaning_result_to_dataframe`
- `static_arbitrage_report_to_records`
- `static_arbitrage_report_to_dataframe`
- Snapshot round-trip through records and DataFrame
- Enriched quote export to records and DataFrame
- Validation, cleaning, and static-arbitrage diagnostics export
- Stable DataFrame columns for empty exports
- Snapshot and underlying quote consistency checks when rebuilding snapshots
- Pure-Python records conversion without requiring pandas

### Boundary Notes

pandas remains a boundary-layer dependency. The core market-data models,
validation, enrichment, cleaning, and static-arbitrage diagnostics do not import
pandas. DataFrame functions import pandas only when called and raise a clear
runtime error if pandas is unavailable. DataFrame input normalises pandas-native
missing values such as `pd.NA`, `NaT`, and `NaN` to `None` before entering the
pure-Python records parser.

Records and DataFrames include trade rows with `record_type="trade"` when a
snapshot has trades. Existing quote rows are accepted without a type tag.
Quote sizes, volume, open interest, and trade sizes retain their integer
precision when missing values are present.

The records representation is the stable intermediate format for dataset
exports and future serialisation work. DataFrame conversion is a convenience
wrapper around records conversion, not an internal storage model.

`option_chain_from_records` rebuilds one non-empty snapshot at a time. Empty
records cannot reconstruct a snapshot. A trade-only snapshot has trade rows and
can be reconstructed. Snapshot-level metadata and underlying quote fields must
be consistent across all rows.

Enriched quote exports include the original canonical quote and contract
columns plus derived research columns, so resulting research tables can still
audit bid, ask, liquidity, contract identity, and source metadata.

### Status

**Complete**

---

## Stage 2.10 — Serialisation and Dataset Snapshots

### Scope

Produce reproducible canonical datasets with metadata, diagnostics, and
deterministic identifiers.

### Planned Capabilities

- Canonical JSON output
- CSV export
- Metadata manifest
- Schema version
- Source information
- Ingestion timestamp
- Valuation timestamp
- Normalisation configuration
- Cleaning configuration
- Rate and dividend assumptions
- Accepted and rejected quote counts
- Diagnostic summary
- Deterministic dataset identifiers

### Implemented Capabilities

- `DATASET_SCHEMA_VERSION`
- `DatasetManifest`
- `DatasetSnapshotWriteResult`
- `canonical_snapshot_to_json(snapshot)`
- `write_canonical_json(snapshot, path)`
- `write_option_chain_csv(snapshot, path)`
- `write_market_data_dataset(...)`
- Stable canonical JSON payloads
- Deterministic option-chain CSV export
- Reproducible dataset directory layout
- Manifest metadata for schema, source, timestamps, configuration, assumptions,
  input/canonical/accepted/rejected counts, diagnostic counts, hashes, and
  dataset ID
- Optional raw source copying with input hashing
- Validation, cleaning, and static-arbitrage diagnostic export
- Typed JSON-compatible metadata canonicalization
- UTC `Z` datetime canonicalization
- Standard JSON output with `NaN` and `Infinity` rejected
- Dataset identity covers canonical, accepted, rejected, diagnostic, raw, and
  semantic configuration payloads

### Boundary Notes

- Serialisation uses only standard-library JSON, CSV, hashing, and filesystem
  APIs.
- pandas remains an optional boundary tool from Stage 2.9 and is not required
  by dataset serialisation.
- `processed/canonical_option_chain.csv` records the full canonical snapshot.
- `processed/accepted_quotes.csv` records cleaning-accepted quotes when a
  cleaning result is provided, otherwise it mirrors the canonical snapshot.
- `rejected/rejected_quotes.csv` records one rejected quote per row.
- `diagnostics/cleaning.json` records one cleaning diagnostic per issue.
- Dataset identifiers are content hashes over stable serialised payloads and
  manifest inputs. They are reproducibility identifiers, not market-data vendor
  identifiers.
- `ingestion_timestamp` is audit metadata and intentionally excluded from
  `dataset_id`.
- `output_hash` hashes canonical logical content, not the physical bytes of
  output files.
- The logical content hash includes the full accepted/rejected quote
  partitions, not only their counts.
- Canonical JSON includes trades and preserves the underlying quote and snapshot
  metadata for snapshots without quotes. CSV files remain quote tables.
- Dataset files are staged before publication. A rewrite withdraws the old
  manifest, publishes the new manifest last, and restores the previous files
  after a publication error. Readers holding an earlier manifest must coordinate
  with writers during a rewrite. An interrupted process can leave a sibling
  lock and recovery backup.
- Configuration metadata requires non-empty string object keys; unsupported or
  ambiguous key types are rejected instead of silently converted.

### Target Dataset Layout

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

Example `manifest.json`:

```json
{
  "schema_version": "1.0",
  "source": "test_fixture",
  "as_of": "2026-07-30T08:00:00Z",
  "day_count": "ACT/365F",
  "cleaning_config": {},
  "input_hash": "...",
  "output_hash": "..."
}
```

### Stage 2 Development Order

```text
2.1 Canonical models
        ↓
2.2 Validation framework
        ↓
2.3 CSV ingestion
        ↓
2.4 Normalisation
        ↓
2.5 Rates, dividends, and forwards
        ↓
2.6 Derived fields and year fractions
        ↓
2.7 Cleaning policies
        ↓
2.8 Static-arbitrage diagnostics
        ↓
2.9 pandas interoperability
        ↓
2.10 Serialisation and dataset snapshots
```

### Status

**Complete**

---

# Stage 3 — Volatility Research

## Objective

Transform cleaned option prices into implied-volatility observations, research-ready
smiles, non-parametric surfaces, model-independent diagnostics, and calibrated
parametric volatility models.

The development dependency is:

```text
cleaned market prices
-> all implied-volatility observations
-> selected and structured smiles
-> interpolated surface
-> model-independent diagnostics
-> SVI / SSVI calibration
-> historical volatility research
```

---

## Stage 3.1 — Implied Volatility Chains

### Scope

Convert cleaned option prices into structured implied-volatility data. This stage
generates all usable IV observations; it does not decide which observations enter
smile research.

### Planned Capabilities

- Chain-wide bid, midpoint, and ask implied volatility
- Solver status and machine-readable failure reasons
- No-arbitrage failure reporting
- Explicit handling for lower-bound `0.0` and upper-bound `inf`
- Vega calculation
- Spot moneyness and forward moneyness
- Log-forward-moneyness
- Deterministic ordering
- Chain-level summary
- Records and pandas export

### Failure Reasons

- `MISSING_PRICE`
- `OUTSIDE_BOUNDS`
- `INVALID_INPUT`
- `SOLVER_FAILED`
- `NON_FINITE_RESULT`

### Diagnostic Flags

- `LOW_VEGA`
- `VEGA_UNAVAILABLE`
- `UPPER_BOUND_IV`

Failure reasons explain why a result has no IV. Diagnostic flags describe quality,
stability, or special boundary behaviour for returned observations. Stage 3.1 v1
uses `UPPER_BOUND_IV` for the legal `inf` upper-bound result and treats unexpected
`nan` or non-boundary `inf` as `FAILED / NON_FINITE_RESULT`. A finite successful IV
with unavailable Vega remains usable but carries `VEGA_UNAVAILABLE`; Stage 3.1 does
not expose `UNSTABLE_IV` until a distinct stability definition exists.

### Completion Standard

- Synthetic chains recover the input volatility
- Bid IV <= mid IV <= ask IV
- A single failed inversion does not fail the whole chain
- Status and failure reason are machine-readable
- Vega and moneyness fields are complete
- Output ordering is deterministic
- Records and DataFrame exports preserve semantic fields
- `0.0`, `inf`, missing prices, and invalid prices have explicit behaviour

### Implementation Notes

- Initial API added via `build_implied_volatility_chain`
- Chain records and pandas export added
- Bid, midpoint, and ask IV results carry status, failure reason, Vega, and
  diagnostic flags
- The solver and chain share one scale-aware absolute/relative price-bound
  tolerance policy and use BSM bounds for every exercise style
- `IV = 0.0` maps to `Vega = 0.0` as an explicit boundary representation policy,
  not as a continuous-limit claim for every degenerate parameter combination
- The public volatility-chain export uses `log_forward_moneyness =
  log(strike / forward_price)` to fix the skew-coordinate sign convention
- End-to-end CSV pipeline reuses Stage 2 ingestion, enrichment, and cleaning APIs
  before calling `build_implied_volatility_chain`
- Deterministic 50,000-row synthetic smoke/performance coverage is kept behind the
  separate `large` pytest marker
- Deterministic IV-chain CSV export preserves all Stage 3.1 semantic fields

### Status

**Complete**

---

## Stage 3.2 — Smile Representation and Skew Analysis

### Scope

Construct research-ready smiles from IV observations and extract smile metrics.
This stage owns quote-selection policy for smile research.

### Planned Capabilities

- Expiry-level `VolatilitySmile`
- OTM quote selection
- Duplicate-strike resolution
- ATM definition
- Strike, delta, and log-moneyness coordinates
- Bid-mid-ask IV spread
- Liquidity-aware point selection
- ATM volatility
- Skew slope
- Curvature
- Risk reversals
- Butterflies
- Term structure

### Status

**Complete**

### Implemented in Stage 3.2.1-3.2.6

- `VolatilitySmilePoint`, `VolatilitySmile`, `SmileSelectionConfig`, and
  `SmileSelectionDiagnostic` domain models
- Expiry grouping directly from `ImpliedVolatilityChain`
- Explicit bid, midpoint, or ask IV source selection
- Forward-based OTM selection and configurable liquidity filters
- Latest-timestamp resolution and explicit call/put duplicate-strike policy
- Separate nearest-forward and observed-ATM semantics
- Group market-state consistency validation with group-level diagnostics
- Machine-readable unavailable delta and IV-spread flags
- Strike, delta, moneyness, bid-mid-ask IV, Vega, and liquidity representation
- Stable records and pandas exports for selected points and diagnostics
- Medium pipeline integration and separately marked 50,000-row smoke coverage

### Implemented Metrics and Term Structure

- Observed or bracketed linear total-variance ATM volatility
- Explicit no-extrapolation policy with independent metric failure reasons
- Unweighted local quadratic fit on configurable observed points
- Total-variance skew slope `dw/dk` at `k=0`
- Total-variance curvature `d2w/dk2` at `k=0`
- Partial success across ATM, skew, and curvature
- Stable records and pandas export with source-point and fit-policy semantics
- Synthetic linear/quadratic recovery and pipeline integration coverage
- Signed-delta target IV interpolation with exact-match precedence
- Explicit no-extrapolation, ambiguity, and degenerate-coordinate diagnostics
- Call-minus-put risk reversals and symmetric delta butterflies
- Reuse of the exact Stage 3.2 ATM result for butterfly calculations
- Independent call, put, RR, and BF statuses with partial success
- Expiry term structures grouped by symbol and valuation snapshot
- Explicit duplicate-expiry/maturity error policy
- Aggregate Stage 3.2 orchestration and independent summary counts
- Deterministic records, pandas DataFrames, CSV bytes, and SHA-256 exports
- Medium integration and separately marked 50,000-row end-to-end validation

---

## Stage 3.3 — Non-Parametric Volatility Surface

### Scope

Construct transparent, explainable volatility surfaces before introducing
parametric models.

### Planned Capabilities

- Total variance interpolation using `w(k,T)=sigma_imp^2(k,T)T`
- Log-forward-moneyness coordinate
- Per-expiry smile interpolation
- Maturity interpolation
- Explicit extrapolation policy
- Sparse-grid handling
- Fitted-versus-observed residuals
- Query API

### Validation

- Recovery of observed liquid quotes
- Stability under sparse data
- Cross-validation across withheld quotes
- Fitted-versus-observed residual reporting

### Status

**Complete**. See [interpolation, sparse-grid policy, and validation](volatility_surface.md).

---

## Stage 3.4 — Surface and Smile Arbitrage Diagnostics

### Scope

Detect, report, and quantify model-independent arbitrage diagnostics for observed
smiles and interpolated surfaces. This stage does not automatically repair
violations.

### Planned Capabilities

- Calendar total-variance monotonicity
- Strike convexity and butterfly consistency
- Risk-neutral density positivity diagnostics
- Wing behaviour
- Interpolation-induced violations
- Observed quote residuals
- Violation severity and magnitude
- Detect, report, and quantify-only workflow

### Status

**Complete**. See [calendar and sampled strike diagnostics](volatility_surface.md#static-arbitrage-diagnostics).

---

## Stage 3.5 — Parametric Volatility Models

### Planned Capabilities

- SVI per-expiry smile parameterisation
- SSVI surface construction
- Weighted least squares
- Spread weighting
- Vega weighting
- Parameter constraints
- Robust initialisation
- Calibration diagnostics
- Fitted residuals
- Arbitrage diagnostics
- Comparison against the non-parametric surface

### Status

**Complete**. See [calibration, diagnostics, and held-out comparison](parametric_volatility.md).

---

## Stage 3.6 — Volatility Dynamics

### Scope

Conduct historical volatility research using time-series snapshots rather than a
single-day option chain.

### Planned Research

- Realised versus implied volatility
- Volatility risk premium
- Sticky-strike behaviour
- Sticky-delta behaviour
- Smile response to spot moves
- Term-structure evolution
- Event volatility
- Regime analysis

### Status

**Complete**. See the [SPY volatility dynamics study](../research/real_data/volatility_dynamics.md).

---

# Stage 4 — Strategy Research

## Objective

Build a reproducible framework for defining, valuing, and analysing option strategies.

---

## Stage 4.1 — Instruments and Positions

### Planned Capabilities

- Option contract objects
- Underlying positions
- Cash positions
- Long and short quantities
- Contract multipliers
- Position-level valuation
- Position-level Greeks

### Status

**Planned**

---

## Stage 4.2 — Multi-Leg Strategies

### Planned Capabilities

- Vertical spreads
- Straddles
- Strangles
- Butterflies
- Condors
- Calendars
- Diagonals
- Covered positions
- Custom strategy composition

### Status

**Planned**

---

## Stage 4.3 — Payoff and Scenario Analysis

### Planned Capabilities

- Expiry payoff
- Mark-to-market P&L
- Spot-volatility scenario grids
- Time-decay scenarios
- Break-even analysis
- Maximum gain and loss
- Greek decomposition
- Transaction-cost assumptions

### Status

**Planned**

---

## Stage 4.4 — Historical Strategy Research

### Planned Capabilities

- Entry and exit rules
- Option selection rules
- Rolling logic
- Position sizing
- Transaction costs
- Slippage assumptions
- Survivorship-safe data handling
- Performance attribution

### Status

**Planned**

---

# Stage 5 — Portfolio Risk Engine

## Objective

Aggregate instrument-level valuation and sensitivities into portfolio-level risk measures.

---

## Stage 5.1 — Portfolio Valuation

### Planned Capabilities

- Multi-instrument portfolios
- Net present value
- Aggregated Greeks
- Grouping by underlying
- Grouping by expiry
- Grouping by strategy
- Currency-aware valuation architecture

### Status

**Planned**

---

## Stage 5.2 — Scenario Risk

### Planned Capabilities

- Spot shocks
- Volatility shocks
- Rate shocks
- Time-decay shocks
- Parallel and non-parallel volatility moves
- Combined stress scenarios
- Full revaluation
- Greek approximation comparison

### Status

**Planned**

---

## Stage 5.3 — P&L Attribution

### Planned Capabilities

- Delta contribution
- Gamma contribution
- Vega contribution
- Theta contribution
- Rho contribution
- Higher-order residual
- Realised versus predicted P&L
- Daily attribution reports

### Status

**Planned**

---

## Stage 5.4 — Statistical Risk

### Planned Capabilities

- Historical Value at Risk
- Parametric Value at Risk
- Expected Shortfall
- Volatility and correlation estimation
- Stress-period replay
- Model-risk comparison

### Status

**Planned**

---

# Stage 6 — Electronic Market-Making Simulation

## Objective

Build an event-driven simulation of an options market maker managing quotes, fills, inventory, and risk.

This stage is intended to integrate the pricing, volatility, strategy, and risk components developed earlier.

---

## Stage 6.1 — Market Simulator

### Planned Capabilities

- Event-driven simulation loop
- Underlying-price process
- Option fair-value updates
- Bid and ask quotes
- Order arrivals
- Probabilistic or queue-based fills
- Position and cash accounting
- Configurable latency assumptions

### Status

**Planned**

---

## Stage 6.2 — Quoting Engine

### Planned Capabilities

- Fair-value-based quoting
- Minimum spread
- Volatility-aware spread
- Liquidity-aware spread
- Inventory skew
- Delta-risk skew
- Quote-size control
- Quote refresh logic

### Status

**Planned**

---

## Stage 6.3 — Hedging Engine

### Planned Capabilities

- Delta hedging
- Threshold-based hedging
- Periodic hedging
- Transaction costs
- Hedge slippage
- Hedge latency
- Residual risk tracking
- Comparison of hedging policies

### Status

**Planned**

---

## Stage 6.4 — Inventory and Risk Controls

### Planned Capabilities

- Position limits
- Delta limits
- Gamma limits
- Vega limits
- Loss limits
- Quote widening
- Quote withdrawal
- Kill-switch logic
- Risk-aware size reduction

### Status

**Planned**

---

## Stage 6.5 — Market-Making Evaluation

### Planned Metrics

- Gross and net P&L
- Spread capture
- Hedge P&L
- Inventory P&L
- Adverse selection
- Transaction costs
- Risk-adjusted returns
- Maximum drawdown
- Inventory utilisation
- Fill rate
- Quote competitiveness

### Planned Experiments

- Symmetric versus inventory-skewed quoting
- Different hedge thresholds
- Different volatility regimes
- Spread-width sensitivity
- Latency sensitivity
- Informed versus uninformed order flow
- Risk-limit effectiveness

### Status

**Planned**

---

# Stage 7 — Advanced Models and Research

## Objective

Extend the platform beyond the initial Black–Scholes–Merton assumptions.

Potential areas will be prioritised based on the maturity of the earlier stages.

---

## Candidate Extensions

### Pricing Models

- Local volatility
- Heston stochastic volatility
- Merton jump diffusion
- SABR
- Finite-difference PDE methods
- Least-Squares Monte Carlo
- Fourier pricing methods

### Products

- Barrier options
- Asian options
- Digital options
- Lookback options
- Bermudan options
- Variance swaps
- Volatility swaps

### Market Microstructure

- Limit-order-book simulation
- Queue-position modelling
- Adverse-selection models
- Order-flow imbalance
- Multi-venue execution
- Latency and stale-quote risk

### Research Infrastructure

- Calibration pipelines
- Experiment tracking
- Reproducible reports
- Benchmark datasets
- Performance profiling
- Parallel computation
- Optional compiled acceleration

### Status

**Exploratory**

---

# Cross-Cutting Engineering Work

The following work applies throughout all stages rather than belonging to a single milestone.

## Documentation

- Public API documentation
- Mathematical assumptions
- Formula and unit conventions
- Usage examples
- Numerical limitations
- Research methodology
- Architecture decisions

## Quality Assurance

- Unit tests
- Property-based tests
- Regression tests
- Numerical convergence tests
- Cross-model validation
- Static analysis
- Type checking
- Continuous integration

## Performance

Optimisation should occur only after correctness is established.

Potential work includes:

- Profiling
- Vectorisation
- Memory reduction
- Parallel simulation
- Caching
- Optional NumPy or compiled backends
- Benchmark tracking

## Reproducibility

- Fixed random seeds where appropriate
- Versioned configuration
- Raw and processed data separation
- Deterministic test fixtures
- Documented environment setup
- Saved experiment outputs

---

# Near-Term Priorities

Begin [Stage 4.1 — Instruments and Positions](#stage-41--instruments-and-positions).
Position-level valuation and Greeks provide the basis for multi-leg strategy
construction and scenario analysis.

The project should not advance to complex volatility or market-making research until the foundational numerical methods are independently validated.

---

# Definition of Project Success

NCX Derivatives will be considered successful when it can:

- Price standard European and American derivatives using multiple methods
- Recover and analyse implied volatility from market quotes
- Construct and diagnose volatility smiles and surfaces
- Represent and value multi-leg portfolios
- Explain portfolio P&L through risk sensitivities
- Simulate a market maker that quotes, trades, hedges, and controls inventory
- Produce reproducible quantitative research supported by documented assumptions and automated tests

The final objective is not to reproduce the infrastructure of a full trading firm. It is to build a transparent, technically rigorous miniature derivatives research and market-making stack that demonstrates the mathematical, engineering, and decision-making foundations behind professional options trading.
