# Stage 3.2 Completion Summary

Stage 3.2 consumes Stage 3.1 `ImpliedVolatilityChain` output and completes the
following immutable analysis path:

```text
IV observations
-> selected expiry smiles
-> ATM/skew/curvature
-> signed-delta IV
-> RR/BF
-> expiry term structures
-> deterministic exports
```

It does not construct or interpolate a volatility surface.

## Run

From the repository root:

```powershell
.\.venv\Scripts\python.exe -B examples\market_data\11_stage_3_2_volatility_smiles.py --rows 50000 --output-dir .tmp\examples_output\volatility_smiles
```

The normal suite excludes the registered `large` marker. Run the 50k smoke test
separately:

```powershell
.\.venv\Scripts\python.exe -B -m pytest -m large tests\test_volatility_pipeline_large.py --basetemp=.tmp\pytest-large -p no:cacheprovider -vv
```

## Observed 50,000-Row Conservation

Observed on 2026-08-02 with the commands above:

```text
input rows                         50,000
Stage 3.1 IV quotes               49,985
selected smile points             24,990
excluded IV quotes                24,995
smiles                               500
local metric results                 500
delta metric aggregates              500
term structures                       100
term-structure expiry points           500
```

The selection accounting identity is:

```text
49,985 IV quotes = 24,990 selected points + 24,995 excluded quotes
```

Every smile produced one local result, one aggregate delta result, and one term
structure point. No failed row or metric was silently dropped.

## Observed Metric Outcomes

```text
metric             success  failure
ATM                    500        0
skew                   500        0
curvature              500        0
25-delta call           500        0
25-delta put            500        0
RR25                    500        0
BF25                    500        0
```

Other inputs are not required to achieve all-success outcomes. Failures remain in
the same result and export tables with machine-readable reasons.

## Observed Throughput

```text
pipeline                              5,276 input rows/second
smile selection                      37,124 IV quotes/second
local metrics                         6,274 smiles/second
delta metrics                         4,742 smiles/second
term-structure assembly             287,704 expiry results/second
generation-to-export                  3,483 input rows/second
```

`pipeline` throughput covers ingestion through Stage 3.1 IV construction and is
reported by `VolatilityPipelineResult`. `generation-to-export` starts before
synthetic CSV generation and ends after all Stage 3.2 exports. These rates have
different boundaries and must not be compared as the same benchmark. All values
are development-machine observations, not service-level guarantees.

## Deterministic Export Hashes

```text
smile_metrics.csv       283fb1052ab9805b86d85acfecf5754026e07c0feece1b0901783d404ac90295
delta_volatility.csv    03378b02fe87f25c1292bc6a2d7de8bbaab827e0ddbca94548460f80c4455621
risk_reversals.csv      5ffca412a44c99e6a4a05da3f033f8343893406b19a4c60af91083dcd5050250
butterflies.csv         90bad749306d437f1d6350734d97d445f88a1c2baf6baae38d2eeffeca5fc600
delta_structures.csv    296a681bf3d8dfb2f78b4a9fb955238857e6a09e88794bf0e352acd61803747b
term_structures.csv     e3b623a1099b8f2eda141d76395c10ba81d590cffdeccf5f01cec32a72c8a17f
analysis_summary.csv    c83004171c94d18659fef6de88a985346193e609b7fed0fbe92b42a845c76b0a
```

The large test writes each analysis export twice and requires identical records,
bytes, and SHA-256 values. It also reruns aggregate analysis with reversed smile
input and requires identical delta and term-structure records.

## Next Stage

Stage 3.2 is complete. The [Stage 3.3 non-parametric surface](volatility_surface.md)
uses its selected smiles as interpolation inputs.
