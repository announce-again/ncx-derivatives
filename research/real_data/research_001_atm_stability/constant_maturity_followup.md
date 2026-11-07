# Research 001A fixed-tenor follow-up

The 2010–2023 SPY reconstruction gives the same stability ranking under all three ATM designs and both historical carry specifications. Median absolute changes fall from 21D through 45D, 90D, and 150D. That ranking also holds on the 1,537 session pairs that support all four tenors, all three designs, and both carry specifications. [Original Research 001A report](outputs/spy_2010_2023_carry_comparison/research_001A_full_report_en.md)

## Coverage and stability

Coverage is the share of the 3,508 observed SPY sessions with a constant-maturity level. Counts are identical across both specifications. Changes below are median absolute one-session changes in volatility points, calculated on the intersection of all three designs and both carry specifications at each target tenor.

| Target | Fixed-tenor coverage | Common pairs | Baseline fixed expiry | Baseline nearest | Baseline fixed tenor | Trailing fixed expiry | Trailing nearest | Trailing fixed tenor |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 21D | 3,315 / 3,508 (94.5%) | 3,088 | 0.746 | 0.760 | 0.755 | 0.876 | 0.879 | 0.862 |
| 45D | 3,500 / 3,508 (99.8%) | 3,075 | 0.602 | 0.624 | 0.596 | 0.727 | 0.761 | 0.684 |
| 90D | 3,500 / 3,508 (99.8%) | 3,291 | 0.491 | 0.502 | 0.480 | 0.493 | 0.512 | 0.476 |
| 150D | 2,198 / 3,508 (62.7%) | 1,979 | 0.457 | 0.462 | 0.439 | 0.417 | 0.429 | 0.398 |

Eight sessions lack eligible expiry quotes at every tenor. Another 185 sessions lack a lower bracket at 21D, and 1,302 lack an upper bracket at 150D within the original 180D eligibility cap. The median maturity bracket spans 2, 14, 28, and 28 calendar days at 21D, 45D, 90D, and 150D, respectively, counting exact matches as zero-width brackets.

## Effect of changing the selected expiry

The roll component is the signed nearest-tenor move minus the signed move of the currently selected contract across the same two sessions. It is zero when the selected expiry stays the same. On rollover pairs its median absolute magnitude under baseline carry is 0.332, 0.313, 0.311, and 0.185 volatility points at 21D, 45D, 90D, and 150D. Under trailing-dividend carry the corresponding values are 0.396, 0.381, 0.322, and 0.140. The rollover subsamples contain 1,172, 492, 186, and 115 pairs, respectively. Rollovers contribute to measured day-to-day moves, while the maturity ranking persists in the fixed-expiry and constant-maturity series.

## Pairing and reproduction

- **Fixed expiry** uses the current session's nearest contract on both dates. The contract must have an eligible ATM midpoint on the preceding session. This defines a paired fixed-expiry control and differs from the original report's pooled DTE-bin comparison.
- **Nearest tenor** uses each date's published nearest-expiry selection and the original tenor-specific tolerances.
- **Constant maturity** uses eligible 7–180D expiry nodes with at least five selected smile points and a successful midpoint ATM estimate. On an exact DTE match it uses the observed node. Otherwise it interpolates ATM total variance between the immediately surrounding maturities in ACT/365F and converts back to IV. It records missing brackets and uses no extrapolation.

The preceding date comes from the frozen underlying observation panel, so a missing option quote cannot cause a series to bridge across an observed SPY session. The output includes missingness, brackets, weights, raw moves, rollover flags, per-tenor and balanced sample summaries, and SHA-256 hashes of the input and output artifacts.

Run from the repository root with the research dependencies installed.

```powershell
$root = 'research/real_data/research_001_atm_stability/outputs'
python -m research.real_data.research_001_atm_stability.compare_constant_maturity `
  --baseline "$root/spy_2010_2023_ncx_baseline" `
  --trailing-dividend "$root/spy_2010_2023_ncx_trailing_dividend" `
  --output-dir reports/generated/research_001a_constant_maturity
```
