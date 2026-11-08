# Stage 3.6 SPY volatility dynamics

This study uses the frozen 2010–2023 Research 001A SPY option-expiry and underlying-return panels. It runs the same workflow for the projected-cash-dividend baseline and trailing-dividend sensitivity. The two runs use the same observed sessions and fixed tenors. [Research 001A inputs and provenance](research_001_atm_stability/outputs/spy_2010_2023_carry_comparison/research_001A_full_report_en.md)

## Implied and forward realized volatility

We use the Stage 3.3 ATM fixed-tenor midpoint estimate with no extrapolation. Forward realized variance is the sum of squared subsequent close-to-close log returns through the last observed close on or before the calendar horizon, annualized by `365 / tenor_days`. We exclude windows when the dataset ends before the target date, an included return crosses a split or uses an invalid price, or the window crosses or ends inside an observation gap longer than seven calendar days. The variance risk premium proxy is ATM IV squared minus that ex-post annualized realized variance. The return series uses observed SPY prices and the implied leg is ATM variance, so this proxy differs from a dividend-adjusted variance swap measure. Adjacent daily windows overlap.

| Tenor | Matched sessions | Baseline median IV | Median forward realized vol | Baseline median variance gap | Trailing median variance gap |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 21D | 3,289 | 14.08% | 12.02% | 0.00471 | 0.00484 |
| 45D | 3,439 | 14.91% | 12.60% | 0.00607 | 0.00594 |
| 90D | 3,377 | 15.87% | 13.13% | 0.00841 | 0.00825 |
| 150D | 2,071 | 17.27% | 13.99% | 0.01061 | 0.01080 |

Regimes use the current 21D IV against its trailing 252-session 25th and 75th percentiles, using only preceding data and requiring 126 valid observations. In the baseline, median absolute daily 21D IV moves are 0.47 volatility points in the low regime and 1.63 in the high regime. The high-regime 21D median variance gap is positive (0.00877), while the mean is negative (−0.01235). The term panel also reports annualized forward variance and its daily change for each adjacent tenor pair, retaining negative values if the observed curve produces them.

## Smile response and sticky conventions

The smile panel pairs the same expiry on adjacent observed sessions. The Research 001A eligibility filter admits 7–180 DTE, so the output bucket labeled `01-30` contains only 7–30 DTE. The descriptive slope of the change in ATM IV against the log spot return is −1.66 at 7–30 DTE and −0.76 at 121–180 DTE in the baseline. The output also reports separate response slopes and sample counts for the 25Δ risk reversal, butterfly, and local total-variance skew.

The sticky-strike prediction evaluates the prior smile at today's ATM forward strike. It reconstructs the prior 25Δ wing strikes from their signed BSM spot deltas, IVs, forward and dividend discount factor, then interpolates total variance between those strikes and the prior ATM anchor. The sticky-delta prediction holds the prior smile in forward-normalized call-equivalent delta and solves the fixed point for today's ATM-forward IV. Both predictions use the same current maturity and strike target for comparison and evaluate errors against today's ATM midpoint. The prior shape comes from three reconstructed anchors. Missing anchors, unsupported strike queries, and unverified delta roots are excluded from the paired comparison.

| DTE | Baseline common pairs | Sticky-strike MAE | Sticky-delta MAE |
| --- | ---: | ---: | ---: |
| 7–30 | 20,988 | 0.778 | 1.295 |
| 31–60 | 10,520 | 0.561 | 0.992 |
| 61–120 | 8,846 | 0.458 | 0.760 |
| 121–180 | 6,279 | 0.419 | 0.656 |

MAE is in volatility points. Sticky strike had smaller ATM prediction errors in all four bands under both carry specifications. The comparison conditions on the current forward and uses a reconstructed prior shape, so it measures relative fit to these EOD observations.

## Event windows

The default examples compare the previous observed close with the first close on or after three FOMC release dates. The March 15 release maps to the Monday session. The output records the SPY return, its square, the prior ATM annualized variance scaled by the calendar length of the close-to-close window, and changes in ATM IV and adjacent forward variance. These windows contain all information arriving between the two closes. [March 3, 2020 release](https://www.federalreserve.gov/newsevents/pressreleases/monetary20200303a.htm) · [March 15, 2020 release](https://www.federalreserve.gov/newsevents/pressreleases/monetary20200315a.htm) · [June 15, 2022 release](https://www.federalreserve.gov/newsevents/pressreleases/monetary20220615a.htm)

| Release | Mapped session | SPY log return | 21D ATM IV change | 90D ATM IV change |
| --- | --- | ---: | ---: | ---: |
| 2020-03-03 | 2020-03-03 | −2.82% | +8.89 vp | +2.85 vp |
| 2020-03-15 | 2020-03-16 | −12.23% | +31.82 vp | +21.59 vp |
| 2022-06-15 | 2022-06-15 | +1.41% | −1.42 vp | −1.79 vp |

## Reproduction

Install the research dependencies and run from the repository root. Substitute `spy_2010_2023_ncx_trailing_dividend` and use a separate output directory for the alternative. To supply custom event windows, pass `--events-csv` with `event_date,event` columns and use the date of the intended first post-release close or a non-session release date.

```powershell
$root = 'research/real_data/research_001_atm_stability/outputs'
python -m research.real_data.volatility_dynamics `
  --research-001a "$root/spy_2010_2023_ncx_baseline" `
  --output-dir reports/generated/stage_3_6_spy
```

The command writes session-level term and expiry-level smile Parquet panels, term/regime and smile-response summaries, event results, and a SHA-256 manifest. The baseline rerun reproduces all output hashes.
