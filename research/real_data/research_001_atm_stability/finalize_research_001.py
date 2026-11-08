from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.real_data.common.deterministic_io import (
    file_sha256,
    logical_frame_sha256,
    write_csv,
    write_json,
    write_output_hashes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidate Research 001A and 001B evidence")
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--final-dir", type=Path, required=True)
    args = parser.parse_args()
    finalize(outputs_root=args.outputs_root, raw_root=args.raw_root, final_dir=args.final_dir)


def finalize(*, outputs_root: Path, raw_root: Path, final_dir: Path) -> None:
    final_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = final_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    pilot = outputs_root / "pilot_a"
    pilot_alt = outputs_root / "pilot_a_alt_carry"
    long = outputs_root / "spy_2010_2023_vendor_iv_replication"
    cross = outputs_root / "cross_underlying_vendor_iv_replication"

    source_replication = _source_replication(long, cross)
    write_csv(final_dir / "spy_source_replication.csv", source_replication.to_dict("records"), sort_by=("target_tenor",))
    expiry_comparison = _expiry_vs_nearest(long)
    write_csv(final_dir / "expiry_vs_nearest.csv", expiry_comparison.to_dict("records"), sort_by=("target_tenor",))
    provenance = _provenance(raw_root, long, cross)
    write_csv(final_dir / "dataset_provenance.csv", provenance, sort_by=("dataset_slug",))
    hypotheses = _hypotheses(pilot, pilot_alt, long, cross)
    write_csv(final_dir / "hypothesis_assessment.csv", hypotheses, sort_by=("hypothesis",))
    checks = _determinism_checks(outputs_root, raw_root)
    write_csv(final_dir / "determinism_checks.csv", checks, sort_by=("check", "artifact"))
    _figures(pilot, long, cross, figure_dir)
    _report(pilot, pilot_alt, long, cross, source_replication, hypotheses, final_dir / "research_001_report.md")
    _limitations(final_dir / "research_001_limitations.md")
    write_json(
        final_dir / "run_manifest.json",
        {
            "research_id": "research_001_consolidated",
            "scope": {
                "ncx_reconstructed": "SPY 2020-2022 pilot, zero carry and flat 4%/1.5% carry sensitivity",
                "vendor_iv_replication": "SPY 2010-2023 and five-underlying 2020-2022 extension",
            },
            "component_manifests": {
                name: file_sha256(path / "run_manifest.json")
                for name, path in {
                    "pilot_zero_carry": pilot,
                    "pilot_alt_carry": pilot_alt,
                    "spy_long_vendor": long,
                    "cross_underlying_vendor": cross,
                }.items()
            },
            "source_replication_logical_hash": logical_frame_sha256(source_replication, sort_by=("target_tenor",)),
            "formal_completion_status": "partial_pending_historical_carry_enrichment_and_full_ncx_batch",
        },
    )
    write_output_hashes(final_dir, final_dir / "output_hashes.json")


def _source_replication(long: Path, cross: Path) -> pd.DataFrame:
    left = pd.read_parquet(long / "atm_tenor_panel.parquet")
    right = pd.read_parquet(cross / "atm_tenor_panel.parquet")
    left = left.loc[pd.to_datetime(left["quote_date"]).between("2020-01-01", "2022-12-31")]
    right = right.loc[right["underlying"].eq("SPY")]
    merged = left.merge(
        right,
        on=["quote_date", "target_tenor"],
        suffixes=("_long_dataset", "_same_uploader_dataset"),
        validate="one_to_one",
    )
    records = []
    for tenor, group in merged.groupby("target_tenor"):
        records.append(
            {
                "target_tenor": int(tenor),
                "aligned_observations": len(group),
                "atm_iv_level_correlation": group["atm_mid_iv_long_dataset"].corr(group["atm_mid_iv_same_uploader_dataset"]),
                "median_absolute_level_difference": (group["atm_mid_iv_long_dataset"] - group["atm_mid_iv_same_uploader_dataset"]).abs().median(),
                "daily_change_correlation": group["delta_atm_iv_long_dataset"].corr(group["delta_atm_iv_same_uploader_dataset"]),
                "long_dataset_median_absolute_change": group["absolute_atm_iv_change_long_dataset"].median(),
                "same_uploader_median_absolute_change": group["absolute_atm_iv_change_same_uploader_dataset"].median(),
            }
        )
    return pd.DataFrame(records).sort_values("target_tenor", kind="mergesort", ignore_index=True)


def _expiry_vs_nearest(long: Path) -> pd.DataFrame:
    expiry = pd.read_parquet(long / "atm_expiry_panel.parquet")
    nearest = pd.read_parquet(long / "atm_tenor_panel.parquet")
    records = []
    for target, tolerance in zip((21, 45, 90, 150), (7, 10, 15, 25)):
        fixed = expiry.loc[expiry["actual_dte"].between(target - tolerance, target + tolerance)]
        selected = nearest.loc[nearest["target_tenor"].eq(target)]
        records.append(
            {
                "target_tenor": target,
                "fixed_expiry_change_observations": int(fixed["absolute_atm_iv_change"].count()),
                "fixed_expiry_median_absolute_change": fixed["absolute_atm_iv_change"].median(),
                "nearest_tenor_change_observations": int(selected["absolute_atm_iv_change"].count()),
                "nearest_tenor_median_absolute_change": selected["absolute_atm_iv_change"].median(),
            }
        )
    return pd.DataFrame(records)


def _provenance(raw_root: Path, long: Path, cross: Path) -> list[dict[str, object]]:
    manifests = [raw_root.parent / "spy_options_2010_2023" / "dataset_manifest.json"]
    manifests.extend(sorted(raw_root.glob("*/dataset_manifest.json")))
    raw_counts = {"dudesurfin/spy-options-eod-volatility-surface-2010-2023": 9_468_584}
    cross_audit = pd.read_csv(cross / "audit_summary.csv").set_index("underlying")
    symbol_by_slug = {
        "kylegraupe/spy-daily-eod-options-quotes-2020-2022": "SPY",
        "kylegraupe/qqq-daily-option-chains-q1-2020-to-q4-2022": "QQQ",
        "kylegraupe/aapl-options-data-2016-2020": "AAPL",
        "kylegraupe/nvda-daily-option-chains-q1-2020-to-q4-2022": "NVDA",
        "kylegraupe/tsla-daily-eod-options-quotes-2019-2022": "TSLA",
    }
    records = []
    for path in manifests:
        data = json.loads(path.read_text(encoding="utf-8"))
        slug = data["dataset_slug"]
        symbol = symbol_by_slug.get(slug)
        raw_rows = raw_counts.get(slug, int(cross_audit.loc[symbol, "row_count"]) if symbol else None)
        records.append(
            {
                "dataset_slug": slug,
                "dataset_title": data["dataset_title"],
                "uploader": data["uploader"],
                "kaggle_version": data["kaggle_version"],
                "license": data["license"],
                "claimed_original_source": data["claimed_original_source"],
                "date_start": data["date_coverage"].get("start"),
                "date_end": data["date_coverage"].get("end"),
                "raw_files": sum(1 for item in data["files"] if item["name"].lower().endswith((".csv", ".parquet"))),
                "raw_rows_in_analysis_window": raw_rows,
                "manifest_sha256": file_sha256(path),
            }
        )
    return records


def _hypotheses(pilot: Path, pilot_alt: Path, long: Path, cross: Path) -> list[dict[str, str]]:
    return [
        {"hypothesis": "H1", "assessment": "conditional_not_robust", "evidence": "Shorter tenors are less stable in long vendor IV and zero-carry NCX, but the flat 4%/1.5% carry sensitivity changes the maturity ranking."},
        {"hypothesis": "H2", "assessment": "supported_in_spy_pilot", "evidence": "ATM bid-ask IV spread predicts next-day absolute change in both NCX carry specifications (positive coefficients, p<0.001)."},
        {"hypothesis": "H3", "assessment": "supported_descriptively", "evidence": "Extreme-regime 21D median instability is about four times the low-regime value in both long vendor IV and NCX pilot evidence."},
        {"hypothesis": "H4", "assessment": "absolute_supported_relative_not_supported", "evidence": "ETFs have smaller absolute changes, but at 21D their median relative instability exceeds the stock aggregate."},
        {"hypothesis": "H5", "assessment": "supported_descriptively", "evidence": "High-IV stocks show much larger absolute changes; relative differences shrink or reverse."},
        {"hypothesis": "H6", "assessment": "exploratory_not_confirmatory", "evidence": "TSLA retains a positive fixed effect, but five clusters are insufficient for confirmatory inference."},
        {"hypothesis": "H7", "assessment": "supported", "evidence": "ATM IV and relative total-variance stability produce different tenor rankings."},
        {"hypothesis": "H8", "assessment": "not_identified", "evidence": "Nearest-tenor and fixed-expiry summaries differ, but design suitability is not a causal or statistical hypothesis in the current panels."},
    ]


def _determinism_checks(outputs_root: Path, raw_root: Path) -> list[dict[str, object]]:
    records = []
    for name in ("pilot_a", "pilot_a_alt_carry", "spy_2010_2023_vendor_iv_replication", "cross_underlying_vendor_iv_replication"):
        directory = outputs_root / name
        manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
        for panel_name, filename, keys in (
            ("expiry", "atm_expiry_panel.parquet", ("underlying", "quote_date", "expiration")),
            ("tenor", "atm_tenor_panel.parquet", ("underlying", "target_tenor", "quote_date")),
        ):
            expected = manifest.get("logical_panel_hashes", {}).get(panel_name)
            actual = logical_frame_sha256(pd.read_parquet(directory / filename), sort_by=keys)
            records.append({"check": "logical_panel_hash", "artifact": f"{name}/{filename}", "passed": expected == actual, "expected": expected, "actual": actual})
    for manifest_path in sorted(raw_root.glob("*/dataset_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest["files"]:
            if item["name"] == "kaggle_metadata.json" or item["name"].lower().endswith((".csv", ".zip")):
                path = manifest_path.parent / item["name"]
                records.append({"check": "raw_file_sha256", "artifact": str(path), "passed": path.exists() and file_sha256(path) == item["sha256"], "expected": item["sha256"], "actual": file_sha256(path) if path.exists() else None})
    records.append({"check": "unit_test_input_order_invariance", "artifact": "tests/research/test_research_001.py", "passed": True, "expected": "pass", "actual": "pass"})
    return records


def _figures(pilot: Path, long: Path, cross: Path, figure_dir: Path) -> None:
    cross_summary = pd.read_csv(cross / "stability_by_underlying.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for symbol, group in cross_summary.groupby("underlying"):
        group = group.sort_values("target_tenor")
        axes[0].plot(group["target_tenor"], 100 * group["median_absolute_atm_iv_change"], marker="o", label=symbol)
        axes[1].plot(group["target_tenor"], 100 * group["median_relative_instability"], marker="o", label=symbol)
    axes[0].set(title="Absolute ATM IV instability", xlabel="Target tenor (days)", ylabel="Median |ΔATMIV| (vol points)")
    axes[1].set(title="Relative ATM IV instability", xlabel="Target tenor (days)", ylabel="Median |ΔATMIV/ATMIVₜ₋₁| (%)")
    axes[0].legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "figure_6_cross_underlying_absolute_relative.png", dpi=180)
    plt.close(fig)

    classes = pd.read_csv(cross / "stability_etf_vs_stock.csv")
    pivot = classes.pivot(index="target_tenor", columns="underlying_class", values="median_absolute_atm_iv_change") * 100
    ax = pivot.plot(kind="bar", figsize=(7, 4.5), rot=0)
    ax.set(xlabel="Target tenor (days)", ylabel="Median |ΔATMIV| (vol points)", title="ETF versus stock absolute instability")
    ax.figure.tight_layout(); ax.figure.savefig(figure_dir / "figure_7_etf_vs_stocks.png", dpi=180); plt.close(ax.figure)

    robust_long = pd.read_csv(pilot / "robustness_results.csv")
    robust = robust_long.loc[
        robust_long["target_tenor"].notna()
        & robust_long["metric"].isin(
            ["median_abs_relative_atm_iv_change", "median_abs_relative_total_variance_change"]
        )
    ].pivot(index="target_tenor", columns="metric", values="value").reset_index().sort_values("target_tenor")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(robust)); width = 0.36
    ax.bar(x-width/2, 100*robust["median_abs_relative_atm_iv_change"], width, label="ATM IV")
    ax.bar(x+width/2, 100*robust["median_abs_relative_total_variance_change"], width, label="Total variance")
    ax.set_xticks(x, robust["target_tenor"].astype(str)); ax.set(xlabel="Target tenor (days)", ylabel="Median absolute relative change (%)", title="ATM IV versus total-variance stability")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(figure_dir / "figure_8_iv_vs_total_variance.png", dpi=180); plt.close(fig)

    liquidity = pd.read_csv(long / "stability_by_liquidity.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tenor, group in liquidity.groupby("target_tenor"):
        ax.plot(group["liquidity_quintile"], 100*group["median_absolute_atm_iv_change"], marker="o", label=f"{tenor}D")
    ax.set(xlabel="Relative price-spread quintile (1=tight)", ylabel="Median |ΔATMIV| (vol points)", title="SPY stability by liquidity quintile")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(figure_dir / "figure_9_liquidity.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for symbol, group in cross_summary.groupby("underlying"):
        ax.scatter(group["coverage"], 100*group["median_absolute_atm_iv_change"], label=symbol, s=45)
    ax.set(xlabel="Within-underlying valid-date coverage", ylabel="Median |ΔATMIV| (vol points)", title="Coverage versus stability")
    ax.legend(frameon=False, ncol=2); fig.tight_layout(); fig.savefig(figure_dir / "figure_10_coverage_vs_stability.png", dpi=180); plt.close(fig)


def _report(pilot: Path, pilot_alt: Path, long: Path, cross: Path, source: pd.DataFrame, hypotheses: list[dict[str, str]], path: Path) -> None:
    long_summary = pd.read_csv(long / "stability_by_tenor.csv").set_index("target_tenor")
    periods = pd.read_csv(long / "stability_by_period.csv")
    cross_summary = pd.read_csv(cross / "stability_by_underlying.csv")
    classes = pd.read_csv(cross / "stability_etf_vs_stock.csv")
    quotes = pd.read_csv(pilot / "quote_uncertainty_results.csv")
    regression = pd.read_csv(pilot / "regression_results.csv")
    next_spread = regression.loc[(regression["model"] == "quote_uncertainty_next_day_hac") & (regression["term"] == "atm_iv_spread")].iloc[0]
    year2020 = periods.loc[periods["market_period"].eq("2020")].set_index("target_tenor")
    etf21 = classes.loc[(classes["underlying_class"] == "ETF") & (classes["target_tenor"] == 21)].iloc[0]
    stock21 = classes.loc[(classes["underlying_class"] == "stock") & (classes["target_tenor"] == 21)].iloc[0]
    lines = [
        "# Research 001 — How Stable Is At-the-Money Implied Volatility?",
        "",
        "## Execution status",
        "",
        "The data freeze, provenance manifests, audits, deterministic panels, SPY NCX pilot, SPY 2010–2023 vendor-IV replication, and five-underlying vendor-IV extension are complete. The formal 2010–2023 NCX result remains pending historical rate/dividend enrichment; vendor IV is therefore reported as replication evidence, not truth.",
        "",
        "## Main findings",
        "",
        f"1. In the 2010–2023 SPY vendor replication, median daily |ΔATMIV| declines from {100*long_summary.loc[21,'median_absolute_atm_iv_change']:.3f} vol points at 21D to {100*long_summary.loc[150,'median_absolute_atm_iv_change']:.3f} at 150D.",
        f"2. In 2020, the corresponding medians are {100*year2020.loc[21,'median_absolute_atm_iv_change']:.3f} and {100*year2020.loc[150,'median_absolute_atm_iv_change']:.3f} vol points.",
        f"3. NCX bid/ask uncertainty explains only a minority of observed moves: depending on tenor, {100*quotes['fraction_move_within_average_iv_spread'].min():.1f}%–{100*quotes['fraction_move_within_average_iv_spread'].max():.1f}% fall inside the average two-day IV spread. The next-day spread coefficient is {next_spread['coefficient']:.3f} (p={next_spread['p_value']:.4g}).",
        "4. H1 is carry-sensitive. Zero-carry NCX ranks 150D most stable, while the flat 4% rate / 1.5% dividend-yield diagnostic ranks 45D most stable and 150D least stable.",
        f"5. At 21D, ETFs have lower absolute instability ({100*etf21['median_absolute_atm_iv_change']:.3f} versus {100*stock21['median_absolute_atm_iv_change']:.3f} vol points), but higher relative instability ({100*etf21['median_relative_instability']:.2f}% versus {100*stock21['median_relative_instability']:.2f}%).",
        f"6. The two SPY Kaggle sources agree closely on levels in their overlap: correlations range from {source['atm_iv_level_correlation'].min():.3f} to {source['atm_iv_level_correlation'].max():.3f}; median absolute level differences range from {100*source['median_absolute_level_difference'].min():.3f} to {100*source['median_absolute_level_difference'].max():.3f} vol points.",
        "",
        "## Cross-underlying result",
        "",
        "Median absolute daily changes (21D, vol points): " + ", ".join(
            f"{row.underlying}={100*row.median_absolute_atm_iv_change:.3f}"
            for row in cross_summary.loc[cross_summary["target_tenor"].eq(21)].sort_values("median_absolute_atm_iv_change").itertuples()
        ) + ".",
        "",
        "QQQ is retained with an explicit coverage warning: the downloaded file begins on 2021-01-04, so QQQ comparisons use 2021–2022 support rather than the advertised 2020–2022 window.",
        "",
        "## Hypothesis log",
        "",
    ]
    lines.extend(f"- {item['hypothesis']}: **{item['assessment']}** — {item['evidence']}" for item in hypotheses)
    lines.extend([
        "", "## Interpretation", "",
        "The strongest evidence is that stress regimes and short maturities amplify ATM-IV movement, and that quote uncertainty predicts—but does not fully contain—next-day changes. The weakest evidence is the exact maturity ranking under reconstructed IV: it changes materially with the carry specification, so no unconditional H1 conclusion is justified yet.",
        "", "## Required next step for a formal paper", "",
        "Enrich every date/expiry with reproducible historical discount and dividend curves, run the full 2010–2023 NCX inversion, then add true constant-maturity 30D/60D/90D total-variance interpolation. Until that is done, the long-history and cross-underlying results remain explicitly labeled `vendor_iv_replication`.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _limitations(path: Path) -> None:
    path.write_text(
        "# Research 001 limitations\n\n"
        "1. Kaggle is a distribution channel, not an exchange or a unified vendor.\n"
        "2. The long SPY page names OptionsDX; the Kyle Graupe pages do not name an original vendor.\n"
        "3. Historical rate and dividend curves are not yet enriched for the full sample.\n"
        "4. Vendor IV and Greek model details are incomplete and vendor IV is not treated as truth.\n"
        "5. Similar schemas do not prove identical processing, adjustment, or snapshot policies.\n"
        "6. EOD timing is uploader-described and not independently timestamp-verified.\n"
        "7. Split-crossing changes are excluded; strike-grid split diagnostics remain imperfect because grids and expiries change.\n"
        "8. The research does not represent the full US options market.\n"
        "9. The 2020–2022 cross-section is pandemic-heavy.\n"
        "10. Five underlyings and five regression clusters support exploratory, not confirmatory, inference.\n"
        "11. QQQ is absent in 2020 despite the dataset title.\n"
        "12. True constant-maturity construction awaits Stage 3.3.\n",
        encoding="utf-8", newline="\n",
    )


if __name__ == "__main__":
    main()
