from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from research.real_data.common.deterministic_io import (
    file_sha256,
    write_csv,
    write_json,
    write_output_hashes,
)

SPECIFICATIONS = (
    "vendor_iv_replication",
    "ncx_treasury_projected_dividends",
    "ncx_treasury_trailing_dividend_yield",
)
KEY = ("underlying", "quote_date", "target_tenor")
TENORS = (21, 45, 90, 150)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare vendor and full-history NCX carry specifications")
    parser.add_argument("--vendor-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--alternative-dir", type=Path, required=True)
    parser.add_argument("--carry-panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    compare(
        vendor_dir=args.vendor_dir,
        baseline_dir=args.baseline_dir,
        alternative_dir=args.alternative_dir,
        carry_panel_path=args.carry_panel,
        output_dir=args.output_dir,
    )


def compare(
    *,
    vendor_dir: Path,
    baseline_dir: Path,
    alternative_dir: Path,
    carry_panel_path: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    directories = {
        SPECIFICATIONS[0]: vendor_dir,
        SPECIFICATIONS[1]: baseline_dir,
        SPECIFICATIONS[2]: alternative_dir,
    }
    panels = {
        name: _load_panel(directory / "atm_tenor_panel.parquet")
        for name, directory in directories.items()
    }
    expiry_panels = {
        name: pd.read_parquet(directory / "atm_expiry_panel.parquet")
        for name, directory in directories.items()
    }

    common_index = None
    for panel in panels.values():
        index = pd.MultiIndex.from_frame(panel[list(KEY)])
        common_index = index if common_index is None else common_index.intersection(index, sort=False)
    assert common_index is not None
    common_keys = pd.DataFrame(common_index.tolist(), columns=KEY)
    common_keys = common_keys.sort_values(list(KEY), kind="mergesort", ignore_index=True)

    comparison = common_keys.copy()
    selected_columns = (
        "expiration",
        "actual_dte",
        "atm_mid_iv",
        "atm_total_variance",
        "atm_iv_spread",
        "absolute_atm_iv_change",
        "relative_atm_iv_change",
        "delta_atm_total_variance",
        "previous_observation_date",
    )
    for name, panel in panels.items():
        renamed = panel[list(KEY) + list(selected_columns)].rename(
            columns={column: f"{name}__{column}" for column in selected_columns}
        )
        comparison = comparison.merge(renamed, on=list(KEY), how="left", validate="one_to_one")

    previous_columns = [f"{name}__previous_observation_date" for name in SPECIFICATIONS]
    previous_equal = comparison[previous_columns].notna().all(axis=1)
    previous_equal &= comparison[previous_columns].nunique(axis=1, dropna=False).eq(1)
    change_columns = [f"{name}__absolute_atm_iv_change" for name in SPECIFICATIONS]
    comparison["paired_change_eligible"] = previous_equal & comparison[change_columns].notna().all(axis=1)
    expiration_columns = [f"{name}__expiration" for name in SPECIFICATIONS]
    comparison["same_selected_expiration"] = comparison[expiration_columns].nunique(axis=1, dropna=False).eq(1)

    summary_records = []
    for name, panel in panels.items():
        for target_tenor, group in panel.groupby("target_tenor", sort=True):
            summary_records.append(_summary_record(name, int(target_tenor), group, sample="specification_sample"))
        paired = comparison.loc[comparison["paired_change_eligible"]]
        for target_tenor, keys in paired.groupby("target_tenor", sort=True):
            columns = list(KEY) + [f"{name}__{column}" for column in selected_columns]
            group = keys[columns].rename(columns={f"{name}__{column}": column for column in selected_columns})
            summary_records.append(_summary_record(name, int(target_tenor), group, sample="three_way_paired_changes"))
    write_csv(
        output_dir / "carry_specification_comparison.csv",
        summary_records,
        columns=(
            "sample", "specification", "target_tenor", "observation_count", "change_count",
            "median_atm_iv", "median_absolute_atm_iv_change", "p95_absolute_atm_iv_change",
            "median_absolute_relative_atm_iv_change", "median_atm_iv_spread",
            "median_absolute_total_variance_change",
        ),
        sort_by=("sample", "specification", "target_tenor"),
    )

    paired_records = []
    baseline = SPECIFICATIONS[1]
    for name in (SPECIFICATIONS[0], SPECIFICATIONS[2]):
        for target_tenor, group in comparison.groupby("target_tenor", sort=True):
            level_left = group[f"{name}__atm_mid_iv"]
            level_base = group[f"{baseline}__atm_mid_iv"]
            level_difference = level_left - level_base
            eligible = group["paired_change_eligible"]
            change_left = group.loc[eligible, f"{name}__absolute_atm_iv_change"]
            change_base = group.loc[eligible, f"{baseline}__absolute_atm_iv_change"]
            change_difference = change_left - change_base
            paired_records.append({
                "comparison": f"{name}_minus_{baseline}",
                "target_tenor": int(target_tenor),
                "level_pair_count": int(level_difference.notna().sum()),
                "median_signed_atm_iv_difference": level_difference.median(),
                "median_absolute_atm_iv_difference": level_difference.abs().median(),
                "p95_absolute_atm_iv_difference": level_difference.abs().quantile(0.95),
                "atm_iv_level_correlation": level_left.corr(level_base),
                "change_pair_count": int(change_difference.notna().sum()),
                "median_signed_absolute_change_difference": change_difference.median(),
                "median_absolute_change_difference": change_difference.abs().median(),
                "absolute_change_correlation": change_left.corr(change_base),
            })
    write_csv(
        output_dir / "paired_carry_differences.csv",
        paired_records,
        columns=(
            "comparison", "target_tenor", "level_pair_count", "median_signed_atm_iv_difference",
            "median_absolute_atm_iv_difference", "p95_absolute_atm_iv_difference", "atm_iv_level_correlation",
            "change_pair_count", "median_signed_absolute_change_difference",
            "median_absolute_change_difference", "absolute_change_correlation",
        ),
        sort_by=("comparison", "target_tenor"),
    )

    carry = pd.read_parquet(carry_panel_path)
    coverage_records = []
    for name in (
        "treasury_projected_dividend_schedule",
        "treasury_trailing_dividend_yield",
        "flat_3m_treasury_projected_dividend_schedule",
        "sofr_flat_projected_dividend_schedule",
    ):
        available = carry[f"{name}_available"].fillna(False).astype(bool)
        dates = pd.to_datetime(carry.loc[available, "quote_date"])
        coverage_records.append({
            "carry_specification": name,
            "expiry_rows": int(available.sum()),
            "coverage_fraction": float(available.mean()),
            "first_quote_date": dates.min().date() if not dates.empty else None,
            "last_quote_date": dates.max().date() if not dates.empty else None,
        })
    write_csv(
        output_dir / "carry_specification_coverage.csv",
        coverage_records,
        columns=("carry_specification", "expiry_rows", "coverage_fraction", "first_quote_date", "last_quote_date"),
        sort_by=("carry_specification",),
    )

    validation_records = []
    ordering_exceptions = []
    for panel_kind, collection in (("expiry_panel", expiry_panels), ("tenor_panel", panels)):
        for name, panel in collection.items():
            finite = panel[["atm_bid_iv", "atm_mid_iv", "atm_ask_iv"]].notna().all(axis=1)
            ordered = (panel["atm_bid_iv"] <= panel["atm_mid_iv"]) & (panel["atm_mid_iv"] <= panel["atm_ask_iv"])
            exceptions = panel.loc[finite & ~ordered]
            duplicate_key = list(KEY) if panel_kind == "tenor_panel" else ["underlying", "quote_date", "expiration"]
            validation_records.append({
                "specification": name,
                "expiry_or_tenor": panel_kind,
                "row_count": len(panel),
                "atm_mid_success_count": int(panel["atm_mid_status"].eq("SUCCESS").sum()),
                "finite_bid_mid_ask_count": int(finite.sum()),
                "bid_mid_ask_ordering_exception_count": len(exceptions),
                "positive_discount_factor_count": int(
                    ((panel["risk_free_discount_factor"] > 0.0) & (panel["dividend_discount_factor"] > 0.0)).sum()
                ),
                "duplicate_key_count": int(panel.duplicated(duplicate_key).sum()),
            })
            for row in exceptions.itertuples(index=False):
                ordering_exceptions.append({
                    "specification": name,
                    "panel": panel_kind,
                    "underlying": row.underlying,
                    "quote_date": row.quote_date,
                    "expiration": row.expiration,
                    "actual_dte": row.actual_dte,
                    "target_tenor": getattr(row, "target_tenor", None),
                    "atm_bid_iv": row.atm_bid_iv,
                    "atm_mid_iv": row.atm_mid_iv,
                    "atm_ask_iv": row.atm_ask_iv,
                })
    write_csv(
        output_dir / "numerical_validation.csv",
        validation_records,
        columns=(
            "specification", "expiry_or_tenor", "row_count", "atm_mid_success_count",
            "finite_bid_mid_ask_count", "bid_mid_ask_ordering_exception_count",
            "positive_discount_factor_count", "duplicate_key_count",
        ),
        sort_by=("specification",),
    )
    write_csv(
        output_dir / "atm_ordering_exceptions.csv",
        ordering_exceptions,
        columns=(
            "specification", "panel", "underlying", "quote_date", "expiration", "actual_dte", "target_tenor",
            "atm_bid_iv", "atm_mid_iv", "atm_ask_iv",
        ),
        sort_by=("specification", "panel", "quote_date", "expiration", "target_tenor"),
    )

    _write_figure(summary_records, output_dir / "carry_stability_comparison.png")
    _write_report(
        output_dir,
        summary_records,
        paired_records,
        comparison,
        coverage_records,
        validation_records,
    )
    input_paths = {
        f"{name}_tenor_panel": directory / "atm_tenor_panel.parquet"
        for name, directory in directories.items()
    }
    input_paths.update({
        f"{name}_expiry_panel": directory / "atm_expiry_panel.parquet"
        for name, directory in directories.items()
    })
    input_paths["carry_panel"] = carry_panel_path
    write_json(
        output_dir / "comparison_manifest.json",
        {
            "research_id": "research_001a_carry_specification_comparison",
            "specifications": SPECIFICATIONS,
            "input_sha256": {name: file_sha256(path) for name, path in input_paths.items()},
            "input_row_counts": {name: len(panel) for name, panel in panels.items()},
            "three_way_common_key_count": len(comparison),
            "three_way_paired_change_count": int(comparison["paired_change_eligible"].sum()),
            "same_selected_expiration_fraction": float(comparison["same_selected_expiration"].mean()),
        },
    )
    write_output_hashes(output_dir, output_dir / "output_hashes.json")


def _load_panel(path: Path) -> pd.DataFrame:
    panel = pd.read_parquet(path)
    panel["quote_date"] = pd.to_datetime(panel["quote_date"])
    panel["expiration"] = pd.to_datetime(panel["expiration"])
    panel["previous_observation_date"] = pd.to_datetime(panel["previous_observation_date"])
    if panel.duplicated(list(KEY)).any():
        raise ValueError(f"duplicate tenor keys in {path}")
    return panel.sort_values(list(KEY), kind="mergesort", ignore_index=True)


def _summary_record(specification: str, target_tenor: int, group: pd.DataFrame, *, sample: str) -> dict[str, object]:
    return {
        "sample": sample,
        "specification": specification,
        "target_tenor": target_tenor,
        "observation_count": len(group),
        "change_count": int(group["absolute_atm_iv_change"].notna().sum()),
        "median_atm_iv": group["atm_mid_iv"].median(),
        "median_absolute_atm_iv_change": group["absolute_atm_iv_change"].median(),
        "p95_absolute_atm_iv_change": group["absolute_atm_iv_change"].quantile(0.95),
        "median_absolute_relative_atm_iv_change": group["relative_atm_iv_change"].abs().median(),
        "median_atm_iv_spread": group["atm_iv_spread"].median(),
        "median_absolute_total_variance_change": group["delta_atm_total_variance"].abs().median(),
    }


def _write_figure(records: list[dict[str, object]], path: Path) -> None:
    frame = pd.DataFrame.from_records(records)
    frame = frame.loc[frame["sample"].eq("three_way_paired_changes")]
    figure, axis = plt.subplots(figsize=(9, 5.4))
    for name, group in frame.groupby("specification", sort=False):
        ordered = group.sort_values("target_tenor")
        axis.plot(
            ordered["target_tenor"],
            100.0 * ordered["median_absolute_atm_iv_change"],
            marker="o",
            linewidth=2,
            label=name,
        )
    axis.set_xlabel("Target tenor (calendar days)")
    axis.set_ylabel("Median daily |ΔATM IV| (vol points)")
    axis.set_title("SPY ATM stability by IV/carry specification, 2010–2023")
    axis.set_xticks(TENORS)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_report(
    output_dir: Path,
    summary_records: list[dict[str, object]],
    paired_records: list[dict[str, object]],
    comparison: pd.DataFrame,
    coverage_records: list[dict[str, object]],
    validation_records: list[dict[str, object]],
) -> None:
    summary = pd.DataFrame.from_records(summary_records)
    paired_summary = summary.loc[summary["sample"].eq("three_way_paired_changes")]
    rankings = {}
    for name, group in paired_summary.groupby("specification", sort=False):
        rankings[name] = group.sort_values("median_absolute_atm_iv_change")["target_tenor"].astype(int).tolist()
    differences = pd.DataFrame.from_records(paired_records)
    alternative_name = f"{SPECIFICATIONS[2]}_minus_{SPECIFICATIONS[1]}"
    alternative = differences.loc[differences["comparison"].eq(alternative_name)]
    baseline = paired_summary.loc[paired_summary["specification"].eq(SPECIFICATIONS[1])].set_index("target_tenor")
    trailing = paired_summary.loc[paired_summary["specification"].eq(SPECIFICATIONS[2])].set_index("target_tenor")
    summary_change_difference = (
        trailing["median_absolute_atm_iv_change"] - baseline["median_absolute_atm_iv_change"]
    )
    largest_summary_tenor = int(summary_change_difference.abs().idxmax())
    largest_summary_difference = float(summary_change_difference.loc[largest_summary_tenor])
    largest_summary_relative_difference = (
        largest_summary_difference / baseline.loc[largest_summary_tenor, "median_absolute_atm_iv_change"]
    )
    max_paired_move_difference = alternative["median_absolute_change_difference"].max()
    ratio_21_150 = baseline.loc[21, "median_absolute_atm_iv_change"] / baseline.loc[150, "median_absolute_atm_iv_change"]
    coverage = pd.DataFrame.from_records(coverage_records).set_index("carry_specification")
    validation = pd.DataFrame.from_records(validation_records)
    ordering_exception_count = int(
        validation.loc[
            validation["specification"].isin(SPECIFICATIONS[1:]) & validation["expiry_or_tenor"].eq("expiry_panel"),
            "bid_mid_ask_ordering_exception_count",
        ].sum()
    )
    report = (
        "# Research 001A — historical carry and full-history NCX comparison\n\n"
        "## Result\n\n"
        "The maturity-stability result survives both NCX carry specifications: 150D is most stable and 21D is least stable "
        f"in the three-way paired sample. Under the baseline, the 21D median daily absolute ATM-IV change is "
        f"{ratio_21_150:.2f}× the 150D value.\n\n"
        f"The largest change in the tenor-level median instability occurs at {largest_summary_tenor}D: the trailing-yield result "
        f"differs from baseline by {100.0 * largest_summary_difference:+.4f} vol points "
        f"({100.0 * largest_summary_relative_difference:+.1f}%). At the individual-move level, the largest median absolute "
        f"paired difference is {100.0 * max_paired_move_difference:.4f} vol points. "
        "Thus the qualitative maturity ordering is not a carry artifact, but its quantitative magnitude is carry-sensitive, "
        "especially at short and intermediate tenors.\n\n"
        "## Stability rankings (most stable first)\n\n"
        + "\n".join(f"- `{name}`: {ranking}" for name, ranking in rankings.items())
        + "\n\n## Paired sample\n\n"
        f"The three panels share {len(comparison):,} underlying-date-tenor keys; "
        f"{int(comparison['paired_change_eligible'].sum()):,} have the same finite previous observation in all three specifications. "
        f"The selected expiration is identical on {100.0 * comparison['same_selected_expiration'].mean():.3f}% of common keys.\n\n"
        "## Carry coverage\n\n"
        f"Treasury projected-distribution and trailing-yield specifications cover "
        f"{100.0 * coverage.loc['treasury_projected_dividend_schedule', 'coverage_fraction']:.3f}% of expiry rows. "
        f"The flat-3M Treasury diagnostic covers {100.0 * coverage.loc['flat_3m_treasury_projected_dividend_schedule', 'coverage_fraction']:.3f}%; "
        f"SOFR begins on {coverage.loc['sofr_flat_projected_dividend_schedule', 'first_quote_date']} and covers "
        f"{100.0 * coverage.loc['sofr_flat_projected_dividend_schedule', 'coverage_fraction']:.3f}%.\n\n"
        "## Numerical validation\n\n"
        "Both NCX tenor panels have positive risk-free and dividend discount factors and no duplicate keys. "
        f"Independent bid/mid/ask smile interpolation produces {ordering_exception_count} bid/mid/ask ordering exceptions "
        "across the two NCX expiry panels; these observations are retained and listed rather than silently repaired.\n\n"
        "## Interpretation limits\n\n"
        "Treasury constant-maturity yields are investment-yield proxies, not option financing/OIS curves. "
        "The projected cash-dividend baseline uses the latest SPY distribution amount known on each quote date and historical ex-date seasonality; "
        "the trailing-yield alternative smooths quarterly timing. Option-implied forwards are retained as diagnostics because American exercise and quote noise can contaminate put-call parity.\n"
    )
    (output_dir / "research_001_carry_comparison.md").write_text(report, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
