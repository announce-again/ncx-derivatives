"""Build research smiles directly from the Stage 3.1 pipeline output."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from time import perf_counter

from ncx_derivatives.volatility import (
    SMILE_GROUP_DIAGNOSTIC_COLUMNS,
    SMILE_POINT_COLUMNS,
    SMILE_SELECTION_DIAGNOSTIC_COLUMNS,
    SmileAnalysisConfig,
    SmileAnalysisMetric,
    SmileAnalysisResult,
    SyntheticOptionDatasetConfig,
    build_volatility_smiles,
    build_volatility_term_structures,
    calculate_smile_delta_metrics_for_smiles,
    calculate_smile_metrics_for_smiles,
    run_csv_volatility_pipeline,
    smile_group_diagnostics_to_records,
    smile_selection_diagnostics_to_records,
    synthetic_option_quote_csv_config,
    synthetic_volatility_pipeline_carry,
    synthetic_volatility_pipeline_cleaning_config,
    volatility_smiles_to_records,
    write_smile_analysis_csv,
    write_synthetic_option_quote_csv,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".tmp/examples_output/volatility_smiles"),
    )
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    config = SyntheticOptionDatasetConfig(
        row_count=arguments.rows,
        seed=arguments.seed,
    )
    source_path = arguments.output_dir / "synthetic_option_quotes.csv"
    smile_path = arguments.output_dir / "volatility_smile_points.csv"
    diagnostic_path = arguments.output_dir / "smile_selection_diagnostics.csv"
    group_diagnostic_path = (
        arguments.output_dir / "smile_group_diagnostics.csv"
    )
    full_started = perf_counter()
    generated = write_synthetic_option_quote_csv(source_path, config)
    pipeline = run_csv_volatility_pipeline(
        generated.path,
        ingestion_config=synthetic_option_quote_csv_config(),
        carry=synthetic_volatility_pipeline_carry(config),
        valuation_date=config.valuation_date,
        cleaning_config=synthetic_volatility_pipeline_cleaning_config(),
    )

    selection_started = perf_counter()
    selection = build_volatility_smiles(pipeline.implied_volatility_chain)
    selection_seconds = perf_counter() - selection_started
    point_records = volatility_smiles_to_records(selection.smiles)
    diagnostic_records = smile_selection_diagnostics_to_records(
        selection.diagnostics,
    )
    group_diagnostic_records = smile_group_diagnostics_to_records(
        selection.group_diagnostics,
    )
    metrics_started = perf_counter()
    metrics = calculate_smile_metrics_for_smiles(selection.smiles)
    metrics_seconds = perf_counter() - metrics_started
    delta_started = perf_counter()
    delta_metrics = calculate_smile_delta_metrics_for_smiles(
        selection.smiles,
        local_metric_results=metrics,
    )
    delta_seconds = perf_counter() - delta_started
    term_started = perf_counter()
    term_structures = build_volatility_term_structures(metrics, delta_metrics)
    term_seconds = perf_counter() - term_started
    analysis = SmileAnalysisResult(
        local_metrics=metrics,
        delta_metrics=delta_metrics,
        term_structures=term_structures,
        config=SmileAnalysisConfig(),
    )
    _write_records(smile_path, SMILE_POINT_COLUMNS, point_records)
    _write_records(
        diagnostic_path,
        SMILE_SELECTION_DIAGNOSTIC_COLUMNS,
        diagnostic_records,
    )
    _write_records(
        group_diagnostic_path,
        SMILE_GROUP_DIAGNOSTIC_COLUMNS,
        group_diagnostic_records,
    )
    analysis_exports = write_smile_analysis_csv(
        arguments.output_dir / "analysis",
        analysis,
    )
    generation_to_export_seconds = perf_counter() - full_started

    summary = selection.summary
    reason_counts = Counter(
        reason.value
        for diagnostic in selection.diagnostics
        for reason in diagnostic.reasons
    )
    selection_throughput = (
        0.0
        if selection_seconds == 0.0
        else summary.input_quote_count / selection_seconds
    )
    observed_atm_count = sum(
        smile.has_observed_atm_point for smile in selection.smiles
    )
    metric_throughput = (
        0.0 if metrics_seconds == 0.0 else len(metrics) / metrics_seconds
    )
    delta_throughput = (
        0.0
        if delta_seconds == 0.0
        else len(delta_metrics) / delta_seconds
    )
    term_throughput = (
        0.0
        if term_seconds == 0.0
        else sum(len(term.points) for term in term_structures) / term_seconds
    )
    generation_to_export_throughput = (
        0.0
        if generation_to_export_seconds == 0.0
        else generated.row_count / generation_to_export_seconds
    )
    analysis_summary = analysis.summary

    def outcome(metric, magnitude=None):
        count = analysis_summary.outcome(metric, magnitude)
        return {
            "success": count.success_count,
            "failure": count.failure_count,
        }

    print(f"Generated input rows: {generated.row_count:,}")
    print(f"Stage 3.1 IV quotes: {pipeline.counts.iv_quote_count:,}")
    print(
        "Stage 3.2 smiles: "
        f"smiles={summary.smile_count:,}, "
        f"points={summary.selected_point_count:,}, "
        f"excluded={summary.excluded_quote_count:,}, "
        f"empty={summary.empty_smile_count:,}, "
        f"observed_atm={observed_atm_count:,}",
    )
    print(f"Selection diagnostics: {dict(sorted(reason_counts.items()))}")
    print(
        "Group diagnostics: "
        f"groups={summary.group_diagnostic_count:,}, "
        f"rejected_quotes={summary.group_rejected_quote_count:,}",
    )
    print(
        "Stage 3.2 analysis: "
        f"local_results={analysis_summary.local_metric_result_count:,}, "
        f"delta_results={analysis_summary.delta_metric_result_count:,}, "
        f"term_structures={analysis_summary.term_structure_count:,}, "
        f"term_points={analysis_summary.term_structure_point_count:,}",
    )
    print(f"ATM: {outcome(SmileAnalysisMetric.ATM)}")
    print(f"Skew: {outcome(SmileAnalysisMetric.SKEW)}")
    print(f"Curvature: {outcome(SmileAnalysisMetric.CURVATURE)}")
    print(
        "25-delta call: "
        f"{outcome(SmileAnalysisMetric.DELTA_CALL, 0.25)}",
    )
    print(
        "25-delta put: "
        f"{outcome(SmileAnalysisMetric.DELTA_PUT, 0.25)}",
    )
    print(
        "RR25: "
        f"{outcome(SmileAnalysisMetric.RISK_REVERSAL, 0.25)}",
    )
    print(
        "BF25: "
        f"{outcome(SmileAnalysisMetric.BUTTERFLY, 0.25)}",
    )
    print(f"Pipeline throughput: {pipeline.input_rows_per_second:,.0f} rows/second")
    print(
        "Smile selection throughput: "
        f"{selection_throughput:,.0f} IV quotes/second",
    )
    print(
        "Smile metrics throughput: "
        f"{metric_throughput:,.0f} smiles/second",
    )
    print(
        "Delta metrics throughput: "
        f"{delta_throughput:,.0f} smiles/second",
    )
    print(
        "Term-structure assembly throughput: "
        f"{term_throughput:,.0f} expiry results/second",
    )
    print(
        "Generation-to-export throughput: "
        f"{generation_to_export_throughput:,.0f} rows/second",
    )
    print(f"Smile points: {smile_path}")
    print(f"Selection diagnostics: {diagnostic_path}")
    print(f"Group diagnostics: {group_diagnostic_path}")
    for export in analysis_exports.exports:
        print(
            f"{export.name}: {export.path} "
            f"({export.row_count:,} rows, sha256={export.sha256})",
        )


def _write_records(
    path: Path,
    columns: tuple[str, ...],
    records: tuple[dict[str, object], ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
