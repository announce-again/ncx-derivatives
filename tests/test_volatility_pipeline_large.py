from time import perf_counter

import pytest

from ncx_derivatives.volatility import (
    SmileAnalysisConfig,
    SmileAnalysisMetric,
    SmileAnalysisResult,
    SyntheticOptionDatasetConfig,
    analyze_volatility_smiles,
    build_volatility_smiles,
    build_volatility_term_structures,
    calculate_smile_delta_metrics_for_smiles,
    calculate_smile_metrics_for_smiles,
    delta_structure_results_to_records,
    delta_volatility_results_to_records,
    run_csv_volatility_pipeline,
    synthetic_option_quote_csv_config,
    synthetic_volatility_pipeline_carry,
    synthetic_volatility_pipeline_cleaning_config,
    volatility_term_structures_to_records,
    write_implied_volatility_chain_csv,
    write_smile_analysis_csv,
    write_synthetic_option_quote_csv,
)


@pytest.mark.large
def test_large_50k_csv_to_iv_and_smile_pipeline_smoke_and_throughput(
    tmp_path,
    record_property,
) -> None:
    config = SyntheticOptionDatasetConfig(row_count=50_000)
    generation_started = perf_counter()
    source = write_synthetic_option_quote_csv(tmp_path / "quotes-50k.csv", config)

    result = run_csv_volatility_pipeline(
        source.path,
        ingestion_config=synthetic_option_quote_csv_config(),
        carry=synthetic_volatility_pipeline_carry(config),
        valuation_date=config.valuation_date,
        cleaning_config=synthetic_volatility_pipeline_cleaning_config(),
    )
    exported = write_implied_volatility_chain_csv(
        tmp_path / "iv-chain-50k.csv",
        result.implied_volatility_chain,
    )
    smile_started = perf_counter()
    smile_selection = build_volatility_smiles(result.implied_volatility_chain)
    smile_seconds = perf_counter() - smile_started
    metrics_started = perf_counter()
    metrics = calculate_smile_metrics_for_smiles(smile_selection.smiles)
    metrics_seconds = perf_counter() - metrics_started
    delta_started = perf_counter()
    delta_metrics = calculate_smile_delta_metrics_for_smiles(
        smile_selection.smiles,
        local_metric_results=metrics,
    )
    delta_seconds = perf_counter() - delta_started
    term_started = perf_counter()
    term_structures = build_volatility_term_structures(
        metrics,
        delta_metrics,
    )
    term_seconds = perf_counter() - term_started
    analysis = SmileAnalysisResult(
        local_metrics=metrics,
        delta_metrics=delta_metrics,
        term_structures=term_structures,
        config=SmileAnalysisConfig(),
    )
    first_exports = write_smile_analysis_csv(
        tmp_path / "analysis-first",
        analysis,
    )
    generation_to_export_seconds = perf_counter() - generation_started
    second_exports = write_smile_analysis_csv(
        tmp_path / "analysis-second",
        analysis,
    )
    reversed_analysis = analyze_volatility_smiles(
        reversed(smile_selection.smiles),
    )
    pipeline_throughput = result.input_rows_per_second
    generation_to_export_throughput = (
        config.row_count / generation_to_export_seconds
    )
    record_property("rows", config.row_count)
    record_property("pipeline_seconds", result.timings.total_seconds)
    record_property("pipeline_rows_per_second", pipeline_throughput)
    record_property("smile_selection_seconds", smile_seconds)
    record_property(
        "smile_selected_points",
        smile_selection.summary.selected_point_count,
    )
    record_property("smile_metrics_seconds", metrics_seconds)
    record_property("smile_metric_result_count", len(metrics))
    record_property("delta_metrics_seconds", delta_seconds)
    record_property("delta_metric_result_count", len(delta_metrics))
    record_property("term_structure_seconds", term_seconds)
    record_property("term_structure_count", len(term_structures))
    record_property(
        "generation_to_export_seconds",
        generation_to_export_seconds,
    )
    record_property(
        "generation_to_export_rows_per_second",
        generation_to_export_throughput,
    )

    assert source.ingestion_bad_row_count == 5
    assert source.cleaning_bad_row_count == 10
    assert source.iv_bad_row_count == 5
    assert result.counts.input_row_count == 50_000
    assert result.counts.ingestion_failed_row_count == 5
    assert result.counts.snapshot_count == 100
    assert result.counts.enriched_quote_count == 49_995
    assert result.counts.cleaning_rejected_count == 10
    assert result.counts.cleaning_accepted_count == 49_985
    assert result.counts.iv_quote_count == 49_985
    assert result.counts.iv_failure_count == 15
    assert exported.row_count == 49_985
    assert exported.byte_count > 0
    assert smile_selection.summary.smile_count == 500
    assert smile_selection.summary.empty_smile_count == 0
    assert smile_selection.summary.group_diagnostic_count == 0
    assert all(
        smile.nearest_atm_point is not None
        for smile in smile_selection.smiles
    )
    assert (
        smile_selection.summary.selected_point_count
        + smile_selection.summary.excluded_quote_count
        == result.counts.iv_quote_count
    )
    assert len(metrics) == smile_selection.summary.smile_count == 500
    assert all(metric.atm.is_success for metric in metrics)
    assert all(metric.skew.is_success for metric in metrics)
    assert all(metric.curvature.is_success for metric in metrics)
    assert len(delta_metrics) == len(metrics) == 500
    assert sum(len(structure.points) for structure in term_structures) == 500
    assert len(term_structures) == 100
    summary = analysis.summary
    assert summary.input_smile_count == 500
    assert summary.local_metric_result_count == 500
    assert summary.delta_metric_result_count == 500
    assert summary.term_structure_point_count == 500
    for metric in (
        SmileAnalysisMetric.ATM,
        SmileAnalysisMetric.SKEW,
        SmileAnalysisMetric.CURVATURE,
    ):
        count = summary.outcome(metric)
        assert count.success_count + count.failure_count == 500
    for metric in (
        SmileAnalysisMetric.DELTA_CALL,
        SmileAnalysisMetric.DELTA_PUT,
        SmileAnalysisMetric.RISK_REVERSAL,
        SmileAnalysisMetric.BUTTERFLY,
    ):
        count = summary.outcome(metric, 0.25)
        assert count.success_count + count.failure_count == 500
    assert delta_volatility_results_to_records(delta_metrics) == (
        delta_volatility_results_to_records(reversed_analysis.delta_metrics)
    )
    assert delta_structure_results_to_records(delta_metrics) == (
        delta_structure_results_to_records(reversed_analysis.delta_metrics)
    )
    assert volatility_term_structures_to_records(term_structures) == (
        volatility_term_structures_to_records(
            reversed_analysis.term_structures,
        )
    )
    for first_export, second_export in zip(
        first_exports.exports,
        second_exports.exports,
    ):
        assert first_export.row_count == second_export.row_count
        assert first_export.sha256 == second_export.sha256
        assert first_export.path.read_bytes() == second_export.path.read_bytes()
    assert pipeline_throughput > 100.0
    assert generation_to_export_throughput > 100.0
