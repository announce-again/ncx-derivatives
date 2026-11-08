from dataclasses import replace
from datetime import datetime, timedelta, timezone
from math import exp

import pytest

from ncx_derivatives.market_data import (
    CarryAssumptions,
    ExerciseStyle,
    FlatDividendYieldCurve,
    FlatZeroRateCurve,
    OptionChainSnapshot,
    OptionContract,
    OptionQuote,
    OptionType,
    UnderlyingQuote,
    enrich_option_chain_snapshot,
)
from ncx_derivatives.pricing import call_price, put_price
from ncx_derivatives.volatility import (
    BUTTERFLY_COLUMNS,
    DELTA_STRUCTURE_COLUMNS,
    DELTA_VOLATILITY_COLUMNS,
    RISK_REVERSAL_COLUMNS,
    TERM_STRUCTURE_COLUMNS,
    DeltaInterpolationMethod,
    DeltaMetricConfig,
    DeltaMetricFailureReason,
    DeltaStructureFailureReason,
    DuplicateTermStructurePolicy,
    SmileAnalysisConfig,
    SmileAnalysisMetric,
    SmileMetricStatus,
    SmilePointDiagnosticFlag,
    SmileSelectionConfig,
    SyntheticOptionDatasetConfig,
    VolatilityTermStructure,
    analyze_volatility_smiles,
    build_implied_volatility_chain,
    build_volatility_smiles,
    butterfly_results_to_dataframe,
    butterfly_results_to_records,
    calculate_risk_reversal,
    calculate_smile_delta_metrics,
    calculate_smile_metrics,
    calculate_symmetric_delta_butterfly,
    delta_structure_results_to_dataframe,
    delta_structure_results_to_records,
    delta_volatility_results_to_dataframe,
    delta_volatility_results_to_records,
    interpolate_smile_at_delta,
    risk_reversal_results_to_dataframe,
    risk_reversal_results_to_records,
    run_csv_volatility_pipeline,
    smile_analysis_summary_to_records,
    synthetic_option_quote_csv_config,
    synthetic_volatility_pipeline_carry,
    synthetic_volatility_pipeline_cleaning_config,
    volatility_term_structures_to_dataframe,
    volatility_term_structures_to_records,
    write_smile_analysis_csv,
    write_synthetic_option_quote_csv,
)
from ncx_derivatives.volatility import delta_metrics as delta_metrics_module

UTC = timezone.utc
AS_OF = datetime(2026, 7, 30, 14, 30, tzinfo=UTC)


def _smile(
    specs,
    *,
    symbol="AAPL",
    as_of=AS_OF,
    expiration_days=365,
):
    expiration = as_of.date() + timedelta(days=expiration_days)
    maturity = expiration_days / 365.0
    quotes = []
    ordered_specs = sorted(specs, key=lambda spec: spec[0])
    for index, (coordinate, option_type, _delta, volatility) in enumerate(
        ordered_specs,
    ):
        strike = 100.0 * exp(coordinate)
        price_function = (
            call_price if option_type is OptionType.CALL else put_price
        )
        price = price_function(100.0, strike, maturity, 0.0, volatility)
        quotes.append(
            OptionQuote(
                contract=OptionContract(
                    underlying_symbol=symbol,
                    expiration=expiration,
                    strike=strike,
                    option_type=option_type,
                    exercise_style=ExerciseStyle.EUROPEAN,
                    source_contract_id=f"delta-{index}",
                ),
                quote_timestamp=as_of,
                bid=price,
                ask=price,
                bid_size=100,
                ask_size=100,
            ),
        )
    snapshot = OptionChainSnapshot(
        underlying_symbol=symbol,
        as_of=as_of,
        quotes=tuple(quotes),
        underlying_quote=UnderlyingQuote(
            symbol=symbol,
            quote_timestamp=as_of,
            price=100.0,
        ),
    )
    enriched = enrich_option_chain_snapshot(
        snapshot,
        carry=CarryAssumptions(
            risk_free_curve=FlatZeroRateCurve(0.0),
            dividend_curve=FlatDividendYieldCurve(0.0),
        ),
        valuation_date=as_of.date(),
    )
    selected = build_volatility_smiles(
        build_implied_volatility_chain(enriched),
        SmileSelectionConfig(otm_only=False),
    )
    assert len(selected.smiles) == 1
    smile = selected.smiles[0]
    points = []
    for point, spec in zip(smile.points, ordered_specs):
        delta = spec[2]
        flags = tuple(
            flag
            for flag in point.diagnostic_flags
            if flag
            not in {
                SmilePointDiagnosticFlag.DELTA_BOUNDARY_UNAVAILABLE,
                SmilePointDiagnosticFlag.DELTA_NUMERICAL_FAILURE,
            }
        )
        if delta is None:
            flags += (SmilePointDiagnosticFlag.DELTA_NUMERICAL_FAILURE,)
        points.append(
            replace(
                point,
                implied_volatility=spec[3],
                delta=delta,
                diagnostic_flags=flags,
            ),
        )
    return _replace_points(smile, points)


def _replace_points(smile, points):
    values = tuple(points)
    nearest = (
        None
        if not values
        else min(
            range(len(values)),
            key=lambda index: (
                abs(values[index].log_forward_moneyness),
                values[index].strike,
                values[index].option_type.value,
            ),
        )
    )
    return replace(smile, points=values, nearest_atm_index=nearest)


def _known_rr_bf_smile(*, all_positive_coordinates=False):
    put_k = 0.02 if all_positive_coordinates else -0.08
    return _smile(
        (
            (put_k, OptionType.PUT, -0.25, 0.24),
            (0.04, OptionType.CALL, 0.50, 0.18),
            (0.09, OptionType.CALL, 0.25, 0.20),
        )
        if all_positive_coordinates
        else (
            (-0.08, OptionType.PUT, -0.25, 0.24),
            (0.0, OptionType.CALL, 0.50, 0.18),
            (0.09, OptionType.CALL, 0.25, 0.20),
        ),
    )


def test_exact_signed_delta_matches_take_precedence() -> None:
    smile = _smile(
        (
            (0.01, OptionType.CALL, 0.20, 0.30),
            (0.02, OptionType.CALL, 0.40, 0.22),
            (0.03, OptionType.CALL, 0.25, 0.20),
            (0.04, OptionType.CALL, 0.10, 0.35),
        ),
    )

    result = interpolate_smile_at_delta(smile, OptionType.CALL, 0.25)

    assert result.is_success
    assert result.method is DeltaInterpolationMethod.OBSERVED
    assert result.implied_volatility == pytest.approx(0.20)
    assert result.left_point is result.right_point


@pytest.mark.parametrize(
    ("option_type", "target", "specs", "expected"),
    (
        (
            OptionType.CALL,
            0.25,
            (
                (0.02, OptionType.CALL, 0.40, 0.18),
                (0.09, OptionType.CALL, 0.10, 0.30),
            ),
            0.24,
        ),
        (
            OptionType.PUT,
            -0.25,
            (
                (-0.10, OptionType.PUT, -0.10, 0.20),
                (-0.03, OptionType.PUT, -0.40, 0.32),
            ),
            0.26,
        ),
    ),
)
def test_irregular_signed_delta_linear_interpolation(
    option_type,
    target,
    specs,
    expected,
) -> None:
    result = interpolate_smile_at_delta(_smile(specs), option_type, target)

    assert result.is_success
    assert result.method is DeltaInterpolationMethod.LINEAR_IV
    assert result.implied_volatility == pytest.approx(expected)
    assert result.interpolation_weight == pytest.approx(0.5)


@pytest.mark.parametrize("target", (0.05, 0.45))
def test_delta_interpolation_does_not_extrapolate(target) -> None:
    smile = _smile(
        (
            (0.02, OptionType.CALL, 0.10, 0.30),
            (0.09, OptionType.CALL, 0.40, 0.18),
        ),
    )

    result = interpolate_smile_at_delta(smile, OptionType.CALL, target)

    assert result.status is SmileMetricStatus.FAILED
    assert (
        result.failure_reason
        is DeltaMetricFailureReason.TARGET_DELTA_NOT_BRACKETED
    )


def test_unavailable_delta_and_missing_option_type_are_distinct() -> None:
    smile = _smile(((0.03, OptionType.CALL, None, 0.20),))

    call = interpolate_smile_at_delta(smile, OptionType.CALL, 0.25)
    put = interpolate_smile_at_delta(smile, OptionType.PUT, -0.25)

    assert call.failure_reason is DeltaMetricFailureReason.DELTA_UNAVAILABLE
    assert call.usable_point_count == 0
    assert call.excluded_point_count == 1
    assert (
        put.failure_reason
        is DeltaMetricFailureReason.NO_POINTS_FOR_OPTION_TYPE
    )


def test_repeated_delta_coordinates_are_machine_readable() -> None:
    smile = _smile(
        (
            (0.01, OptionType.CALL, 0.20, 0.25),
            (0.02, OptionType.CALL, 0.20, 0.24),
            (0.03, OptionType.CALL, 0.40, 0.20),
        ),
    )

    result = interpolate_smile_at_delta(smile, OptionType.CALL, 0.30)

    assert (
        result.failure_reason
        is DeltaMetricFailureReason.DEGENERATE_DELTA_COORDINATES
    )
    assert result.left_point is not None
    assert result.right_point is not None


def test_non_monotonic_multiple_brackets_are_ambiguous() -> None:
    smile = _smile(
        (
            (0.01, OptionType.CALL, 0.20, 0.25),
            (0.02, OptionType.CALL, 0.40, 0.20),
            (0.03, OptionType.CALL, 0.10, 0.30),
            (0.04, OptionType.CALL, 0.30, 0.23),
        ),
    )

    result = interpolate_smile_at_delta(smile, OptionType.CALL, 0.25)

    assert (
        result.failure_reason
        is DeltaMetricFailureReason.AMBIGUOUS_DELTA_BRACKET
    )


def test_multiple_exact_matches_are_reported_as_ambiguous() -> None:
    smile = _smile(
        (
            (0.01, OptionType.CALL, 0.25, 0.20),
            (0.02, OptionType.CALL, 0.25, 0.21),
        ),
    )

    result = interpolate_smile_at_delta(smile, OptionType.CALL, 0.25)

    assert (
        result.failure_reason
        is DeltaMetricFailureReason.AMBIGUOUS_DELTA_BRACKET
    )


def test_non_finite_interpolation_result_is_protected(monkeypatch) -> None:
    smile = _smile(
        (
            (0.02, OptionType.CALL, 0.10, 0.30),
            (0.09, OptionType.CALL, 0.40, 0.18),
        ),
    )
    monkeypatch.setattr(
        delta_metrics_module,
        "_linear_interpolate_iv",
        lambda *_args: float("inf"),
    )

    result = interpolate_smile_at_delta(smile, OptionType.CALL, 0.25)

    assert result.failure_reason is DeltaMetricFailureReason.NON_FINITE_RESULT


def test_risk_reversal_and_symmetric_butterfly_known_values() -> None:
    smile = _known_rr_bf_smile()
    local = calculate_smile_metrics(smile)

    reversal = calculate_risk_reversal(smile)
    butterfly = calculate_symmetric_delta_butterfly(
        smile,
        atm_result=local.atm,
    )

    assert reversal.is_success
    assert reversal.value == pytest.approx(-0.04)
    assert butterfly.is_success
    assert butterfly.value == pytest.approx(0.04)
    assert butterfly.atm_result is local.atm


def test_rr_can_succeed_when_butterfly_atm_fails() -> None:
    smile = _known_rr_bf_smile(all_positive_coordinates=True)
    local = calculate_smile_metrics(smile)

    reversal = calculate_risk_reversal(smile)
    butterfly = calculate_symmetric_delta_butterfly(
        smile,
        atm_result=local.atm,
    )

    assert reversal.is_success
    assert not local.atm.is_success
    assert not butterfly.is_success
    assert butterfly.failure_reason is DeltaStructureFailureReason.ATM_FAILED


def test_rr_leg_failures_are_independent_and_machine_readable() -> None:
    put_only = _smile(((-0.08, OptionType.PUT, -0.25, 0.24),))
    call_only = _smile(((0.08, OptionType.CALL, 0.25, 0.20),))

    call_failure = calculate_risk_reversal(put_only)
    put_failure = calculate_risk_reversal(call_only)

    assert (
        call_failure.failure_reason
        is DeltaStructureFailureReason.CALL_INTERPOLATION_FAILED
    )
    assert (
        put_failure.failure_reason
        is DeltaStructureFailureReason.PUT_INTERPOLATION_FAILED
    )


@pytest.mark.parametrize(
    ("smile", "reason"),
    (
        (
            lambda: _smile(((-0.08, OptionType.PUT, -0.25, 0.24),)),
            DeltaStructureFailureReason.CALL_INTERPOLATION_FAILED,
        ),
        (
            lambda: _smile(((0.08, OptionType.CALL, 0.25, 0.20),)),
            DeltaStructureFailureReason.PUT_INTERPOLATION_FAILED,
        ),
    ),
)
def test_butterfly_leg_failures_are_independent(smile, reason) -> None:
    result = calculate_symmetric_delta_butterfly(smile())

    assert not result.is_success
    assert result.failure_reason is reason


def test_rr_and_butterfly_non_finite_outputs_are_protected(monkeypatch) -> None:
    smile = _known_rr_bf_smile()
    monkeypatch.setattr(
        delta_metrics_module,
        "_risk_reversal_value",
        lambda *_args: float("inf"),
    )
    reversal = calculate_risk_reversal(smile)
    monkeypatch.setattr(
        delta_metrics_module,
        "_butterfly_value",
        lambda *_args: float("inf"),
    )
    butterfly = calculate_symmetric_delta_butterfly(smile)

    assert (
        reversal.failure_reason
        is DeltaStructureFailureReason.NON_FINITE_RESULT
    )
    assert (
        butterfly.failure_reason
        is DeltaStructureFailureReason.NON_FINITE_RESULT
    )


def test_invalid_delta_magnitude_is_structured() -> None:
    result = calculate_risk_reversal(_known_rr_bf_smile(), 0.0)

    assert not result.is_success
    assert (
        result.failure_reason
        is DeltaStructureFailureReason.INVALID_DELTA_MAGNITUDE
    )


def test_delta_aggregate_supports_multiple_magnitudes_and_partial_success() -> None:
    smile = _known_rr_bf_smile()
    config = DeltaMetricConfig(standard_delta_magnitudes=(0.35, 0.25, 0.25))

    result = calculate_smile_delta_metrics(smile, config)

    assert result.config.standard_delta_magnitudes == (0.25, 0.35)
    assert len(result.delta_results) == 4
    assert result.risk_reversal(0.25).is_success
    assert not result.risk_reversal(0.35).is_success


def test_delta_records_dataframes_and_input_order_are_deterministic() -> None:
    pytest.importorskip("pandas")
    smile = _smile(
        (
            (-0.08, OptionType.PUT, -0.25, 0.24),
            (0.0, OptionType.CALL, 0.50, 0.18),
            (0.03, OptionType.CALL, 0.40, 0.19),
            (0.09, OptionType.CALL, 0.10, 0.22),
        ),
    )
    shuffled = _replace_points(smile, reversed(smile.points))
    first = calculate_smile_delta_metrics(smile)
    second = calculate_smile_delta_metrics(shuffled)

    assert delta_volatility_results_to_records((first,)) == (
        delta_volatility_results_to_records((second,))
    )
    assert delta_structure_results_to_records((first,)) == (
        delta_structure_results_to_records((second,))
    )
    assert risk_reversal_results_to_records((first,)) == (
        risk_reversal_results_to_records((second,))
    )
    assert butterfly_results_to_records((first,)) == (
        butterfly_results_to_records((second,))
    )
    assert tuple(delta_volatility_results_to_dataframe((first,)).columns) == (
        DELTA_VOLATILITY_COLUMNS
    )
    assert tuple(delta_structure_results_to_dataframe((first,)).columns) == (
        DELTA_STRUCTURE_COLUMNS
    )
    assert tuple(risk_reversal_results_to_dataframe((first,)).columns) == (
        RISK_REVERSAL_COLUMNS
    )
    assert tuple(butterfly_results_to_dataframe((first,)).columns) == (
        BUTTERFLY_COLUMNS
    )


def test_term_structures_group_and_order_multiple_symbols_and_snapshots() -> None:
    second_asof = AS_OF + timedelta(minutes=1)
    smiles = (
        _known_rr_bf_smile(),
        _smile(
            (
                (-0.08, OptionType.PUT, -0.25, 0.25),
                (0.0, OptionType.CALL, 0.50, 0.19),
                (0.09, OptionType.CALL, 0.25, 0.21),
            ),
            expiration_days=90,
        ),
        _smile(
            (
                (-0.08, OptionType.PUT, -0.25, 0.24),
                (0.0, OptionType.CALL, 0.50, 0.18),
                (0.09, OptionType.CALL, 0.25, 0.20),
            ),
            symbol="MSFT",
        ),
        _smile(
            (
                (-0.08, OptionType.PUT, -0.25, 0.24),
                (0.0, OptionType.CALL, 0.50, 0.18),
                (0.09, OptionType.CALL, 0.25, 0.20),
            ),
            as_of=second_asof,
        ),
    )

    result = analyze_volatility_smiles(reversed(smiles))

    assert len(result.term_structures) == 3
    aapl = next(
        structure
        for structure in result.term_structures
        if structure.underlying_symbol == "AAPL"
        and structure.valuation_timestamp == AS_OF
    )
    assert [point.time_to_maturity for point in aapl.points] == sorted(
        point.time_to_maturity for point in aapl.points
    )
    assert len(aapl.points) == 2
    assert all(point.rr_25 is not None for point in aapl.points)


def test_term_structure_preserves_partial_failures_and_exports_none() -> None:
    smile = _known_rr_bf_smile(all_positive_coordinates=True)
    analysis = analyze_volatility_smiles((smile,))

    records = volatility_term_structures_to_records(analysis.term_structures)

    assert records[0]["rr_25"] == pytest.approx(-0.04)
    assert records[0]["bf_25"] is None
    assert records[0]["bf_25_status"] == "FAILED"
    assert tuple(
        volatility_term_structures_to_dataframe(
            analysis.term_structures,
        ).columns
    ) == TERM_STRUCTURE_COLUMNS


def test_duplicate_expiry_policy_and_cross_group_invariants_are_explicit() -> None:
    first = analyze_volatility_smiles((_known_rr_bf_smile(),))
    point = first.term_structures[0].points[0]

    with pytest.raises(ValueError, match="duplicate expiration"):
        VolatilityTermStructure(
            underlying_symbol="AAPL",
            valuation_timestamp=AS_OF,
            points=(point, point),
            duplicate_policy=DuplicateTermStructurePolicy.ERROR,
        )

    other = analyze_volatility_smiles(
        (
            _smile(
                (
                    (-0.08, OptionType.PUT, -0.25, 0.24),
                    (0.0, OptionType.CALL, 0.50, 0.18),
                    (0.09, OptionType.CALL, 0.25, 0.20),
                ),
                symbol="MSFT",
            ),
        ),
    )
    with pytest.raises(ValueError, match="underlying_symbol"):
        VolatilityTermStructure(
            underlying_symbol="AAPL",
            valuation_timestamp=AS_OF,
            points=(point, other.term_structures[0].points[0]),
        )


def test_analysis_empty_input_and_independent_summary_counts() -> None:
    empty = analyze_volatility_smiles(())
    assert empty.summary.input_smile_count == 0
    assert empty.term_structures == ()

    config = SmileAnalysisConfig(
        delta_metric_config=DeltaMetricConfig(
            standard_delta_magnitudes=(0.25, 0.35),
        ),
    )
    result = analyze_volatility_smiles((_known_rr_bf_smile(),), config)
    summary = result.summary

    assert summary.local_metric_result_count == 1
    assert summary.delta_metric_result_count == 1
    assert summary.term_structure_point_count == 1
    assert summary.outcome(SmileAnalysisMetric.RISK_REVERSAL, 0.25).success_count == 1
    assert summary.outcome(SmileAnalysisMetric.RISK_REVERSAL, 0.35).failure_count == 1
    assert len(smile_analysis_summary_to_records(result)) == 11


def test_analysis_csv_bytes_and_hashes_are_deterministic(tmp_path) -> None:
    smiles = (
        _known_rr_bf_smile(),
        _smile(
            (
                (-0.08, OptionType.PUT, -0.25, 0.25),
                (0.0, OptionType.CALL, 0.50, 0.19),
                (0.09, OptionType.CALL, 0.25, 0.21),
            ),
            expiration_days=90,
        ),
    )
    first = analyze_volatility_smiles(smiles)
    second = analyze_volatility_smiles(reversed(smiles))

    first_exports = write_smile_analysis_csv(tmp_path / "first", first)
    second_exports = write_smile_analysis_csv(tmp_path / "second", second)

    assert [export.name for export in first_exports.exports] == [
        export.name for export in second_exports.exports
    ]
    for first_export, second_export in zip(
        first_exports.exports,
        second_exports.exports,
    ):
        assert first_export.sha256 == second_export.sha256
        assert first_export.path.read_bytes() == second_export.path.read_bytes()


def test_medium_pipeline_preserves_every_smile_through_full_analysis(
    tmp_path,
) -> None:
    config = SyntheticOptionDatasetConfig(row_count=2_000)
    source = write_synthetic_option_quote_csv(tmp_path / "quotes.csv", config)
    pipeline = run_csv_volatility_pipeline(
        source.path,
        ingestion_config=synthetic_option_quote_csv_config(),
        carry=synthetic_volatility_pipeline_carry(config),
        valuation_date=config.valuation_date,
        cleaning_config=synthetic_volatility_pipeline_cleaning_config(),
    )
    selection = build_volatility_smiles(pipeline.implied_volatility_chain)

    analysis = analyze_volatility_smiles(selection.smiles)
    exports = write_smile_analysis_csv(tmp_path / "analysis", analysis)

    summary = analysis.summary
    assert summary.input_smile_count == selection.summary.smile_count == 20
    assert summary.local_metric_result_count == 20
    assert summary.delta_metric_result_count == 20
    assert summary.term_structure_point_count == 20
    assert summary.term_structure_count == 4
    assert all(
        count.success_count + count.failure_count == 20
        for count in summary.outcome_counts
    )
    assert all(export.byte_count > 0 for export in exports.exports)
