from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from math import exp, sqrt

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
    SMILE_METRIC_COLUMNS,
    AtmInterpolationMethod,
    LocalCurvatureMethod,
    LocalSkewMethod,
    LocalSmileFitConfig,
    SmileMetricConfig,
    SmileMetricFailureReason,
    SmileMetricStatus,
    SurfaceFailureReason,
    SurfaceViolationKind,
    SyntheticOptionDatasetConfig,
    VolatilitySurface,
    build_implied_volatility_chain,
    build_volatility_smiles,
    calculate_smile_metrics,
    calculate_smile_metrics_for_smiles,
    diagnose_surface_arbitrage,
    interpolate_total_variance,
    run_csv_volatility_pipeline,
    smile_metrics_to_dataframe,
    smile_metrics_to_records,
    synthetic_option_quote_csv_config,
    synthetic_volatility_pipeline_carry,
    synthetic_volatility_pipeline_cleaning_config,
    write_synthetic_option_quote_csv,
)


UTC = timezone.utc
AS_OF = datetime(2026, 7, 30, 14, 30, tzinfo=UTC)
VALUATION_DATE = date(2026, 7, 30)


def _smile_from_total_variance(
    coordinates,
    total_variance_function,
    *,
    expiration_days: int = 365,
):
    expiration = VALUATION_DATE + timedelta(days=expiration_days)
    maturity = expiration_days / 365.0
    quotes = []
    for index, coordinate in enumerate(coordinates):
        strike = 100.0 * exp(coordinate)
        option_type = OptionType.PUT if coordinate < 0.0 else OptionType.CALL
        total_variance = total_variance_function(coordinate)
        volatility = sqrt(total_variance / maturity)
        price_function = (
            put_price if option_type is OptionType.PUT else call_price
        )
        price = price_function(
            100.0,
            strike,
            maturity,
            0.0,
            volatility,
        )
        quotes.append(
            OptionQuote(
                contract=OptionContract(
                    underlying_symbol="AAPL",
                    expiration=expiration,
                    strike=strike,
                    option_type=option_type,
                    exercise_style=ExerciseStyle.EUROPEAN,
                    source_contract_id=f"metric-{index}",
                ),
                quote_timestamp=AS_OF,
                bid=price,
                ask=price,
                bid_size=100,
                ask_size=100,
            ),
        )
    snapshot = OptionChainSnapshot(
        underlying_symbol="AAPL",
        as_of=AS_OF,
        quotes=tuple(quotes),
        underlying_quote=UnderlyingQuote(
            symbol="AAPL",
            quote_timestamp=AS_OF,
            price=100.0,
        ),
    )
    enriched = enrich_option_chain_snapshot(
        snapshot,
        carry=CarryAssumptions(
            risk_free_curve=FlatZeroRateCurve(0.0),
            dividend_curve=FlatDividendYieldCurve(0.0),
        ),
        valuation_date=VALUATION_DATE,
    )
    selection = build_volatility_smiles(
        build_implied_volatility_chain(enriched),
    )
    assert selection.summary.selected_point_count == len(coordinates)
    return selection.smiles[0]


def test_maturity_interpolation_retains_exact_nodes_and_rejects_extrapolation():
    nodes = ((0.5, 0.03), (0.1, 0.01))
    exact = interpolate_total_variance(nodes, 0.1)
    assert exact.total_variance == 0.01
    assert exact.lower_maturity == exact.upper_maturity == 0.1
    middle = interpolate_total_variance(nodes, 0.2)
    assert middle.total_variance == pytest.approx(0.015)
    assert middle.upper_weight == pytest.approx(0.25)
    assert interpolate_total_variance(nodes, 0.05).failure_reason is SurfaceFailureReason.NO_LOWER_MATURITY
    assert interpolate_total_variance(nodes, 1.0).failure_reason is SurfaceFailureReason.NO_UPPER_MATURITY
    assert interpolate_total_variance((), 0.2).failure_reason is SurfaceFailureReason.NO_MATURITY_NODES
    with pytest.raises(ValueError, match="duplicate"):
        interpolate_total_variance(((0.1, 0.01), (0.1, 0.02)), 0.1)


def test_surface_queries_a_ragged_grid_and_reports_holdout_residuals():
    front = _smile_from_total_variance((-0.10, 0.0, 0.10), lambda k: 0.04 + 0.04 * k * k)
    back = _smile_from_total_variance((0.0, 0.15), lambda k: 0.09 + 0.03 * k, expiration_days=730)
    surface = VolatilitySurface((back, front))

    query = surface.query(0.05, 1.5)
    assert query.total_variance == pytest.approx((0.0402 + 0.0915) / 2, abs=1e-8)
    assert query.implied_volatility == pytest.approx(sqrt(query.total_variance / 1.5))
    assert (query.lower_maturity, query.upper_maturity, query.upper_weight) == (1.0, 2.0, 0.5)
    assert surface.query(-0.05, 1.0).implied_volatility is not None
    assert surface.query(-0.05, 1.5).failure_reason is SurfaceFailureReason.OUTSIDE_STRIKE_SUPPORT
    assert all(row.total_variance_residual == 0 for row in surface.residuals())
    held_out = surface.residuals(leave_one_out=True)
    assert [row.total_variance_residual for row in held_out if row.fitted_total_variance is not None] == pytest.approx([0.0004], abs=1e-8)
    with pytest.raises(ValueError, match="duplicate"):
        VolatilitySurface((front, front))


def test_surface_does_not_skip_a_sparse_intermediate_expiry():
    front = _smile_from_total_variance((-0.1, 0.1), lambda _: 0.04)
    middle = _smile_from_total_variance((0.05,), lambda _: 0.06, expiration_days=730)
    back = _smile_from_total_variance((-0.1, 0.1), lambda _: 0.08, expiration_days=1095)
    surface = VolatilitySurface((front, middle, back))

    assert surface.query(0.0, 1.5).failure_reason is SurfaceFailureReason.OUTSIDE_STRIKE_SUPPORT
    assert surface.query(middle.points[0].log_forward_moneyness, 2.0).total_variance == pytest.approx(0.06)


def test_surface_diagnostics_cover_calendar_and_interpolation_induced_butterflies():
    front = _smile_from_total_variance((-0.25, 0.0, 0.25), lambda _: 0.04)
    later = _smile_from_total_variance((-0.25, 0.0, 0.25), lambda _: 0.06, expiration_days=730)
    clean = diagnose_surface_arbitrage(VolatilitySurface((front, later)))
    assert not clean.violations
    assert clean.calendar_comparison_count == 3
    assert [item.maturity for item in clean.strike_coverage] == [1.0, 1.5, 2.0]
    assert all(item.minimum_normalized_density_estimate > 0 for item in clean.strike_coverage)

    convexity_input = _smile_from_total_variance((-0.25, 0.0, 0.25), lambda k: 0.008 if k <= 0 else 0.08)
    report = diagnose_surface_arbitrage(VolatilitySurface((convexity_input, later)))
    butterfly = next(item for item in report.violations if item.kind is SurfaceViolationKind.BUTTERFLY)
    assert butterfly.uses_interpolation
    assert butterfly.normalized_density_estimate < 0
    calendar = next(item for item in report.violations if item.kind is SurfaceViolationKind.CALENDAR_TOTAL_VARIANCE)
    assert calendar.magnitude == pytest.approx(0.02, abs=1e-8)
    assert calendar.later_maturity == 2.0
    assert not calendar.uses_interpolation


def test_surface_diagnostics_report_disjoint_strike_domains():
    front = _smile_from_total_variance((-0.25, -0.1), lambda _: 0.04)
    back = _smile_from_total_variance((0.1, 0.25), lambda _: 0.08, expiration_days=730)
    diagnostics = diagnose_surface_arbitrage(VolatilitySurface((front, back)))
    assert diagnostics.calendar_comparison_count == 0
    assert diagnostics.calendar_pairs_without_overlap == ((1.0, 2.0),)
    assert diagnostics.strike_coverage[1].sample_count == 0


def test_surface_diagnostics_compare_nonadjacent_expiries_across_a_strike_gap():
    front = _smile_from_total_variance((-0.25, -0.1), lambda _: 0.08)
    middle = _smile_from_total_variance((0.1, 0.25), lambda _: 0.06, expiration_days=730)
    back = _smile_from_total_variance((-0.25, -0.1), lambda _: 0.04, expiration_days=1095)
    diagnostics = diagnose_surface_arbitrage(VolatilitySurface((front, middle, back)))

    calendar = [item for item in diagnostics.violations if item.kind is SurfaceViolationKind.CALENDAR_TOTAL_VARIANCE]
    assert diagnostics.calendar_pairs_without_overlap == ((1.0, 2.0), (2.0, 3.0))
    assert len(calendar) == 2
    assert all(item.maturity == 1.0 and item.later_maturity == 3.0 for item in calendar)


def test_observed_atm_is_used_directly() -> None:
    a, b, c = 0.04, -0.03, 0.20
    smile = _smile_from_total_variance(
        (-0.10, 0.0, 0.08),
        lambda k: a + b * k + 0.5 * c * k * k,
    )

    result = calculate_smile_metrics(smile)

    assert result.atm.status is SmileMetricStatus.SUCCESS
    assert result.atm.method is AtmInterpolationMethod.OBSERVED
    assert result.atm.atm_total_variance == pytest.approx(a, abs=1e-8)
    assert result.atm.atm_volatility == pytest.approx(sqrt(a), abs=1e-8)
    assert result.atm.left_point is smile.observed_atm_point
    assert result.atm.center_point is smile.observed_atm_point
    assert result.atm.right_point is smile.observed_atm_point
    assert result.skew.total_variance_skew_slope == pytest.approx(b, abs=1e-7)
    assert result.curvature.total_variance_curvature == pytest.approx(
        c,
        abs=1e-6,
    )


def test_linear_total_variance_atm_interpolation_recovers_level() -> None:
    a, b = 0.04, -0.03
    smile = _smile_from_total_variance(
        (-0.12, -0.04, 0.03, 0.11),
        lambda k: a + b * k,
    )

    result = calculate_smile_metrics(smile)

    assert result.atm.status is SmileMetricStatus.SUCCESS
    assert result.atm.method is AtmInterpolationMethod.LINEAR_TOTAL_VARIANCE
    assert result.atm.atm_total_variance == pytest.approx(a, abs=1e-8)
    assert result.atm.atm_volatility == pytest.approx(sqrt(a), abs=1e-8)
    assert result.atm.left_point is not None
    assert result.atm.right_point is not None
    assert result.atm.left_point.log_forward_moneyness == pytest.approx(-0.04)
    assert result.atm.right_point.log_forward_moneyness == pytest.approx(0.03)


def test_quadratic_local_fit_recovers_slope_and_curvature_on_irregular_grid() -> None:
    a, b, c = 0.04, -0.03, 0.20
    smile = _smile_from_total_variance(
        (-0.13, -0.05, 0.02, 0.09),
        lambda k: a + b * k + 0.5 * c * k * k,
    )

    result = calculate_smile_metrics(smile)

    assert result.skew.status is SmileMetricStatus.SUCCESS
    assert result.skew.method is LocalSkewMethod.QUADRATIC_LOCAL_FIT
    assert result.skew.total_variance_skew_slope == pytest.approx(b, abs=1e-7)
    assert result.skew.fit_intercept == pytest.approx(a, abs=1e-7)
    assert result.curvature.status is SmileMetricStatus.SUCCESS
    assert result.curvature.method is LocalCurvatureMethod.QUADRATIC_LOCAL_FIT
    assert result.curvature.total_variance_curvature == pytest.approx(
        c,
        abs=1e-6,
    )
    assert len(result.curvature.source_points) == 4


def test_two_bracketing_points_support_atm_and_skew_but_not_curvature() -> None:
    a, b = 0.04, -0.03
    smile = _smile_from_total_variance(
        (-0.05, 0.03),
        lambda k: a + b * k,
    )

    result = calculate_smile_metrics(smile)

    assert result.atm.is_success
    assert result.skew.is_success
    assert result.skew.method is LocalSkewMethod.BRACKET_SECANT
    assert result.skew.total_variance_skew_slope == pytest.approx(b, abs=1e-8)
    assert result.curvature.status is SmileMetricStatus.FAILED
    assert (
        result.curvature.failure_reason
        is SmileMetricFailureReason.INSUFFICIENT_POINTS
    )


def test_single_sided_smile_does_not_extrapolate() -> None:
    smile = _smile_from_total_variance(
        (-0.13, -0.07, -0.02),
        lambda k: 0.04 - 0.03 * k,
    )

    result = calculate_smile_metrics(smile)

    assert result.atm.status is SmileMetricStatus.FAILED
    assert result.atm.failure_reason is SmileMetricFailureReason.ATM_NOT_BRACKETED
    assert result.atm.left_point is not None
    assert result.atm.right_point is None
    assert result.skew.failure_reason is SmileMetricFailureReason.ATM_NOT_BRACKETED
    assert (
        result.curvature.failure_reason
        is SmileMetricFailureReason.ATM_NOT_BRACKETED
    )


def test_observed_atm_can_be_disabled_in_favour_of_bracketed_interpolation() -> None:
    a, b = 0.04, -0.03
    smile = _smile_from_total_variance(
        (-0.08, 0.0, 0.06),
        lambda k: a + b * k,
    )

    result = calculate_smile_metrics(
        smile,
        SmileMetricConfig(allow_observed_atm=False),
    )

    assert result.atm.method is AtmInterpolationMethod.LINEAR_TOTAL_VARIANCE
    assert result.atm.center_point is None
    assert result.atm.atm_total_variance == pytest.approx(a, abs=1e-8)


def test_duplicate_local_coordinates_are_machine_readable() -> None:
    smile = _smile_from_total_variance(
        (-0.05, 0.03, 0.08),
        lambda k: 0.04 - 0.03 * k + 0.10 * k * k,
    )
    points = list(smile.points)
    duplicate_enriched = replace(
        points[2].iv_quote.enriched_quote,
        log_moneyness=points[1].log_forward_moneyness,
    )
    duplicate_quote = replace(
        points[2].iv_quote,
        enriched_quote=duplicate_enriched,
    )
    points[2] = replace(points[2], iv_quote=duplicate_quote)
    duplicate_smile = replace(smile, points=tuple(points))

    result = calculate_smile_metrics(duplicate_smile)

    assert result.atm.is_success
    assert (
        result.skew.failure_reason
        is SmileMetricFailureReason.DEGENERATE_COORDINATES
    )
    assert (
        result.curvature.failure_reason
        is SmileMetricFailureReason.DEGENERATE_COORDINATES
    )


def test_total_variance_overflow_is_machine_readable() -> None:
    smile = _smile_from_total_variance(
        (-0.05, 0.03),
        lambda k: 0.04 - 0.03 * k,
    )
    points = list(smile.points)
    points[0] = replace(points[0], implied_volatility=1e308)
    overflow_smile = replace(smile, points=tuple(points))

    result = calculate_smile_metrics(overflow_smile)

    assert result.atm.failure_reason is SmileMetricFailureReason.NON_FINITE_RESULT
    assert result.skew.failure_reason is SmileMetricFailureReason.NON_FINITE_RESULT
    assert (
        result.curvature.failure_reason
        is SmileMetricFailureReason.INSUFFICIENT_POINTS
    )

def test_empty_and_non_positive_maturity_fail_independently() -> None:
    base = _smile_from_total_variance(
        (-0.05, 0.03),
        lambda k: 0.04 - 0.03 * k,
    )
    empty = replace(base, points=(), nearest_atm_index=None)
    expired = replace(empty, time_to_maturity=0.0)

    empty_result = calculate_smile_metrics(empty)
    expired_result = calculate_smile_metrics(expired)

    assert empty_result.atm.failure_reason is SmileMetricFailureReason.EMPTY_SMILE
    assert empty_result.skew.failure_reason is SmileMetricFailureReason.EMPTY_SMILE
    assert (
        empty_result.curvature.failure_reason
        is SmileMetricFailureReason.EMPTY_SMILE
    )
    assert (
        expired_result.atm.failure_reason
        is SmileMetricFailureReason.NON_POSITIVE_MATURITY
    )
    assert (
        expired_result.skew.failure_reason
        is SmileMetricFailureReason.NON_POSITIVE_MATURITY
    )


def test_local_fit_window_can_fail_without_changing_atm_interpolation() -> None:
    smile = _smile_from_total_variance(
        (-0.12, -0.04, 0.03, 0.11),
        lambda k: 0.04 - 0.03 * k,
    )
    config = SmileMetricConfig(
        local_fit=LocalSmileFitConfig(maximum_abs_log_moneyness=0.02),
    )

    result = calculate_smile_metrics(smile, config)

    assert result.atm.is_success
    assert result.skew.failure_reason is SmileMetricFailureReason.INSUFFICIENT_POINTS
    assert (
        result.curvature.failure_reason
        is SmileMetricFailureReason.INSUFFICIENT_POINTS
    )


def test_metric_records_dataframe_and_batch_ordering_are_deterministic() -> None:
    pd = pytest.importorskip("pandas")
    long_smile = _smile_from_total_variance(
        (-0.05, 0.03, 0.08),
        lambda k: 0.04 - 0.03 * k + 0.10 * k * k,
        expiration_days=365,
    )
    short_smile = _smile_from_total_variance(
        (-0.05, 0.03, 0.08),
        lambda k: 0.02 - 0.01 * k + 0.05 * k * k,
        expiration_days=30,
    )

    first = calculate_smile_metrics_for_smiles((long_smile, short_smile))
    second = calculate_smile_metrics_for_smiles((short_smile, long_smile))
    records = smile_metrics_to_records(first)
    frame = smile_metrics_to_dataframe(first)

    assert smile_metrics_to_records(first) == smile_metrics_to_records(second)
    assert [record["expiration"] for record in records] == [
        short_smile.expiration,
        long_smile.expiration,
    ]
    assert tuple(frame.columns) == SMILE_METRIC_COLUMNS
    assert frame.to_dict(orient="records") == list(records)
    assert records[0]["atm_status"] == "SUCCESS"
    assert records[0]["curvature_status"] == "SUCCESS"
    assert records[0]["skew_source_strikes"]
    assert records[0]["curvature_source_log_forward_moneyness"]


def test_empty_metric_dataframe_has_stable_columns() -> None:
    pd = pytest.importorskip("pandas")

    frame = smile_metrics_to_dataframe(())

    assert isinstance(frame, pd.DataFrame)
    assert tuple(frame.columns) == SMILE_METRIC_COLUMNS


def test_metric_config_rejects_ambiguous_or_invalid_policies() -> None:
    with pytest.raises(ValueError, match="LINEAR_TOTAL_VARIANCE"):
        SmileMetricConfig(interpolation_method=AtmInterpolationMethod.OBSERVED)
    with pytest.raises(ValueError, match="at least 3"):
        LocalSmileFitConfig(minimum_point_count=2)
    with pytest.raises(ValueError, match="positive integer"):
        LocalSmileFitConfig(max_points_each_side=0)


def test_medium_pipeline_produces_one_metric_result_per_smile(tmp_path) -> None:
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

    metrics = calculate_smile_metrics_for_smiles(selection.smiles)

    assert len(metrics) == selection.summary.smile_count == 20
    assert all(result.atm.is_success for result in metrics)
    assert all(result.skew.is_success for result in metrics)
    assert all(result.curvature.is_success for result in metrics)
    assert len(smile_metrics_to_records(metrics)) == 20
