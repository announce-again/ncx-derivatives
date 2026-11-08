from datetime import date, datetime, timedelta, timezone
from math import inf, nan

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
    SMILE_GROUP_DIAGNOSTIC_COLUMNS,
    SMILE_POINT_COLUMNS,
    SMILE_SELECTION_DIAGNOSTIC_COLUMNS,
    DuplicateStrikePolicy,
    ImpliedVolatilityChain,
    ImpliedVolatilityFailureReason,
    SmileGroupDiagnostic,
    SmileGroupDiagnosticReason,
    SmileIvSource,
    SmilePointDiagnosticFlag,
    SmileSelectionConfig,
    SmileSelectionReason,
    SyntheticOptionDatasetConfig,
    VolatilitySmile,
    VolatilitySmilePoint,
    build_implied_volatility_chain,
    build_volatility_smiles,
    run_csv_volatility_pipeline,
    smile_group_diagnostics_to_dataframe,
    smile_group_diagnostics_to_records,
    smile_selection_diagnostics_to_dataframe,
    smile_selection_diagnostics_to_records,
    synthetic_option_quote_csv_config,
    synthetic_volatility_pipeline_carry,
    synthetic_volatility_pipeline_cleaning_config,
    volatility_smiles_to_dataframe,
    volatility_smiles_to_records,
    write_synthetic_option_quote_csv,
)

UTC = timezone.utc
AS_OF = datetime(2026, 7, 30, 14, 30, tzinfo=UTC)
VALUATION_DATE = date(2026, 7, 30)
EXPIRATION = date(2027, 7, 30)


def _quote(
    option_type: OptionType,
    strike: float,
    bid_volatility: float,
    ask_volatility: float,
    *,
    timestamp: datetime = AS_OF,
    source_id: str | None = None,
    bid_size: int = 10,
    ask_size: int = 10,
    session_volume: int = 100,
    open_interest: int = 1_000,
) -> OptionQuote:
    price_function = call_price if option_type is OptionType.CALL else put_price
    return OptionQuote(
        contract=OptionContract(
            underlying_symbol="AAPL",
            expiration=EXPIRATION,
            strike=strike,
            option_type=option_type,
            exercise_style=ExerciseStyle.EUROPEAN,
            source_contract_id=source_id or f"{option_type.value}-{strike}",
        ),
        quote_timestamp=timestamp,
        bid=price_function(100.0, strike, 1.0, 0.0, bid_volatility),
        ask=price_function(100.0, strike, 1.0, 0.0, ask_volatility),
        bid_size=bid_size,
        ask_size=ask_size,
        session_volume=session_volume,
        open_interest=open_interest,
    )


def _chain(*quotes: OptionQuote, spot: float = 100.0):
    snapshot = OptionChainSnapshot(
        underlying_symbol="AAPL",
        as_of=AS_OF,
        quotes=quotes,
        underlying_quote=UnderlyingQuote(
            symbol="AAPL",
            quote_timestamp=AS_OF,
            price=spot,
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
    return build_implied_volatility_chain(enriched)


def test_default_selection_builds_one_otm_point_per_strike_and_marks_atm() -> None:
    chain = _chain(
        *(
            _quote(option_type, strike, 0.19, 0.21)
            for strike in (80.0, 90.0, 100.0, 110.0, 120.0)
            for option_type in (OptionType.CALL, OptionType.PUT)
        ),
    )

    result = build_volatility_smiles(chain)

    assert result.summary.input_quote_count == 10
    assert result.config == SmileSelectionConfig()
    assert result.summary.smile_count == 1
    assert result.summary.selected_point_count == 5
    assert result.summary.excluded_quote_count == 5
    smile = result.smiles[0]
    assert [point.strike for point in smile.points] == [80, 90, 100, 110, 120]
    assert [point.option_type for point in smile.points] == [
        OptionType.PUT,
        OptionType.PUT,
        OptionType.CALL,
        OptionType.CALL,
        OptionType.CALL,
    ]
    assert smile.nearest_atm_point is not None
    assert smile.nearest_atm_point.strike == 100.0
    assert smile.observed_atm_point is smile.nearest_atm_point
    assert smile.has_observed_atm_point
    assert all(point.delta is not None for point in smile.points)
    assert all(point.iv_bid_ask_spread is not None for point in smile.points)
    assert [diagnostic.reasons for diagnostic in result.diagnostics].count(
        (SmileSelectionReason.NOT_OTM,),
    ) == 4
    assert [diagnostic.reasons for diagnostic in result.diagnostics].count(
        (SmileSelectionReason.DUPLICATE_STRIKE,),
    ) == 1


def test_selection_source_is_explicit_and_export_preserves_iv_observations() -> None:
    result = build_volatility_smiles(
        _chain(_quote(OptionType.CALL, 110.0, 0.18, 0.24)),
        SmileSelectionConfig(iv_source=SmileIvSource.BID),
    )

    point = result.smiles[0].points[0]
    record = volatility_smiles_to_records(result.smiles)[0]

    assert point.implied_volatility == pytest.approx(0.18)
    assert record["iv_source"] == "bid"
    assert record["implied_volatility"] == pytest.approx(0.18)
    assert record["bid_iv"] == pytest.approx(0.18)
    assert record["ask_iv"] == pytest.approx(0.24)
    assert record["iv_bid_ask_spread"] == pytest.approx(0.06)
    assert record["source_status"] == "SUCCESS"
    assert record["is_nearest_atm"] is True
    assert record["is_observed_atm"] is False


def test_missing_failed_and_flagged_results_have_machine_readable_reasons(
    monkeypatch,
) -> None:
    import ncx_derivatives.volatility.chains as chains

    missing = _quote(OptionType.CALL, 110.0, 0.18, 0.22)
    missing = OptionQuote(
        contract=missing.contract,
        quote_timestamp=missing.quote_timestamp,
        bid=None,
        ask=missing.ask,
    )
    missing_result = build_volatility_smiles(
        _chain(missing),
        SmileSelectionConfig(otm_only=False),
    )
    assert missing_result.diagnostics[0].reasons == (
        SmileSelectionReason.MISSING_BID,
        SmileSelectionReason.FAILED_IV,
    )
    assert (
        missing_result.diagnostics[0].source_result.failure_reason
        is ImpliedVolatilityFailureReason.MISSING_PRICE
    )

    low_vega = OptionQuote(
        contract=_quote(OptionType.CALL, 100.0, 0.2, 0.2).contract,
        quote_timestamp=AS_OF,
        bid=0.0,
        ask=0.0,
    )
    low_vega_result = build_volatility_smiles(
        _chain(low_vega),
        SmileSelectionConfig(otm_only=False),
    )
    assert low_vega_result.diagnostics[0].reasons == (
        SmileSelectionReason.LOW_VEGA,
    )

    upper_bound = OptionQuote(
        contract=_quote(OptionType.CALL, 100.0, 0.2, 0.2).contract,
        quote_timestamp=AS_OF,
        bid=100.0,
        ask=100.0,
    )
    upper_bound_result = build_volatility_smiles(
        _chain(upper_bound),
        SmileSelectionConfig(otm_only=False),
    )
    assert upper_bound_result.diagnostics[0].reasons == (
        SmileSelectionReason.UPPER_BOUND_IV,
    )
    non_finite_result = build_volatility_smiles(
        _chain(upper_bound),
        SmileSelectionConfig(
            otm_only=False,
            excluded_diagnostic_flags=(),
        ),
    )
    assert non_finite_result.smiles[0].points == ()
    assert non_finite_result.diagnostics[0].reasons == (
        SmileSelectionReason.NON_FINITE_IV,
    )

    monkeypatch.setattr(chains, "black_scholes_vega", lambda *args: nan)
    unavailable_result = build_volatility_smiles(
        _chain(_quote(OptionType.CALL, 110.0, 0.18, 0.22)),
        SmileSelectionConfig(otm_only=False),
    )
    assert unavailable_result.diagnostics[0].reasons == (
        SmileSelectionReason.VEGA_UNAVAILABLE,
    )


def test_duplicate_resolution_prefers_latest_then_best_liquidity() -> None:
    older = _quote(
        OptionType.CALL,
        110.0,
        0.199,
        0.201,
        timestamp=AS_OF - timedelta(seconds=1),
        source_id="older",
        bid_size=100,
        ask_size=100,
    )
    latest = _quote(
        OptionType.CALL,
        110.0,
        0.18,
        0.24,
        source_id="latest",
        bid_size=1,
        ask_size=1,
    )
    exact_atm_call = _quote(
        OptionType.CALL,
        100.0,
        0.18,
        0.24,
        timestamp=AS_OF + timedelta(seconds=1),
        source_id="atm-call-newer-but-wider",
        bid_size=1,
        ask_size=1,
    )
    exact_atm_put = _quote(
        OptionType.PUT,
        100.0,
        0.199,
        0.201,
        source_id="atm-put",
        bid_size=100,
        ask_size=100,
    )

    result = build_volatility_smiles(
        _chain(older, latest, exact_atm_call, exact_atm_put),
    )

    selected = {
        (point.strike, point.option_type): point
        for point in result.smiles[0].points
    }
    assert selected[(110.0, OptionType.CALL)].quote_timestamp == AS_OF
    assert (100.0, OptionType.PUT) in selected
    assert {diagnostic.reasons for diagnostic in result.diagnostics} == {
        (SmileSelectionReason.STALE_QUOTE,),
        (SmileSelectionReason.DUPLICATE_STRIKE,),
    }


def test_liquidity_filters_exclude_observation_without_deleting_diagnostic() -> None:
    result = build_volatility_smiles(
        _chain(
            _quote(
                OptionType.CALL,
                110.0,
                0.15,
                0.30,
                bid_size=2,
                ask_size=2,
                session_volume=5,
                open_interest=10,
            ),
        ),
        SmileSelectionConfig(
            max_relative_spread=0.01,
            min_bid_size=10,
            min_ask_size=10,
            min_session_volume=100,
            min_open_interest=100,
        ),
    )

    assert result.smiles[0].points == ()
    assert result.smiles[0].nearest_atm_point is None
    assert result.smiles[0].observed_atm_point is None
    assert result.diagnostics[0].reasons == (
        SmileSelectionReason.LIQUIDITY_FILTER,
    )


def test_records_and_dataframes_have_stable_semantics() -> None:
    pd = pytest.importorskip("pandas")
    result = build_volatility_smiles(
        _chain(
            _quote(OptionType.PUT, 90.0, 0.18, 0.22),
            _quote(OptionType.CALL, 90.0, 0.18, 0.22),
        ),
    )

    point_frame = volatility_smiles_to_dataframe(result.smiles)
    diagnostic_frame = smile_selection_diagnostics_to_dataframe(
        result.diagnostics,
    )
    diagnostic_records = smile_selection_diagnostics_to_records(
        result.diagnostics,
    )

    assert isinstance(point_frame, pd.DataFrame)
    assert isinstance(diagnostic_frame, pd.DataFrame)
    assert tuple(point_frame.columns) == SMILE_POINT_COLUMNS
    assert tuple(diagnostic_frame.columns) == SMILE_SELECTION_DIAGNOSTIC_COLUMNS
    assert diagnostic_records[0]["reasons"] == "NOT_OTM"
    assert diagnostic_frame.to_dict(orient="records") == list(diagnostic_records)


def test_empty_chain_has_stable_result_and_dataframe() -> None:
    pd = pytest.importorskip("pandas")

    result = build_volatility_smiles(build_implied_volatility_chain(()))
    frame = volatility_smiles_to_dataframe(result.smiles)

    assert result.smiles == ()
    assert result.diagnostics == ()
    assert result.group_diagnostics == ()
    assert result.summary.input_quote_count == 0
    assert result.summary.smile_count == 0
    assert isinstance(frame, pd.DataFrame)
    assert tuple(frame.columns) == SMILE_POINT_COLUMNS


def test_medium_pipeline_to_smiles_is_deterministic_and_isolates_bad_rows(
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

    first = build_volatility_smiles(pipeline.implied_volatility_chain)
    second = build_volatility_smiles(pipeline.implied_volatility_chain)

    assert first.summary.smile_count == 20
    assert first.summary.empty_smile_count == 0
    assert first.summary.selected_point_count > 900
    assert (
        first.summary.selected_point_count + first.summary.excluded_quote_count
        == pipeline.counts.iv_quote_count
    )
    assert volatility_smiles_to_records(first.smiles) == volatility_smiles_to_records(
        second.smiles,
    )
    assert smile_selection_diagnostics_to_records(
        first.diagnostics,
    ) == smile_selection_diagnostics_to_records(second.diagnostics)
    assert any(
        SmileSelectionReason.FAILED_IV in diagnostic.reasons
        for diagnostic in first.diagnostics
    )
    for smile in first.smiles:
        assert smile.nearest_atm_point is not None
        assert sum(point is smile.nearest_atm_point for point in smile.points) == 1
        for point in smile.points:
            if point.log_forward_moneyness < 0.0:
                assert point.option_type is OptionType.PUT
            elif point.log_forward_moneyness > 0.0:
                assert point.option_type is OptionType.CALL


def test_nearest_forward_point_is_not_mislabelled_as_observed_atm() -> None:
    result = build_volatility_smiles(
        _chain(_quote(OptionType.CALL, 120.0, 0.18, 0.22)),
    )

    smile = result.smiles[0]
    record = volatility_smiles_to_records(result.smiles)[0]

    assert smile.nearest_atm_point is smile.points[0]
    assert smile.observed_atm_point is None
    assert not smile.has_observed_atm_point
    assert record["is_nearest_atm"] is True
    assert record["is_observed_atm"] is False


def test_inconsistent_market_state_rejects_whole_group_with_diagnostic() -> None:
    first = _chain(
        _quote(OptionType.PUT, 90.0, 0.18, 0.22),
        spot=100.0,
    )
    second = _chain(
        _quote(OptionType.CALL, 110.0, 0.18, 0.22),
        spot=101.0,
    )
    chain = ImpliedVolatilityChain(first.quotes + second.quotes)

    result = build_volatility_smiles(chain)
    records = smile_group_diagnostics_to_records(result.group_diagnostics)

    assert result.smiles == ()
    assert result.diagnostics == ()
    assert result.summary.input_quote_count == 2
    assert result.summary.selected_point_count == 0
    assert result.summary.excluded_quote_count == 2
    assert result.summary.group_diagnostic_count == 1
    assert result.summary.group_rejected_quote_count == 2
    assert result.summary.group_reason_counts == (
        (SmileGroupDiagnosticReason.INCONSISTENT_MARKET_STATE, 1),
    )
    diagnostic = result.group_diagnostics[0]
    assert diagnostic.reason is SmileGroupDiagnosticReason.INCONSISTENT_MARKET_STATE
    assert diagnostic.inconsistent_fields == ("spot_price", "forward_price")
    assert diagnostic.quote_count == 2
    assert records[0]["reason"] == "INCONSISTENT_MARKET_STATE"
    assert records[0]["inconsistent_fields"] == "spot_price|forward_price"


def test_market_state_consistency_uses_configurable_isclose_policy() -> None:
    first = _chain(
        _quote(OptionType.PUT, 90.0, 0.18, 0.22),
        spot=100.0,
    )
    second = _chain(
        _quote(OptionType.CALL, 110.0, 0.18, 0.22),
        spot=100.00000000005,
    )
    chain = ImpliedVolatilityChain(first.quotes + second.quotes)

    tolerant = build_volatility_smiles(chain)
    exact = build_volatility_smiles(
        chain,
        SmileSelectionConfig(
            market_state_relative_tolerance=0.0,
            market_state_absolute_tolerance=0.0,
        ),
    )

    assert tolerant.summary.smile_count == 1
    assert tolerant.group_diagnostics == ()
    assert exact.smiles == ()
    assert exact.group_diagnostics[0].inconsistent_fields == (
        "spot_price",
        "forward_price",
    )


def test_stale_inconsistent_quote_does_not_reject_latest_consistent_group() -> None:
    old_quote = _quote(
        OptionType.CALL,
        110.0,
        0.18,
        0.22,
        timestamp=AS_OF - timedelta(seconds=1),
        source_id="old-inconsistent",
    )
    latest_quote = _quote(
        OptionType.CALL,
        110.0,
        0.18,
        0.22,
        source_id="latest-consistent",
    )
    other_strike = _quote(OptionType.PUT, 90.0, 0.18, 0.22)
    old_chain = _chain(old_quote, spot=101.0)
    latest_chain = _chain(latest_quote, other_strike, spot=100.0)

    result = build_volatility_smiles(
        ImpliedVolatilityChain(old_chain.quotes + latest_chain.quotes),
    )

    assert result.summary.smile_count == 1
    assert result.summary.selected_point_count == 2
    assert result.summary.excluded_quote_count == 1
    assert result.group_diagnostics == ()
    assert result.diagnostics[0].iv_quote is old_chain.quotes[0]
    assert result.diagnostics[0].reasons == (
        SmileSelectionReason.STALE_QUOTE,
    )


def test_duplicate_strike_policy_is_explicit_when_otm_filter_is_disabled() -> None:
    liquid_itm_call = _quote(
        OptionType.CALL,
        90.0,
        0.199,
        0.201,
        bid_size=100,
        ask_size=100,
    )
    less_liquid_otm_put = _quote(
        OptionType.PUT,
        90.0,
        0.18,
        0.24,
        bid_size=1,
        ask_size=1,
    )
    chain = _chain(liquid_itm_call, less_liquid_otm_put)

    expected = {
        DuplicateStrikePolicy.PREFER_OTM: OptionType.PUT,
        DuplicateStrikePolicy.MOST_LIQUID: OptionType.CALL,
        DuplicateStrikePolicy.PREFER_CALL: OptionType.CALL,
        DuplicateStrikePolicy.PREFER_PUT: OptionType.PUT,
    }
    for policy, option_type in expected.items():
        result = build_volatility_smiles(
            chain,
            SmileSelectionConfig(
                otm_only=False,
                duplicate_strike_policy=policy,
            ),
        )
        assert result.smiles[0].points[0].option_type is option_type
        assert result.diagnostics[0].reasons == (
            SmileSelectionReason.DUPLICATE_STRIKE,
        )


def test_point_flags_explain_unavailable_delta_and_iv_spread(monkeypatch) -> None:
    zero_iv = OptionQuote(
        contract=_quote(OptionType.CALL, 100.0, 0.2, 0.2).contract,
        quote_timestamp=AS_OF,
        bid=0.0,
        ask=0.0,
    )
    boundary = build_volatility_smiles(
        _chain(zero_iv),
        SmileSelectionConfig(
            otm_only=False,
            excluded_diagnostic_flags=(),
        ),
    ).smiles[0].points[0]
    assert boundary.delta is None
    assert boundary.diagnostic_flags == (
        SmilePointDiagnosticFlag.DELTA_BOUNDARY_UNAVAILABLE,
    )

    one_sided = _quote(OptionType.CALL, 110.0, 0.18, 0.22)
    one_sided = OptionQuote(
        contract=one_sided.contract,
        quote_timestamp=AS_OF,
        bid=one_sided.bid,
        ask=None,
    )
    no_spread = build_volatility_smiles(
        _chain(one_sided),
        SmileSelectionConfig(
            iv_source=SmileIvSource.BID,
            require_two_sided_quote=False,
        ),
    ).smiles[0].points[0]
    assert no_spread.iv_bid_ask_spread is None
    assert no_spread.diagnostic_flags == (
        SmilePointDiagnosticFlag.IV_SPREAD_UNAVAILABLE,
    )

    import ncx_derivatives.volatility.smiles as smiles

    monkeypatch.setattr(
        smiles,
        "call_delta",
        lambda *args: (_ for _ in ()).throw(ValueError("delta failed")),
    )
    failed_delta = build_volatility_smiles(
        _chain(_quote(OptionType.CALL, 110.0, 0.18, 0.22)),
    ).smiles[0].points[0]
    assert failed_delta.delta is None
    assert failed_delta.diagnostic_flags == (
        SmilePointDiagnosticFlag.DELTA_NUMERICAL_FAILURE,
    )

    monkeypatch.setattr(smiles, "call_delta", lambda *args: nan)
    non_finite_delta = build_volatility_smiles(
        _chain(_quote(OptionType.CALL, 110.0, 0.18, 0.22)),
    ).smiles[0].points[0]
    assert non_finite_delta.delta is None
    assert non_finite_delta.diagnostic_flags == (
        SmilePointDiagnosticFlag.DELTA_NUMERICAL_FAILURE,
    )


@pytest.mark.parametrize("bad_iv", [True, nan, inf, -inf])
def test_public_smile_point_requires_finite_non_boolean_iv(bad_iv) -> None:
    iv_quote = _chain(_quote(OptionType.CALL, 110.0, 0.18, 0.22)).quotes[0]

    with pytest.raises(ValueError, match="non-negative and finite"):
        VolatilitySmilePoint(
            iv_quote=iv_quote,
            iv_source=SmileIvSource.MIDPOINT,
            implied_volatility=bad_iv,
            vega=None,
            delta=None,
            iv_bid_ask_spread=None,
            diagnostic_flags=(
                SmilePointDiagnosticFlag.DELTA_BOUNDARY_UNAVAILABLE,
                SmilePointDiagnosticFlag.IV_SPREAD_UNAVAILABLE,
            ),
        )


def test_public_smile_models_enforce_diagnostic_and_nearest_atm_invariants() -> None:
    result = build_volatility_smiles(
        _chain(
            _quote(OptionType.PUT, 90.0, 0.18, 0.22),
            _quote(OptionType.CALL, 110.0, 0.18, 0.22),
        ),
    )
    smile = result.smiles[0]
    iv_quote = smile.points[0].iv_quote

    with pytest.raises(ValueError, match="delta availability"):
        VolatilitySmilePoint(
            iv_quote=iv_quote,
            iv_source=SmileIvSource.MIDPOINT,
            implied_volatility=0.20,
            vega=None,
            delta=None,
            iv_bid_ask_spread=0.02,
        )

    wrong_index = 1 - smile.nearest_atm_index
    with pytest.raises(ValueError, match="nearest-forward"):
        VolatilitySmile(
            underlying_symbol=smile.underlying_symbol,
            valuation_timestamp=smile.valuation_timestamp,
            expiration=smile.expiration,
            time_to_maturity=smile.time_to_maturity,
            spot_price=smile.spot_price,
            forward_price=smile.forward_price,
            points=smile.points,
            nearest_atm_index=wrong_index,
            atm_log_moneyness_tolerance=smile.atm_log_moneyness_tolerance,
            market_state_relative_tolerance=(
                smile.market_state_relative_tolerance
            ),
            market_state_absolute_tolerance=(
                smile.market_state_absolute_tolerance
            ),
        )

    with pytest.raises(ValueError, match="smile group"):
        VolatilitySmile(
            underlying_symbol="MSFT",
            valuation_timestamp=smile.valuation_timestamp,
            expiration=smile.expiration,
            time_to_maturity=smile.time_to_maturity,
            spot_price=smile.spot_price,
            forward_price=smile.forward_price,
            points=smile.points,
            nearest_atm_index=smile.nearest_atm_index,
            atm_log_moneyness_tolerance=smile.atm_log_moneyness_tolerance,
            market_state_relative_tolerance=(
                smile.market_state_relative_tolerance
            ),
            market_state_absolute_tolerance=(
                smile.market_state_absolute_tolerance
            ),
        )

    with pytest.raises(ValueError, match="market state"):
        VolatilitySmile(
            underlying_symbol=smile.underlying_symbol,
            valuation_timestamp=smile.valuation_timestamp,
            expiration=smile.expiration,
            time_to_maturity=smile.time_to_maturity,
            spot_price=smile.spot_price,
            forward_price=smile.forward_price + 1.0,
            points=smile.points,
            nearest_atm_index=smile.nearest_atm_index,
            atm_log_moneyness_tolerance=smile.atm_log_moneyness_tolerance,
            market_state_relative_tolerance=(
                smile.market_state_relative_tolerance
            ),
            market_state_absolute_tolerance=(
                smile.market_state_absolute_tolerance
            ),
        )


def test_public_group_diagnostic_enforces_declared_group_identity() -> None:
    first = _chain(_quote(OptionType.PUT, 90.0, 0.18, 0.22), spot=100.0)
    second = _chain(_quote(OptionType.CALL, 110.0, 0.18, 0.22), spot=101.0)
    result = build_volatility_smiles(
        ImpliedVolatilityChain(first.quotes + second.quotes),
    )
    diagnostic = result.group_diagnostics[0]

    with pytest.raises(ValueError, match="diagnostic group"):
        SmileGroupDiagnostic(
            underlying_symbol="MSFT",
            valuation_timestamp=diagnostic.valuation_timestamp,
            expiration=diagnostic.expiration,
            reason=diagnostic.reason,
            inconsistent_fields=diagnostic.inconsistent_fields,
            iv_quotes=diagnostic.iv_quotes,
        )


def test_group_diagnostic_dataframe_has_stable_columns() -> None:
    pd = pytest.importorskip("pandas")
    first = _chain(_quote(OptionType.PUT, 90.0, 0.18, 0.22), spot=100.0)
    second = _chain(_quote(OptionType.CALL, 110.0, 0.18, 0.22), spot=101.0)
    result = build_volatility_smiles(
        ImpliedVolatilityChain(first.quotes + second.quotes),
    )

    frame = smile_group_diagnostics_to_dataframe(result.group_diagnostics)

    assert isinstance(frame, pd.DataFrame)
    assert tuple(frame.columns) == SMILE_GROUP_DIAGNOSTIC_COLUMNS
