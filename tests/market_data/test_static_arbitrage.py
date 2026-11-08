from dataclasses import FrozenInstanceError
from datetime import date, datetime, timezone

import pytest

from ncx_derivatives.market_data import (
    EnrichedOptionQuote,
    ExerciseStyle,
    NoArbitrageBounds,
    OptionContract,
    OptionQuote,
    OptionType,
    StaticArbitrageCode,
    StaticArbitrageConfig,
    StaticArbitrageDiagnostic,
    StaticArbitrageReport,
    ValidationSeverity,
    diagnose_static_arbitrage,
)

UTC = timezone.utc
VALUATION_TIMESTAMP = datetime(2026, 1, 1, 15, 0, tzinfo=UTC)
VALUATION_DATE = date(2026, 1, 1)


def _quote(
    option_type: OptionType,
    strike: float,
    expiration: date = date(2026, 7, 1),
    midpoint: float = 5.0,
    lower_bound: float = 0.0,
    upper_bound: float = 200.0,
    maturity: float = 0.5,
    risk_free_discount: float = 0.98,
    forward: float = 100.0,
    exercise_style: ExerciseStyle | None = ExerciseStyle.EUROPEAN,
    contract_multiplier: float | None = 100.0,
    currency: str | None = "USD",
    spot: float = 100.0,
    valuation_timestamp: datetime = VALUATION_TIMESTAMP,
) -> EnrichedOptionQuote:
    bid = max(midpoint - 0.05, 0.0)
    ask = midpoint + 0.05
    contract = OptionContract(
        underlying_symbol="AAPL",
        expiration=expiration,
        strike=strike,
        option_type=option_type,
        exercise_style=exercise_style,
        contract_multiplier=contract_multiplier,
        currency=currency,
    )
    option_quote = OptionQuote(
        contract=contract,
        quote_timestamp=valuation_timestamp,
        bid=bid,
        ask=ask,
    )
    return EnrichedOptionQuote(
        quote=option_quote,
        valuation_timestamp=valuation_timestamp,
        valuation_date=VALUATION_DATE,
        spot_price=spot,
        time_to_maturity=maturity,
        midpoint=midpoint,
        absolute_spread=ask - bid,
        relative_spread=(ask - bid) / midpoint if midpoint > 0.0 else None,
        risk_free_discount_factor=risk_free_discount,
        dividend_discount_factor=0.99,
        forward_price=forward,
        spot_moneyness=spot / strike,
        forward_moneyness=forward / strike,
        log_moneyness=0.0,
        intrinsic_value=0.0,
        time_value=midpoint,
        no_arbitrage_bounds=NoArbitrageBounds(lower_bound, upper_bound),
    )


def test_single_contract_bound_diagnostics_quantify_violations() -> None:
    below = _quote(OptionType.CALL, 100.0, midpoint=1.0, lower_bound=2.5)
    above = _quote(OptionType.PUT, 100.0, midpoint=12.0, upper_bound=10.0)

    report = diagnose_static_arbitrage(
        (below, above),
        StaticArbitrageConfig(enable_call_put_parity=False),
    )

    assert report.has_violations
    assert report.violation_count == 2
    assert report.max_violation_amount == pytest.approx(2.0)
    assert report.by_code(StaticArbitrageCode.PRICE_BELOW_LOWER_BOUND)[
        0
    ].violation_amount == pytest.approx(1.5)
    assert report.by_code(StaticArbitrageCode.PRICE_ABOVE_UPPER_BOUND)[
        0
    ].violation_amount == pytest.approx(2.0)


def test_call_put_parity_diagnostic_uses_discounted_forward_strike_relation() -> None:
    call = _quote(
        OptionType.CALL,
        100.0,
        midpoint=8.0,
        risk_free_discount=0.95,
        forward=105.0,
    )
    put = _quote(
        OptionType.PUT,
        100.0,
        midpoint=2.0,
        risk_free_discount=0.95,
        forward=105.0,
    )

    report = diagnose_static_arbitrage(
        (call, put),
        StaticArbitrageConfig(
            enable_single_contract_bounds=False,
            parity_tolerance=0.01,
        ),
    )

    diagnostic = report.by_code(
        StaticArbitrageCode.CALL_PUT_PARITY_VIOLATION,
    )[0]

    assert diagnostic.violation_amount == pytest.approx(1.25)
    assert dict(diagnostic.context)["left"] == "6.0"
    assert dict(diagnostic.context)["right"] == "4.75"


@pytest.mark.parametrize(
    "exercise_style",
    [ExerciseStyle.AMERICAN, ExerciseStyle.BERMUDAN, None],
)
def test_call_put_parity_skips_non_european_options(
    exercise_style: ExerciseStyle | None,
) -> None:
    call = _quote(
        OptionType.CALL,
        100.0,
        midpoint=8.0,
        risk_free_discount=0.95,
        forward=105.0,
        exercise_style=exercise_style,
    )
    put = _quote(
        OptionType.PUT,
        100.0,
        midpoint=2.0,
        risk_free_discount=0.95,
        forward=105.0,
        exercise_style=exercise_style,
    )

    report = diagnose_static_arbitrage(
        (call, put),
        StaticArbitrageConfig(enable_single_contract_bounds=False),
    )

    assert not report.by_code(StaticArbitrageCode.CALL_PUT_PARITY_VIOLATION)


def test_strike_monotonicity_and_vertical_spread_diagnostics() -> None:
    call_low = _quote(OptionType.CALL, 90.0, midpoint=1.0)
    call_high = _quote(OptionType.CALL, 100.0, midpoint=3.0)
    put_low = _quote(OptionType.PUT, 90.0, midpoint=5.0)
    put_high = _quote(OptionType.PUT, 100.0, midpoint=2.0)

    report = diagnose_static_arbitrage(
        (call_low, call_high, put_low, put_high),
        StaticArbitrageConfig(
            enable_single_contract_bounds=False,
            enable_call_put_parity=False,
        ),
    )

    assert report.by_code(StaticArbitrageCode.CALL_MONOTONICITY_VIOLATION)
    assert report.by_code(StaticArbitrageCode.CALL_VERTICAL_SPREAD_BOUND_VIOLATION)
    assert report.by_code(StaticArbitrageCode.PUT_MONOTONICITY_VIOLATION)
    assert report.by_code(StaticArbitrageCode.PUT_VERTICAL_SPREAD_BOUND_VIOLATION)


def test_strike_diagnostics_do_not_mix_contract_families() -> None:
    standard = _quote(
        OptionType.CALL,
        100.0,
        midpoint=1.0,
        contract_multiplier=100.0,
    )
    adjusted = _quote(
        OptionType.CALL,
        110.0,
        midpoint=3.0,
        contract_multiplier=10.0,
    )

    report = diagnose_static_arbitrage(
        (standard, adjusted),
        StaticArbitrageConfig(
            enable_single_contract_bounds=False,
            enable_call_put_parity=False,
        ),
    )

    assert not report.has_violations


def test_single_contract_diagnostic_preserves_original_input_index() -> None:
    missing_midpoint = _quote(OptionType.CALL, 90.0, midpoint=0.0)
    missing_midpoint = EnrichedOptionQuote(
        quote=missing_midpoint.quote,
        valuation_timestamp=missing_midpoint.valuation_timestamp,
        valuation_date=missing_midpoint.valuation_date,
        spot_price=missing_midpoint.spot_price,
        time_to_maturity=missing_midpoint.time_to_maturity,
        midpoint=None,
        absolute_spread=None,
        relative_spread=None,
        risk_free_discount_factor=missing_midpoint.risk_free_discount_factor,
        dividend_discount_factor=missing_midpoint.dividend_discount_factor,
        forward_price=missing_midpoint.forward_price,
        spot_moneyness=missing_midpoint.spot_moneyness,
        forward_moneyness=missing_midpoint.forward_moneyness,
        log_moneyness=missing_midpoint.log_moneyness,
        intrinsic_value=missing_midpoint.intrinsic_value,
        time_value=None,
        no_arbitrage_bounds=missing_midpoint.no_arbitrage_bounds,
    )
    below = _quote(OptionType.CALL, 100.0, midpoint=1.0, lower_bound=2.0)

    report = diagnose_static_arbitrage(
        (missing_midpoint, below),
        StaticArbitrageConfig(enable_call_put_parity=False),
    )

    assert report.by_code(StaticArbitrageCode.PRICE_BELOW_LOWER_BOUND)[
        0
    ].location == ("quotes", "1")


def test_vertical_spread_upper_bound_diagnostics() -> None:
    call_low = _quote(
        OptionType.CALL,
        90.0,
        midpoint=20.0,
        risk_free_discount=0.95,
    )
    call_high = _quote(
        OptionType.CALL,
        100.0,
        midpoint=1.0,
        risk_free_discount=0.95,
    )

    report = diagnose_static_arbitrage(
        (call_low, call_high),
        StaticArbitrageConfig(
            enable_single_contract_bounds=False,
            enable_call_put_parity=False,
        ),
    )

    diagnostic = report.by_code(
        StaticArbitrageCode.CALL_VERTICAL_SPREAD_BOUND_VIOLATION,
    )[0]

    assert diagnostic.violation_amount == pytest.approx(9.5)


def test_european_vertical_spread_uses_discounted_strike_gap() -> None:
    call_low = _quote(
        OptionType.CALL,
        100.0,
        midpoint=10.0,
        risk_free_discount=0.95,
        exercise_style=ExerciseStyle.EUROPEAN,
    )
    call_high = _quote(
        OptionType.CALL,
        110.0,
        midpoint=0.2,
        risk_free_discount=0.95,
        exercise_style=ExerciseStyle.EUROPEAN,
    )

    report = diagnose_static_arbitrage(
        (call_low, call_high),
        StaticArbitrageConfig(
            enable_single_contract_bounds=False,
            enable_call_put_parity=False,
            enable_strike_monotonicity=False,
        ),
    )

    diagnostic = report.by_code(
        StaticArbitrageCode.CALL_VERTICAL_SPREAD_BOUND_VIOLATION,
    )[0]

    assert diagnostic.violation_amount == pytest.approx(0.3)


def test_american_vertical_spread_is_not_compared_with_discounted_gap() -> None:
    call_low = _quote(
        OptionType.CALL,
        100.0,
        midpoint=10.0,
        risk_free_discount=0.95,
        exercise_style=ExerciseStyle.AMERICAN,
    )
    call_high = _quote(
        OptionType.CALL,
        110.0,
        midpoint=0.2,
        risk_free_discount=0.95,
        exercise_style=ExerciseStyle.AMERICAN,
    )

    report = diagnose_static_arbitrage(
        (call_low, call_high),
        StaticArbitrageConfig(
            enable_single_contract_bounds=False,
            enable_call_put_parity=False,
            enable_strike_monotonicity=False,
        ),
    )

    assert not report.by_code(
        StaticArbitrageCode.CALL_VERTICAL_SPREAD_BOUND_VIOLATION,
    )


@pytest.mark.parametrize(
    "exercise_style",
    [ExerciseStyle.BERMUDAN, None],
)
def test_unknown_and_bermudan_vertical_bounds_are_skipped(
    exercise_style: ExerciseStyle | None,
) -> None:
    put_low = _quote(
        OptionType.PUT,
        100.0,
        midpoint=0.2,
        risk_free_discount=0.95,
        exercise_style=exercise_style,
    )
    put_high = _quote(
        OptionType.PUT,
        110.0,
        midpoint=10.0,
        risk_free_discount=0.95,
        exercise_style=exercise_style,
    )

    report = diagnose_static_arbitrage(
        (put_low, put_high),
        StaticArbitrageConfig(
            enable_single_contract_bounds=False,
            enable_call_put_parity=False,
            enable_strike_monotonicity=False,
        ),
    )

    assert not report.by_code(
        StaticArbitrageCode.PUT_VERTICAL_SPREAD_BOUND_VIOLATION,
    )


def test_convexity_and_butterfly_diagnostics_are_reported_without_repair() -> None:
    left = _quote(OptionType.CALL, 90.0, midpoint=10.0)
    middle = _quote(OptionType.CALL, 100.0, midpoint=11.0)
    right = _quote(OptionType.CALL, 110.0, midpoint=8.0)

    report = diagnose_static_arbitrage(
        (left, middle, right),
        StaticArbitrageConfig(
            enable_single_contract_bounds=False,
            enable_call_put_parity=False,
            enable_strike_monotonicity=False,
            enable_vertical_spread_bounds=False,
        ),
    )

    assert report.by_code(StaticArbitrageCode.CONVEXITY_VIOLATION)[
        0
    ].violation_amount == pytest.approx(2.0)
    assert report.by_code(StaticArbitrageCode.BUTTERFLY_ARBITRAGE)[
        0
    ].violation_amount == pytest.approx(2.0)
    assert middle.midpoint == 11.0


def test_heuristic_calendar_consistency_is_disabled_by_default() -> None:
    near = _quote(
        OptionType.CALL,
        100.0,
        expiration=date(2026, 3, 1),
        midpoint=6.0,
        maturity=0.15,
    )
    far = _quote(
        OptionType.CALL,
        100.0,
        expiration=date(2026, 9, 1),
        midpoint=4.0,
        maturity=0.65,
    )

    default_report = diagnose_static_arbitrage(
        (near, far),
        StaticArbitrageConfig(
            enable_single_contract_bounds=False,
            enable_call_put_parity=False,
            enable_strike_monotonicity=False,
            enable_vertical_spread_bounds=False,
            enable_convexity=False,
            enable_butterfly=False,
        ),
    )

    assert not default_report.has_violations

    enabled_report = diagnose_static_arbitrage(
        (near, far),
        StaticArbitrageConfig(
            enable_single_contract_bounds=False,
            enable_call_put_parity=False,
            enable_strike_monotonicity=False,
            enable_vertical_spread_bounds=False,
            enable_convexity=False,
            enable_butterfly=False,
            enable_heuristic_calendar_consistency=True,
        ),
    )

    assert enabled_report.by_code(
        StaticArbitrageCode.HEURISTIC_CALENDAR_CONSISTENCY_VIOLATION,
    )[
        0
    ].violation_amount == pytest.approx(2.0)


def test_calendar_diagnostics_do_not_mix_contract_families() -> None:
    near = _quote(
        OptionType.CALL,
        100.0,
        expiration=date(2026, 3, 1),
        midpoint=6.0,
        maturity=0.15,
        contract_multiplier=100.0,
    )
    far_adjusted = _quote(
        OptionType.CALL,
        100.0,
        expiration=date(2026, 9, 1),
        midpoint=4.0,
        maturity=0.65,
        contract_multiplier=10.0,
    )

    report = diagnose_static_arbitrage(
        (near, far_adjusted),
        StaticArbitrageConfig(
            enable_single_contract_bounds=False,
            enable_call_put_parity=False,
            enable_strike_monotonicity=False,
            enable_vertical_spread_bounds=False,
            enable_convexity=False,
            enable_butterfly=False,
            enable_heuristic_calendar_consistency=True,
        ),
    )

    assert not report.has_violations


def test_report_is_clean_when_prices_satisfy_enabled_rules() -> None:
    call_low = _quote(OptionType.CALL, 90.0, midpoint=12.0)
    call_mid = _quote(OptionType.CALL, 100.0, midpoint=7.0)
    call_high = _quote(OptionType.CALL, 110.0, midpoint=4.0)

    report = diagnose_static_arbitrage(
        (call_low, call_mid, call_high),
        StaticArbitrageConfig(
            enable_single_contract_bounds=False,
            enable_call_put_parity=False,
        ),
    )

    assert not report.has_violations
    assert report.max_violation_amount == 0.0


def test_static_arbitrage_models_validate_inputs_and_are_immutable() -> None:
    with pytest.raises(ValueError, match="bound_tolerance"):
        StaticArbitrageConfig(bound_tolerance=-0.1)

    with pytest.raises(ValueError, match="code"):
        StaticArbitrageDiagnostic(
            severity=ValidationSeverity.WARNING,
            code="INVALID",  # type: ignore[arg-type]
            message="invalid",
            violation_amount=1.0,
        )

    config = StaticArbitrageConfig()
    with pytest.raises(FrozenInstanceError):
        config.parity_tolerance = 1.0  # type: ignore[misc]


def test_diagnose_static_arbitrage_validates_inputs() -> None:
    with pytest.raises(ValueError, match="enriched_quotes"):
        diagnose_static_arbitrage((_quote(OptionType.CALL, 100.0).quote,))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="config"):
        diagnose_static_arbitrage((), config="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="code"):
        StaticArbitrageReport().by_code("INVALID")  # type: ignore[arg-type]


def test_diagnose_static_arbitrage_rejects_mixed_global_valuation_state() -> None:
    first = _quote(OptionType.CALL, 100.0, spot=100.0)
    second = _quote(OptionType.CALL, 110.0, spot=101.0)

    with pytest.raises(ValueError, match="spot_price"):
        diagnose_static_arbitrage((first, second))


def test_same_expiry_groups_reject_mixed_carry_assumptions() -> None:
    first = _quote(OptionType.CALL, 100.0, risk_free_discount=0.98)
    second = _quote(OptionType.CALL, 110.0, risk_free_discount=0.97)

    with pytest.raises(ValueError, match="risk_free_discount_factor"):
        diagnose_static_arbitrage(
            (first, second),
            StaticArbitrageConfig(
                enable_single_contract_bounds=False,
                enable_call_put_parity=False,
            ),
        )
