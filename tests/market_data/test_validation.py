from datetime import date, datetime, timedelta, timezone

import pytest

from ncx_derivatives.market_data import (
    BuiltinValidationCode,
    ExerciseStyle,
    OptionChainSnapshot,
    OptionContract,
    OptionQuote,
    OptionTrade,
    OptionType,
    UnderlyingQuote,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
    validate_option_chain_snapshot,
    validate_option_contract,
    validate_option_quote,
    validate_option_trade,
    validate_underlying_quote,
)

UTC = timezone.utc
AS_OF = datetime(2026, 7, 30, 14, 30, tzinfo=UTC)


def _contract(
    option_type: OptionType = OptionType.CALL,
    strike: float = 180.0,
) -> OptionContract:
    return OptionContract(
        underlying_symbol="AAPL",
        expiration=date(2026, 8, 21),
        strike=strike,
        option_type=option_type,
        exercise_style=ExerciseStyle.AMERICAN,
        contract_multiplier=100.0,
        currency="USD",
    )


def test_validation_issue_requires_structured_severity() -> None:
    issue = ValidationIssue(
        severity=ValidationSeverity.WARNING,
        code="TEST_WARNING",
        message="test warning",
        location=("quotes", "0"),
    )

    assert issue.severity is ValidationSeverity.WARNING
    assert issue.location == ("quotes", "0")

    with pytest.raises(ValueError, match="severity"):
        ValidationIssue(  # type: ignore[arg-type]
            severity="warning",
            code="TEST_WARNING",
            message="test warning",
        )


def test_validation_issue_validates_code_message_location_and_context() -> None:
    with pytest.raises(ValueError, match="code"):
        ValidationIssue(  # type: ignore[arg-type]
            severity=ValidationSeverity.WARNING,
            code=None,
            message="test warning",
        )

    with pytest.raises(ValueError, match="message"):
        ValidationIssue(
            severity=ValidationSeverity.WARNING,
            code="TEST_WARNING",
            message="",
        )

    with pytest.raises(ValueError, match="location item"):
        ValidationIssue(
            severity=ValidationSeverity.WARNING,
            code="TEST_WARNING",
            message="test warning",
            location=("quotes", 1),  # type: ignore[list-item]
        )

    with pytest.raises(ValueError, match="context values"):
        ValidationIssue(
            severity=ValidationSeverity.WARNING,
            code="TEST_WARNING",
            message="test warning",
            context=(("bid", 1.0),),  # type: ignore[list-item]
        )

    with pytest.raises(ValueError, match="key-value"):
        ValidationIssue(
            severity=ValidationSeverity.WARNING,
            code="TEST_WARNING",
            message="test warning",
            context=("bid",),  # type: ignore[arg-type]
        )


def test_validation_report_splits_by_severity() -> None:
    report = ValidationReport(
        (
            ValidationIssue(
                ValidationSeverity.ERROR,
                "ERROR_CODE",
                "error message",
            ),
            ValidationIssue(
                ValidationSeverity.WARNING,
                "WARNING_CODE",
                "warning message",
            ),
            ValidationIssue(
                ValidationSeverity.INFO,
                "INFO_CODE",
                "info message",
            ),
        ),
    )

    assert report.has_errors
    assert report.has_warnings
    assert not report.is_valid
    assert [issue.code for issue in report.errors] == ["ERROR_CODE"]
    assert [issue.code for issue in report.warnings] == ["WARNING_CODE"]
    assert [issue.code for issue in report.infos] == ["INFO_CODE"]
    assert report.by_code("WARNING_CODE") == report.warnings
    assert report.at_location_prefix() == report.issues


def test_validation_report_validates_combine_inputs() -> None:
    report = ValidationReport(
        (
            ValidationIssue(
                ValidationSeverity.INFO,
                "INFO_CODE",
                "info message",
                location=("quotes", "0"),
            ),
        ),
    )

    combined = ValidationReport.combine(report)
    assert combined.at_location_prefix("quotes") == report.issues

    with pytest.raises(ValueError, match="ValidationReport"):
        ValidationReport.combine(report, "invalid")  # type: ignore[arg-type]


def test_contract_validation_reports_unknown_optional_identity_fields() -> None:
    report = validate_option_contract(
        OptionContract(
            underlying_symbol="AAPL",
            expiration=date(2026, 8, 21),
            strike=180.0,
            option_type=OptionType.CALL,
        ),
    )

    assert not report.has_errors
    assert {issue.code for issue in report.infos} == {
        "MISSING_EXERCISE_STYLE",
        "MISSING_CONTRACT_MULTIPLIER",
        "MISSING_CURRENCY",
    }


def test_quote_validation_separates_warnings_from_errors() -> None:
    missing_bid = OptionQuote(
        contract=_contract(),
        quote_timestamp=AS_OF,
        bid=None,
        ask=5.05,
    )
    empty_market = OptionQuote(
        contract=_contract(),
        quote_timestamp=AS_OF,
        bid=None,
        ask=None,
    )

    missing_bid_report = validate_option_quote(missing_bid)
    empty_market_report = validate_option_quote(empty_market)

    assert not missing_bid_report.has_errors
    assert "MISSING_BID" in {issue.code for issue in missing_bid_report.warnings}
    assert "EMPTY_MARKET" in {issue.code for issue in empty_market_report.errors}


def test_quote_validation_reports_crossed_locked_and_zero_markets() -> None:
    crossed = validate_option_quote(
        OptionQuote(
            contract=_contract(),
            quote_timestamp=AS_OF,
            bid=5.10,
            ask=5.00,
        ),
    )
    locked_zero = validate_option_quote(
        OptionQuote(
            contract=_contract(),
            quote_timestamp=AS_OF,
            bid=0.0,
            ask=0.0,
        ),
    )

    assert "CROSSED_MARKET" in {issue.code for issue in crossed.errors}
    assert "LOCKED_MARKET" in {issue.code for issue in locked_zero.warnings}
    assert {"ZERO_BID", "ZERO_ASK"} <= {
        issue.code for issue in locked_zero.infos
    }


def test_quote_validation_reports_zero_prices_independently() -> None:
    zero_bid_missing_ask = validate_option_quote(
        OptionQuote(
            contract=_contract(),
            quote_timestamp=AS_OF,
            bid=0.0,
            ask=None,
        ),
    )
    missing_bid_zero_ask = validate_option_quote(
        OptionQuote(
            contract=_contract(),
            quote_timestamp=AS_OF,
            bid=None,
            ask=0.0,
        ),
    )

    assert BuiltinValidationCode.ZERO_BID.value in {
        issue.code for issue in zero_bid_missing_ask.infos
    }
    assert BuiltinValidationCode.MISSING_ASK.value in {
        issue.code for issue in zero_bid_missing_ask.warnings
    }
    assert BuiltinValidationCode.ZERO_ASK.value in {
        issue.code for issue in missing_bid_zero_ask.infos
    }
    assert BuiltinValidationCode.MISSING_BID.value in {
        issue.code for issue in missing_bid_zero_ask.warnings
    }


def test_quote_validation_reports_open_interest_reference_issues() -> None:
    missing_date = validate_option_quote(
        OptionQuote(
            contract=_contract(),
            quote_timestamp=AS_OF,
            bid=4.90,
            ask=5.05,
            open_interest=1200,
            open_interest_date=None,
        ),
    )
    future_date = validate_option_quote(
        OptionQuote(
            contract=_contract(),
            quote_timestamp=AS_OF,
            bid=4.90,
            ask=5.05,
            open_interest=1200,
            open_interest_date=date(2026, 7, 31),
        ),
    )

    assert "OPEN_INTEREST_DATE_MISSING" in {
        issue.code for issue in missing_date.warnings
    }
    assert "OPEN_INTEREST_DATE_AFTER_QUOTE" in {
        issue.code for issue in future_date.warnings
    }


def test_trade_validation_reports_missing_and_zero_size() -> None:
    missing_size = validate_option_trade(
        OptionTrade(
            contract=_contract(),
            trade_timestamp=AS_OF,
            price=4.95,
            size=None,
        ),
    )
    zero_size = validate_option_trade(
        OptionTrade(
            contract=_contract(),
            trade_timestamp=AS_OF,
            price=4.95,
            size=0,
        ),
    )

    assert "MISSING_TRADE_SIZE" in {issue.code for issue in missing_size.infos}
    assert "ZERO_TRADE_SIZE" in {issue.code for issue in zero_size.infos}


def test_underlying_validation_reports_market_quality_issues() -> None:
    crossed = validate_underlying_quote(
        UnderlyingQuote(
            symbol="AAPL",
            quote_timestamp=AS_OF,
            price=181.22,
            bid=181.30,
            ask=181.20,
        ),
    )
    incomplete = validate_underlying_quote(
        UnderlyingQuote(
            symbol="AAPL",
            quote_timestamp=AS_OF,
            price=None,
            bid=181.20,
            ask=None,
        ),
    )

    assert "CROSSED_UNDERLYING_MARKET" in {
        issue.code for issue in crossed.errors
    }
    assert "INCOMPLETE_UNDERLYING_MARKET" in {
        issue.code for issue in incomplete.warnings
    }


def test_snapshot_validation_reports_duplicate_quotes() -> None:
    contract = _contract()
    snapshot = OptionChainSnapshot(
        underlying_symbol="AAPL",
        as_of=AS_OF,
        quotes=(
            OptionQuote(contract=contract, quote_timestamp=AS_OF, bid=4.9, ask=5.05),
            OptionQuote(contract=contract, quote_timestamp=AS_OF, bid=4.9, ask=5.05),
        ),
    )

    report = validate_option_chain_snapshot(snapshot)

    assert "DUPLICATE_QUOTE" in {issue.code for issue in report.errors}
    assert "MISSING_UNDERLYING_QUOTE" in {issue.code for issue in report.infos}


def test_snapshot_validation_allows_older_quote_and_underlying_timestamps() -> None:
    quote_timestamp = AS_OF - timedelta(minutes=1)
    underlying_timestamp = AS_OF - timedelta(seconds=30)
    snapshot = OptionChainSnapshot(
        underlying_symbol="AAPL",
        as_of=AS_OF,
        quotes=(
            OptionQuote(
                contract=_contract(),
                quote_timestamp=quote_timestamp,
                bid=4.9,
                ask=5.05,
            ),
        ),
        underlying_quote=UnderlyingQuote(
            symbol="AAPL",
            quote_timestamp=underlying_timestamp,
            price=181.22,
        ),
    )

    report = validate_option_chain_snapshot(snapshot)

    assert BuiltinValidationCode.QUOTE_TIMESTAMP_AFTER_SNAPSHOT.value not in {
        issue.code for issue in report.warnings
    }
    assert (
        BuiltinValidationCode.UNDERLYING_TIMESTAMP_AFTER_SNAPSHOT.value
        not in {issue.code for issue in report.warnings}
    )


def test_snapshot_validation_reports_future_timestamps() -> None:
    future_quote_timestamp = AS_OF + timedelta(seconds=1)
    future_trade_timestamp = AS_OF + timedelta(seconds=2)
    future_underlying_timestamp = AS_OF + timedelta(seconds=3)
    snapshot = OptionChainSnapshot(
        underlying_symbol="AAPL",
        as_of=AS_OF,
        quotes=(
            OptionQuote(
                contract=_contract(),
                quote_timestamp=future_quote_timestamp,
                bid=4.9,
                ask=5.05,
            ),
        ),
        trades=(
            OptionTrade(
                contract=_contract(),
                trade_timestamp=future_trade_timestamp,
                price=4.95,
                size=1,
            ),
        ),
        underlying_quote=UnderlyingQuote(
            symbol="AAPL",
            quote_timestamp=future_underlying_timestamp,
            price=181.22,
        ),
    )

    report = validate_option_chain_snapshot(snapshot)
    warning_codes = {issue.code for issue in report.warnings}

    assert BuiltinValidationCode.QUOTE_TIMESTAMP_AFTER_SNAPSHOT.value in warning_codes
    assert BuiltinValidationCode.TRADE_TIMESTAMP_AFTER_SNAPSHOT.value in warning_codes
    assert (
        BuiltinValidationCode.UNDERLYING_TIMESTAMP_AFTER_SNAPSHOT.value
        in warning_codes
    )


def test_snapshot_validation_prefixes_nested_issue_locations() -> None:
    snapshot = OptionChainSnapshot(
        underlying_symbol="AAPL",
        as_of=AS_OF,
        quotes=(
            OptionQuote(
                contract=_contract(),
                quote_timestamp=AS_OF,
                bid=None,
                ask=5.05,
            ),
        ),
    )

    report = validate_option_chain_snapshot(snapshot)
    missing_bid = next(
        issue
        for issue in report.issues
        if issue.code == "MISSING_BID"
    )

    assert missing_bid.location == ("quotes", "0", "bid")


def test_snapshot_duplicate_detection_is_per_contract_per_snapshot() -> None:
    contract = _contract()
    snapshot = OptionChainSnapshot(
        underlying_symbol="AAPL",
        as_of=AS_OF,
        quotes=(
            OptionQuote(
                contract=contract,
                quote_timestamp=AS_OF - timedelta(seconds=1),
                bid=4.9,
                ask=5.05,
            ),
            OptionQuote(
                contract=contract,
                quote_timestamp=AS_OF,
                bid=4.91,
                ask=5.06,
            ),
        ),
    )

    report = validate_option_chain_snapshot(snapshot)

    assert BuiltinValidationCode.DUPLICATE_QUOTE.value in {
        issue.code for issue in report.errors
    }
    duplicate_issue = report.by_code(BuiltinValidationCode.DUPLICATE_QUOTE.value)[0]
    assert duplicate_issue.context
