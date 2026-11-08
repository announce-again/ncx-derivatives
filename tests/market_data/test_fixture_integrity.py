import csv
import json
import math
from pathlib import Path

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "market_data"


def _read_csv(relative_path: str) -> list[dict[str, str]]:
    with (FIXTURE_ROOT / relative_path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _read_json(relative_path: str) -> dict:
    with (FIXTURE_ROOT / relative_path).open(encoding="utf-8") as file:
        return json.load(file)


def test_cboe_normal_contains_multiple_logical_snapshots() -> None:
    rows = _read_csv("cboe_intervals/normal.csv")

    quote_times = {row["Quote Datetime"] for row in rows}
    option_types = {row["Option Type"] for row in rows}
    expirations = {row["Expiration"] for row in rows}
    strikes = {row["Strike"] for row in rows}

    assert len(rows) == 6
    assert quote_times == {"2026-07-30T14:30:00Z", "2026-07-30T14:31:00Z"}
    assert option_types == {"C", "P"}
    assert len(expirations) >= 2
    assert len(strikes) >= 2


def test_cboe_quality_cases_cover_expected_quote_problems() -> None:
    rows = _read_csv("cboe_intervals/synthetic_quality_cases.csv")
    issues = {row["synthetic_issue"] for row in rows}

    assert "missing_bid" in issues
    assert "missing_ask" in issues
    assert "crossed_quote" in issues
    assert "locked_quote_missing_oi" in issues
    assert "zero_bid_ask_zero_volume" in issues

    crossed = next(row for row in rows if row["synthetic_issue"] == "crossed_quote")
    locked = next(row for row in rows if row["synthetic_issue"] == "locked_quote_missing_oi")
    zero_quote = next(row for row in rows if row["synthetic_issue"] == "zero_bid_ask_zero_volume")

    assert float(crossed["Bid"]) > float(crossed["Ask"])
    assert float(locked["Bid"]) == float(locked["Ask"])
    assert float(zero_quote["Bid"]) == 0.0
    assert float(zero_quote["Ask"]) == 0.0
    assert int(zero_quote["Trade Volume"]) == 0
    assert locked["Open Interest"] == ""


def test_massive_normal_contains_pagination_and_nested_chain_data() -> None:
    payload = _read_json("massive_snapshot/normal.json")

    assert payload["status"] == "OK"
    assert payload["next_url"]
    assert len(payload["results"]) == 2

    option_types = {item["details"]["contract_type"] for item in payload["results"]}
    strikes = {item["details"]["strike_price"] for item in payload["results"]}

    assert option_types == {"call", "put"}
    assert strikes == {180}
    for item in payload["results"]:
        assert "last_quote" in item
        assert "last_trade" in item
        assert "underlying_asset" in item
        assert isinstance(item["last_quote"]["last_updated"], int)


def test_massive_missing_optional_fields_are_intentional() -> None:
    payload = _read_json("massive_snapshot/missing_optional_fields.json")
    first, second = payload["results"]

    assert "last_trade" not in first
    assert "greeks" not in first
    assert first["implied_volatility"] == 5

    assert "last_quote" not in second
    assert "greeks" not in second
    assert "implied_volatility" not in second
    assert second["open_interest"] is None


def test_databento_separated_fixtures_join_by_instrument_id() -> None:
    definitions = _read_csv("databento_separated/definitions.csv")
    bbo = _read_csv("databento_separated/bbo.csv")
    statistics = _read_csv("databento_separated/statistics.csv")

    definition_ids = {row["instrument_id"] for row in definitions}
    bbo_ids = {row["instrument_id"] for row in bbo}
    statistics_ids = {row["instrument_id"] for row in statistics}

    assert bbo_ids <= definition_ids
    assert statistics_ids <= definition_ids
    assert {"1001", "1002", "1003"} <= definition_ids


def test_databento_null_price_sentinel_case_is_present() -> None:
    rows = _read_csv("databento_separated/bbo.csv")
    null_price = next(
        row
        for row in rows
        if row["synthetic_issue"] == "null_bid_price_sentinel_converted_to_nan"
    )

    assert math.isnan(float(null_price["bid_px_00"]))
    assert int(null_price["bid_sz_00"]) == 0
    assert int(null_price["raw_bid_px_00"]) == 9223372036854775807


def test_databento_open_interest_has_reference_date() -> None:
    rows = _read_csv("databento_separated/statistics.csv")

    assert rows
    for row in rows:
        assert row["stat_type"] == "open_interest"
        assert row["as_of_date"] == "2026-07-29"
