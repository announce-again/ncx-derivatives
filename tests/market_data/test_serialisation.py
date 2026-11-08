import csv
import json
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from ncx_derivatives.market_data import (
    CleaningConfig,
    CleaningDiagnostic,
    CleaningResult,
    DatasetManifest,
    ExerciseStyle,
    NoArbitrageBounds,
    OptionChainSnapshot,
    OptionContract,
    OptionQuote,
    OptionTrade,
    OptionType,
    RejectedQuote,
    RejectionReason,
    SourceMetadata,
    StaticArbitrageConfig,
    UnderlyingQuote,
    ValidationSeverity,
    canonical_snapshot_to_json,
    clean_option_chain,
    diagnose_static_arbitrage,
    validate_option_chain_snapshot,
    write_canonical_json,
    write_market_data_dataset,
    write_option_chain_csv,
)
from ncx_derivatives.market_data.derived import EnrichedOptionQuote

UTC = timezone.utc
AS_OF = datetime(2026, 7, 30, 14, 30, tzinfo=UTC)
INGESTED_AT = datetime(2026, 7, 30, 14, 31, tzinfo=UTC)


def _contract(
    option_type: OptionType,
    strike: float,
    multiplier: float = 100.0,
) -> OptionContract:
    return OptionContract(
        underlying_symbol="AAPL",
        expiration=date(2026, 8, 21),
        strike=strike,
        option_type=option_type,
        exercise_style=ExerciseStyle.EUROPEAN,
        contract_multiplier=multiplier,
        currency="USD",
        display_symbol=f"AAPL {option_type.value} {strike} x{multiplier}",
        metadata=SourceMetadata(provider="fixture"),
    )


def _snapshot() -> OptionChainSnapshot:
    return OptionChainSnapshot(
        underlying_symbol="AAPL",
        as_of=AS_OF,
        quotes=(
            OptionQuote(
                contract=_contract(OptionType.CALL, 100.0),
                quote_timestamp=AS_OF,
                bid=None,
                ask=5.1,
                open_interest=100,
                open_interest_date=date(2026, 7, 29),
            ),
            OptionQuote(
                contract=_contract(OptionType.PUT, 100.0),
                quote_timestamp=AS_OF,
                bid=2.0,
                ask=2.2,
                open_interest=100,
                open_interest_date=date(2026, 7, 29),
            ),
        ),
        underlying_quote=UnderlyingQuote(
            symbol="AAPL",
            quote_timestamp=AS_OF,
            price=101.0,
            bid=100.9,
            ask=101.1,
        ),
        metadata=SourceMetadata(provider="fixture", dataset="snapshot"),
    )


def _bad_enriched_quote() -> EnrichedOptionQuote:
    quote = _snapshot().quotes[0]
    return EnrichedOptionQuote(
        quote=quote,
        valuation_timestamp=AS_OF,
        valuation_date=date(2026, 7, 30),
        spot_price=101.0,
        time_to_maturity=22.0 / 365.0,
        midpoint=500.0,
        absolute_spread=None,
        relative_spread=None,
        risk_free_discount_factor=0.99,
        dividend_discount_factor=1.0,
        forward_price=102.0,
        spot_moneyness=101.0 / quote.contract.strike,
        forward_moneyness=102.0 / quote.contract.strike,
        log_moneyness=0.0,
        intrinsic_value=1.0,
        time_value=499.0,
        no_arbitrage_bounds=NoArbitrageBounds(0.0, 101.0),
    )


def test_canonical_snapshot_to_json_is_serialisable() -> None:
    payload = canonical_snapshot_to_json(_snapshot())

    encoded = json.dumps(payload)

    assert payload["schema_version"] == "1.0"
    assert payload["underlying_symbol"] == "AAPL"
    assert payload["as_of"] == "2026-07-30T14:30:00Z"
    assert len(payload["records"]) == 2
    assert "AAPL" in encoded


def test_canonical_json_and_identity_preserve_trades_and_underlying_without_quotes(tmp_path):
    snapshot = replace(_snapshot(), quotes=())
    trade = OptionTrade(_contract(OptionType.CALL, 100.0), AS_OF, price=4.5, size=8)
    with_trade = replace(snapshot, trades=(trade,))
    first = write_market_data_dataset(tmp_path / "first", snapshot, source="fixture")
    second = write_market_data_dataset(tmp_path / "second", with_trade, source="fixture")

    payload = json.loads(second.path_map["canonical_json"].read_text(encoding="utf-8"))
    assert payload["records"] == []
    assert payload["trades"][0]["trade_price"] == 4.5
    assert payload["underlying_quote"]["price"] == 101.0
    assert payload["metadata"]["dataset"] == "snapshot"
    assert first.manifest.dataset_id != second.manifest.dataset_id


@pytest.mark.parametrize("fail_during_publication", [False, True])
def test_failed_dataset_rewrite_keeps_original_files(tmp_path, monkeypatch, fail_during_publication):
    root = tmp_path / "dataset"
    snapshot = _snapshot()
    original = write_market_data_dataset(root, snapshot, source="fixture")
    before = {name: path.read_bytes() for name, path in original.paths}
    (root / "notes.txt").write_text("keep", encoding="utf-8")
    updated = replace(snapshot, quotes=(replace(snapshot.quotes[0], ask=9.0),))
    kwargs = {}
    if fail_during_publication:
        replace_file = Path.replace

        def fail_replacing_canonical_json(path, target):
            if path.name == "canonical_option_chain.json" and "staged" in path.parts:
                assert not (root / "manifest.json").exists()
                assert original.path_map["canonical_csv"].read_bytes() != before["canonical_csv"]
                raise OSError("publication failed")
            return replace_file(path, target)

        monkeypatch.setattr(Path, "replace", fail_replacing_canonical_json)
        error, message = OSError, "publication failed"
    else:
        kwargs["cleaning_config"] = {"max_relative_spread": float("nan")}
        error, message = ValueError, "configuration floats must be finite"

    with pytest.raises(error, match=message):
        write_market_data_dataset(root, updated, source="fixture", **kwargs)

    assert {name: path.read_bytes() for name, path in original.paths} == before
    assert (root / "notes.txt").read_text(encoding="utf-8") == "keep"


def test_write_canonical_json_and_csv(tmp_path) -> None:
    snapshot = _snapshot()
    json_path = write_canonical_json(snapshot, tmp_path / "snapshot.json")
    csv_path = write_option_chain_csv(snapshot, tmp_path / "snapshot.csv")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    with csv_path.open(newline="", encoding="utf-8") as file:
        rows = tuple(csv.DictReader(file))

    assert payload["schema_version"] == "1.0"
    assert len(rows) == 2
    assert rows[0]["underlying_symbol"] == "AAPL"


def test_write_market_data_dataset_creates_target_layout_and_manifest(tmp_path) -> None:
    raw_source = tmp_path / "source.csv"
    raw_source.write_text("symbol,bid\nAAPL,1.0\n", encoding="utf-8")
    validation = validate_option_chain_snapshot(_snapshot())
    cleaning = clean_option_chain(_snapshot(), CleaningConfig(reject_missing_bid=True))
    arbitrage = diagnose_static_arbitrage(
        (_bad_enriched_quote(),),
        StaticArbitrageConfig(enable_call_put_parity=False),
    )

    result = write_market_data_dataset(
        tmp_path / "dataset",
        _snapshot(),
        source="test_fixture",
        raw_source_path=raw_source,
        validation_report=validation,
        cleaning_result=cleaning,
        static_arbitrage_report=arbitrage,
        ingestion_timestamp=INGESTED_AT,
        valuation_timestamp=AS_OF,
        day_count="ACT/365F",
        source_information={"provider": "fixture"},
        normalisation_config={"timezone": "UTC"},
        cleaning_config={"reject_missing_bid": True},
        rate_dividend_assumptions={"risk_free_rate": 0.04, "dividend_yield": 0.01},
    )

    paths = result.path_map
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))

    assert isinstance(result.manifest, DatasetManifest)
    assert paths["raw_source"].is_file()
    assert paths["canonical_csv"].is_file()
    assert paths["canonical_json"].is_file()
    assert paths["accepted_csv"].is_file()
    assert paths["rejected_quotes_csv"].is_file()
    assert paths["validation_json"].is_file()
    assert paths["cleaning_json"].is_file()
    assert paths["arbitrage_json"].is_file()
    assert manifest["schema_version"] == "1.0"
    assert manifest["source"] == "test_fixture"
    assert manifest["as_of"] == "2026-07-30T14:30:00Z"
    assert manifest["input_quote_count"] == len(_snapshot().quotes)
    assert manifest["canonical_quote_count"] == len(_snapshot().quotes)
    assert manifest["accepted_quote_count"] == cleaning.accepted_count
    assert manifest["rejected_quote_count"] == cleaning.rejected_count
    assert manifest["accepted_quote_count"] + manifest["rejected_quote_count"] == (
        manifest["input_quote_count"]
    )
    assert manifest["validation_issue_count"] == len(validation.issues)
    assert manifest["cleaning_diagnostic_count"] == len(cleaning.diagnostics)
    assert manifest["arbitrage_diagnostic_count"] == len(arbitrage.diagnostics)
    assert manifest["cleaning_config"]["reject_missing_bid"] is True
    assert manifest["rate_dividend_assumptions"]["risk_free_rate"] == 0.04
    assert len(manifest["input_hash"]) == 64
    assert len(manifest["output_hash"]) == 64
    assert len(manifest["dataset_id"]) == 64

    with paths["canonical_csv"].open(newline="", encoding="utf-8") as file:
        canonical_rows = tuple(csv.DictReader(file))
    with paths["accepted_csv"].open(newline="", encoding="utf-8") as file:
        accepted_rows = tuple(csv.DictReader(file))
    with paths["rejected_quotes_csv"].open(newline="", encoding="utf-8") as file:
        rejected_rows = tuple(csv.DictReader(file))

    assert len(canonical_rows) == manifest["canonical_quote_count"]
    assert len(accepted_rows) == manifest["accepted_quote_count"]
    assert len(rejected_rows) == manifest["rejected_quote_count"]
    assert canonical_rows[0]["snapshot_as_of"] == "2026-07-30T14:30:00Z"


def test_dataset_identifiers_are_deterministic_for_same_inputs(tmp_path) -> None:
    first = write_market_data_dataset(
        tmp_path / "first",
        _snapshot(),
        source="test_fixture",
        ingestion_timestamp=INGESTED_AT,
    )
    second = write_market_data_dataset(
        tmp_path / "second",
        _snapshot(),
        source="test_fixture",
        ingestion_timestamp=INGESTED_AT,
    )

    assert first.manifest.input_hash == second.manifest.input_hash
    assert first.manifest.output_hash == second.manifest.output_hash
    assert first.manifest.dataset_id == second.manifest.dataset_id


def test_dataset_id_does_not_depend_on_ingestion_timestamp(tmp_path) -> None:
    first = write_market_data_dataset(
        tmp_path / "first",
        _snapshot(),
        source="test_fixture",
        ingestion_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )
    second = write_market_data_dataset(
        tmp_path / "second",
        _snapshot(),
        source="test_fixture",
        ingestion_timestamp=datetime(2026, 1, 2, tzinfo=UTC),
    )

    assert first.manifest.dataset_id == second.manifest.dataset_id
    assert first.manifest.ingestion_timestamp != second.manifest.ingestion_timestamp


def test_typed_configuration_values_remain_distinct_in_dataset_id(tmp_path) -> None:
    integer_config = write_market_data_dataset(
        tmp_path / "integer",
        _snapshot(),
        source="test_fixture",
        ingestion_timestamp=INGESTED_AT,
        cleaning_config={"min_open_interest": 1},
    )
    string_config = write_market_data_dataset(
        tmp_path / "string",
        _snapshot(),
        source="test_fixture",
        ingestion_timestamp=INGESTED_AT,
        cleaning_config={"min_open_interest": "1"},
    )

    assert integer_config.manifest.dataset_id != string_config.manifest.dataset_id
    assert dict(integer_config.manifest.cleaning_config)["min_open_interest"] == 1
    assert dict(string_config.manifest.cleaning_config)["min_open_interest"] == "1"


def test_partition_records_contribute_to_output_hash_and_dataset_id(tmp_path) -> None:
    standard = OptionQuote(
        contract=_contract(OptionType.CALL, 100.0, multiplier=100.0),
        quote_timestamp=AS_OF,
        bid=None,
        ask=5.1,
    )
    adjusted = OptionQuote(
        contract=_contract(OptionType.CALL, 100.0, multiplier=10.0),
        quote_timestamp=AS_OF,
        bid=None,
        ask=5.1,
    )
    snapshot = OptionChainSnapshot(
        underlying_symbol="AAPL",
        as_of=AS_OF,
        quotes=(standard, adjusted),
    )
    diagnostic = CleaningDiagnostic(
        severity=ValidationSeverity.WARNING,
        reason=RejectionReason.MISSING_BID,
        message="Missing bid",
        location=("quotes", "0"),
    )
    reject_standard = CleaningResult(
        accepted=(adjusted,),
        rejected=(RejectedQuote(standard, (diagnostic,)),),
    )
    reject_adjusted = CleaningResult(
        accepted=(standard,),
        rejected=(RejectedQuote(adjusted, (diagnostic,)),),
    )

    first = write_market_data_dataset(
        tmp_path / "standard_rejected",
        snapshot,
        source="test_fixture",
        cleaning_result=reject_standard,
        ingestion_timestamp=INGESTED_AT,
    )
    second = write_market_data_dataset(
        tmp_path / "adjusted_rejected",
        snapshot,
        source="test_fixture",
        cleaning_result=reject_adjusted,
        ingestion_timestamp=INGESTED_AT,
    )

    assert first.manifest.accepted_quote_count == second.manifest.accepted_quote_count
    assert first.manifest.rejected_quote_count == second.manifest.rejected_quote_count
    assert first.manifest.output_hash != second.manifest.output_hash
    assert first.manifest.dataset_id != second.manifest.dataset_id


def test_dataset_manifest_validates_count_invariants() -> None:
    manifest_args = {
        "schema_version": "1.0",
        "source": "fixture",
        "as_of": AS_OF,
        "ingestion_timestamp": INGESTED_AT,
        "valuation_timestamp": None,
        "day_count": None,
        "source_information": (),
        "normalisation_config": (),
        "cleaning_config": (),
        "rate_dividend_assumptions": (),
        "input_quote_count": 2,
        "canonical_quote_count": 2,
        "accepted_quote_count": 1,
        "rejected_quote_count": 1,
        "validation_issue_count": 0,
        "cleaning_diagnostic_count": 0,
        "arbitrage_diagnostic_count": 0,
        "input_hash": "0" * 64,
        "output_hash": "1" * 64,
        "dataset_id": "2" * 64,
    }

    DatasetManifest(**manifest_args)

    with pytest.raises(ValueError, match="canonical_quote_count"):
        DatasetManifest(**{**manifest_args, "canonical_quote_count": 1})

    with pytest.raises(ValueError, match="accepted_quote_count plus rejected_quote_count"):
        DatasetManifest(**{**manifest_args, "accepted_quote_count": 0})


def test_write_market_data_dataset_validates_inputs(tmp_path) -> None:
    with pytest.raises(ValueError, match="source"):
        write_market_data_dataset(tmp_path / "dataset", _snapshot(), source="")

    with pytest.raises(ValueError, match="raw_source_path"):
        write_market_data_dataset(
            tmp_path / "dataset",
            _snapshot(),
            source="fixture",
            raw_source_path=tmp_path / "missing.csv",
        )

    with pytest.raises(ValueError, match="ingestion_timestamp"):
        write_market_data_dataset(
            tmp_path / "dataset",
            _snapshot(),
            source="fixture",
            ingestion_timestamp=datetime(2026, 7, 30, 14, 31),
        )

    with pytest.raises(ValueError, match="configuration floats must be finite"):
        write_market_data_dataset(
            tmp_path / "dataset",
            _snapshot(),
            source="fixture",
            cleaning_config={"max_relative_spread": float("nan")},
        )

    with pytest.raises(ValueError, match="configuration keys"):
        write_market_data_dataset(
            tmp_path / "dataset",
            _snapshot(),
            source="fixture",
            cleaning_config={1: "value"},
        )

    with pytest.raises(ValueError, match="configuration keys"):
        write_market_data_dataset(
            tmp_path / "dataset",
            _snapshot(),
            source="fixture",
            cleaning_config={"nested": {1: "value"}},
        )
