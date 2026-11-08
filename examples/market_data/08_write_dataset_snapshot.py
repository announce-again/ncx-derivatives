"""Write a reproducible market-data dataset snapshot."""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    VALUATION_DATE,
    carry_assumptions,
    cleaning_config,
    output_root,
    sample_csv_path,
    sample_snapshot,
)

from ncx_derivatives.market_data import (  # noqa: E402
    clean_enriched_option_quotes,
    diagnose_static_arbitrage,
    enrich_option_chain_snapshot,
    validate_option_chain_snapshot,
    write_market_data_dataset,
)

INGESTED_AT = datetime(2026, 7, 30, 14, 31, tzinfo=timezone.utc)


def _write_dataset(path: Path):
    snapshot = sample_snapshot()
    validation = validate_option_chain_snapshot(snapshot)
    enriched = enrich_option_chain_snapshot(
        snapshot=snapshot,
        carry=carry_assumptions(),
        valuation_date=VALUATION_DATE,
    )
    cleaning = clean_enriched_option_quotes(enriched, cleaning_config())
    arbitrage = diagnose_static_arbitrage(tuple(cleaning.accepted))

    return write_market_data_dataset(
        path,
        snapshot,
        source="synthetic_cboe_sample",
        raw_source_path=sample_csv_path(),
        validation_report=validation,
        cleaning_result=cleaning,
        static_arbitrage_report=arbitrage,
        ingestion_timestamp=INGESTED_AT,
        valuation_timestamp=snapshot.as_of,
        day_count="ACT/365F",
        source_information={"provider": "example", "sample": "synthetic"},
        normalisation_config={"timezone": "UTC"},
        cleaning_config={"max_relative_spread": 0.80, "min_open_interest": 1},
        rate_dividend_assumptions={"risk_free_rate": 0.04, "dividend_yield": 0.01},
    )


def main() -> None:
    first = _write_dataset(output_root() / "sample_dataset")
    second = _write_dataset(output_root() / "sample_dataset_repeat")

    print("Dataset ID:", first.manifest.dataset_id)
    print("Input hash:", first.manifest.input_hash)
    print("Output hash:", first.manifest.output_hash)
    print("Accepted/rejected:", first.manifest.accepted_quote_count, "/", first.manifest.rejected_quote_count)
    print()
    print("Written paths:")
    for label, path in first.paths:
        print(f"  {label}: {path.relative_to(output_root().parents[1])}")

    assert first.manifest.dataset_id == second.manifest.dataset_id
    print()
    print("Deterministic identity confirmed across separate output directories.")


if __name__ == "__main__":
    main()
