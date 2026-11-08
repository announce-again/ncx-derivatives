"""Run the full Stage 2 market-data pipeline on a fixed sample."""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    VALUATION_DATE,
    carry_assumptions,
    cleaning_config,
    ingest_sample_csv,
    output_root,
    sample_csv_path,
)

from ncx_derivatives.market_data import (  # noqa: E402
    clean_enriched_option_quotes,
    diagnose_static_arbitrage,
    enrich_option_chain_snapshot,
    enriched_quotes_to_dataframe,
    validate_option_chain_snapshot,
    write_market_data_dataset,
)

INGESTED_AT = datetime(2026, 7, 30, 14, 31, tzinfo=timezone.utc)


def main() -> None:
    ingestion = ingest_sample_csv()
    snapshot = ingestion.snapshots[0]
    validation = validate_option_chain_snapshot(snapshot)
    enriched = enrich_option_chain_snapshot(
        snapshot=snapshot,
        carry=carry_assumptions(),
        valuation_date=VALUATION_DATE,
    )
    cleaning = clean_enriched_option_quotes(enriched, cleaning_config())
    arbitrage = diagnose_static_arbitrage(tuple(cleaning.accepted))
    frame = enriched_quotes_to_dataframe(cleaning.accepted)

    output = output_root() / "end_to_end_dataset"
    result = write_market_data_dataset(
        output,
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

    print(f"Input rows: {len(ingestion.raw_records)}")
    print(f"Canonical quotes: {len(snapshot.quotes)}")
    print(f"Validation issues: {len(validation.issues)}")
    print(f"Accepted quotes: {cleaning.accepted_count}")
    print(f"Rejected quotes: {cleaning.rejected_count}")
    print(f"Arbitrage diagnostics: {arbitrage.violation_count}")
    print(f"DataFrame rows: {len(frame)}")
    print(f"Dataset ID: {result.manifest.dataset_id}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
