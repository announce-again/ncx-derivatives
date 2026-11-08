from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from time import perf_counter

from ncx_derivatives.market_data import (
    CarryAssumptions,
    CleaningConfig,
    CsvIngestionConfig,
    CsvIngestionResult,
    DayCount,
    DayCountConvention,
    EnrichedCleaningResult,
    EnrichedOptionQuote,
    clean_enriched_option_quotes,
    enrich_option_chain_snapshot,
    ingest_option_chain_csv,
)
from ncx_derivatives.utils._files import csv_cell, file_sha256

from .chains import (
    IMPLIED_VOLATILITY_CHAIN_COLUMNS,
    ImpliedVolatilityChain,
    _iter_chain_records,
    build_implied_volatility_chain,
)


@dataclass(frozen=True, slots=True)
class VolatilityPipelineCounts:
    input_row_count: int
    ingestion_success_count: int
    ingestion_failed_row_count: int
    ingestion_error_count: int
    snapshot_count: int
    enriched_quote_count: int
    cleaning_accepted_count: int
    cleaning_rejected_count: int
    cleaning_diagnostic_count: int
    iv_quote_count: int
    iv_result_count: int
    iv_success_count: int
    iv_failure_count: int
    iv_quote_with_failure_count: int


@dataclass(frozen=True, slots=True)
class VolatilityPipelineTimings:
    ingestion_seconds: float
    enrichment_seconds: float
    cleaning_seconds: float
    implied_volatility_seconds: float
    total_seconds: float


@dataclass(frozen=True, slots=True)
class VolatilityPipelineResult:
    ingestion: CsvIngestionResult
    enriched_quotes: tuple[EnrichedOptionQuote, ...]
    cleaning: EnrichedCleaningResult
    implied_volatility_chain: ImpliedVolatilityChain
    timings: VolatilityPipelineTimings

    @property
    def counts(self) -> VolatilityPipelineCounts:
        summary = self.implied_volatility_chain.summary
        quotes_with_failure = sum(
            any(
                not result.is_success
                for result in (quote.bid, quote.midpoint, quote.ask)
            )
            for quote in self.implied_volatility_chain.quotes
        )
        return VolatilityPipelineCounts(
            input_row_count=len(self.ingestion.raw_records),
            ingestion_success_count=self.ingestion.successful_row_count,
            ingestion_failed_row_count=self.ingestion.failed_row_count,
            ingestion_error_count=len(self.ingestion.errors),
            snapshot_count=len(self.ingestion.snapshots),
            enriched_quote_count=len(self.enriched_quotes),
            cleaning_accepted_count=self.cleaning.accepted_count,
            cleaning_rejected_count=self.cleaning.rejected_count,
            cleaning_diagnostic_count=len(self.cleaning.diagnostics),
            iv_quote_count=summary.quote_count,
            iv_result_count=summary.result_count,
            iv_success_count=summary.success_count,
            iv_failure_count=summary.failure_count,
            iv_quote_with_failure_count=quotes_with_failure,
        )

    @property
    def input_rows_per_second(self) -> float:
        if self.timings.total_seconds == 0.0:
            return 0.0
        return len(self.ingestion.raw_records) / self.timings.total_seconds


@dataclass(frozen=True, slots=True)
class VolatilityChainCsvExport:
    path: Path
    row_count: int
    byte_count: int
    sha256: str


def run_csv_volatility_pipeline(
    path: str | Path,
    *,
    ingestion_config: CsvIngestionConfig,
    carry: CarryAssumptions,
    valuation_date: date,
    cleaning_config: CleaningConfig | None = None,
    day_count: DayCountConvention | DayCount = DayCountConvention.ACT_365F,
    low_vega_threshold: float = 1e-8,
) -> VolatilityPipelineResult:
    """Run CSV ingestion through the Stage 3.1 IV-chain boundary."""

    started = perf_counter()

    ingestion_started = perf_counter()
    ingestion = ingest_option_chain_csv(path, ingestion_config)
    ingestion_finished = perf_counter()

    enrichment_started = perf_counter()
    enriched_quotes = tuple(
        enriched_quote
        for snapshot in ingestion.snapshots
        for enriched_quote in enrich_option_chain_snapshot(
            snapshot=snapshot,
            carry=carry,
            valuation_date=valuation_date,
            day_count=day_count,
        )
    )
    enrichment_finished = perf_counter()

    cleaning_started = perf_counter()
    cleaning = clean_enriched_option_quotes(enriched_quotes, cleaning_config)
    cleaning_finished = perf_counter()

    implied_volatility_started = perf_counter()
    chain = build_implied_volatility_chain(
        cleaning.accepted,
        low_vega_threshold=low_vega_threshold,
    )
    implied_volatility_finished = perf_counter()

    return VolatilityPipelineResult(
        ingestion=ingestion,
        enriched_quotes=enriched_quotes,
        cleaning=cleaning,
        implied_volatility_chain=chain,
        timings=VolatilityPipelineTimings(
            ingestion_seconds=ingestion_finished - ingestion_started,
            enrichment_seconds=enrichment_finished - enrichment_started,
            cleaning_seconds=cleaning_finished - cleaning_started,
            implied_volatility_seconds=(
                implied_volatility_finished - implied_volatility_started
            ),
            total_seconds=implied_volatility_finished - started,
        ),
    )


def write_implied_volatility_chain_csv(
    path: str | Path,
    chain: ImpliedVolatilityChain,
) -> VolatilityChainCsvExport:
    """Write deterministic Stage 3.1 records to UTF-8 CSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = _iter_chain_records(chain)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=IMPLIED_VOLATILITY_CHAIN_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            {
                field: csv_cell(record[field])
                for field in IMPLIED_VOLATILITY_CHAIN_COLUMNS
            }
            for record in records
        )

    return VolatilityChainCsvExport(
        path=output_path,
        row_count=len(chain.quotes),
        byte_count=output_path.stat().st_size,
        sha256=file_sha256(output_path),
    )
