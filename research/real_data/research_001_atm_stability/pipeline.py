from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from ncx_derivatives.market_data import CarryAssumptions, CleaningConfig
from ncx_derivatives.volatility import (
    ImpliedVolatilityChain,
    run_csv_volatility_pipeline,
)
from research.real_data.common._optional import require_pandas
from research.real_data.common.deterministic_io import write_csv
from research.real_data.common.optionsdx_adapter import (
    STANDARD_COLUMNS,
    optionsdx_stage2_csv_config,
)


@dataclass(frozen=True, slots=True)
class DailyBatchResult:
    chain: ImpliedVolatilityChain
    attrition_records: tuple[dict[str, object], ...]
    stage2_input_paths: tuple[Path, ...]


def run_daily_ncx_pipeline(
    standardized_options,
    *,
    dataset_id: str,
    carry_for_date: Callable[[date], CarryAssumptions],
    interim_directory: str | Path,
    cleaning_config: CleaningConfig | None = None,
) -> DailyBatchResult:
    """Run Stage 2/3.1 date by date so each maturity uses the right valuation date."""

    pandas = require_pandas()
    data = standardized_options.copy()
    data["quote_date"] = pandas.to_datetime(data["quote_date"], errors="coerce")
    if data["quote_date"].isna().any():
        raise ValueError("quote_date must be valid before running the NCX pipeline")
    root = Path(interim_directory)
    quotes = []
    attrition = []
    paths = []
    config = optionsdx_stage2_csv_config(dataset_id=dataset_id)
    cleaning = cleaning_config or CleaningConfig(
        reject_missing_bid=True,
        reject_missing_ask=True,
        reject_crossed_market=True,
        reject_zero_midpoint=True,
    )
    for timestamp, daily in data.groupby("quote_date", sort=True):
        valuation_date = timestamp.date()
        source_path = root / f"stage2_input_{valuation_date.isoformat()}.csv"
        stage2_daily = daily.loc[:, STANDARD_COLUMNS].copy()
        stage2_daily["quote_date"] = stage2_daily["quote_date"].dt.strftime("%Y-%m-%d")
        stage2_daily["expiration"] = pandas.to_datetime(
            stage2_daily["expiration"]
        ).dt.strftime("%Y-%m-%d")
        records = stage2_daily.to_dict("records")
        write_csv(
            source_path,
            records,
            columns=STANDARD_COLUMNS,
            sort_by=("underlying_symbol", "expiration", "strike", "option_type", "source_file", "source_row"),
        )
        paths.append(source_path)
        result = run_csv_volatility_pipeline(
            source_path,
            ingestion_config=config,
            carry=carry_for_date(valuation_date),
            valuation_date=valuation_date,
            cleaning_config=cleaning,
        )
        quotes.extend(result.implied_volatility_chain.quotes)
        counts = result.counts
        attrition.extend(
            (
                _count_record("stage2_ingestion", "row_parse_or_schema_error", valuation_date, counts.input_row_count, counts.ingestion_success_count),
                _count_record("cleaning", "cleaning_policy_rejection", valuation_date, counts.ingestion_success_count, counts.cleaning_accepted_count),
                _count_record("iv_inversion", "one_or_more_iv_results_failed", valuation_date, counts.cleaning_accepted_count, counts.iv_quote_count - counts.iv_quote_with_failure_count),
            )
        )
    return DailyBatchResult(
        chain=ImpliedVolatilityChain(tuple(quotes)),
        attrition_records=tuple(attrition),
        stage2_input_paths=tuple(paths),
    )


def _count_record(stage, reason, valuation_date, raw_count, remaining_count):
    return {
        "stage": stage,
        "reason": reason,
        "underlying": None,
        "quote_date": valuation_date,
        "expiration": None,
        "raw_count": raw_count,
        "remaining_count": remaining_count,
    }
