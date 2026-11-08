from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from research.real_data.common.deterministic_io import (
    file_sha256,
    logical_frame_sha256,
    write_csv,
    write_json,
    write_output_hashes,
    write_parquet,
)
from research.real_data.common.optionsdx_adapter import standardize_optionsdx_wide
from research.real_data.common.returns import build_underlying_return_panel

from .analyse_spy import (
    add_market_period,
    add_volatility_regime,
    freeze_regime_thresholds,
    stability_summary,
)
from .build_expiry_panel import build_expiry_panel
from .build_tenor_panel import (
    add_daily_stability,
    adjacent_expiry_gaps,
    build_nearest_tenor_panel,
)
from .config import Research001Config
from .historical_carry import (
    SUPPORTED_CARRY_SPECIFICATIONS,
    build_carry_assumptions,
    rate_row_map,
)
from .pipeline import run_daily_ncx_pipeline
from .plots import generate_core_figures
from .regressions import run_quote_uncertainty_regression, run_spy_regression

TARGET_TENORS = (21, 45, 90, 150)
TENOR_TOLERANCES = (7, 10, 15, 25)
DATASET_ID = "dudesurfin/spy-options-eod-volatility-surface-2010-2023:v2"
_ELIGIBLE_QUOTES = """
    dte BETWEEN 7 AND 180
    AND strike > 0
    AND underlying_last > 0
    AND abs(ln(strike / underlying_last)) <= ?
"""


@dataclass(frozen=True)
class _CheckpointPaths:
    stage2: Path
    expiry: Path
    attrition: Path
    exclusions: Path

    @classmethod
    def create(cls, root: Path, specification: str) -> _CheckpointPaths:
        paths = cls(
            stage2=root / "stage2_inputs" / specification,
            expiry=root / "expiry_partitions" / specification,
            attrition=root / "attrition_partitions" / specification,
            exclusions=root / "daily_exclusions" / specification,
        )
        for directory in (
            paths.stage2,
            paths.expiry,
            paths.attrition,
            paths.exclusions,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return paths


@dataclass(frozen=True)
class _DailyWorker:
    connection: duckdb.DuckDBPyConnection
    rates_by_date: dict[date, dict[str, object]]
    distributions: pd.DataFrame
    specification: str
    paths: _CheckpointPaths
    max_moneyness: float


_worker: _DailyWorker | None = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full-history SPY NCX reconstruction"
    )
    parser.add_argument("--options-database", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--rate-panel", type=Path, required=True)
    parser.add_argument("--distribution-panel", type=Path, required=True)
    parser.add_argument("--carry-panel", type=Path, required=True)
    parser.add_argument("--option-implied-panel", type=Path, required=True)
    parser.add_argument("--audit-source-dir", type=Path, required=True)
    parser.add_argument("--interim-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--specification", choices=SUPPORTED_CARRY_SPECIFICATIONS, required=True
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-abs-log-moneyness", type=float, default=0.40)
    parser.add_argument("--maximum-dates", type=int)
    args = parser.parse_args()
    run_full_ncx(
        options_database=args.options_database,
        dataset_manifest=args.dataset_manifest,
        rate_panel_path=args.rate_panel,
        distribution_panel_path=args.distribution_panel,
        carry_panel_path=args.carry_panel,
        option_implied_panel_path=args.option_implied_panel,
        audit_source_dir=args.audit_source_dir,
        interim_dir=args.interim_dir,
        output_dir=args.output_dir,
        specification=args.specification,
        workers=args.workers,
        max_abs_log_moneyness=args.max_abs_log_moneyness,
        maximum_dates=args.maximum_dates,
    )


def run_full_ncx(
    *,
    options_database: Path,
    dataset_manifest: Path,
    rate_panel_path: Path,
    distribution_panel_path: Path,
    carry_panel_path: Path,
    option_implied_panel_path: Path,
    audit_source_dir: Path,
    interim_dir: Path,
    output_dir: Path,
    specification: str,
    workers: int,
    max_abs_log_moneyness: float,
    maximum_dates: int | None = None,
) -> None:
    if workers < 1:
        raise ValueError("workers must be positive")
    paths = _CheckpointPaths.create(interim_dir, specification)
    output_dir.mkdir(parents=True, exist_ok=True)

    rates = pd.read_parquet(rate_panel_path)
    distributions = pd.read_parquet(distribution_panel_path)
    dates, raw_count, scoped_count = _load_run_scope(
        options_database,
        rates,
        specification,
        max_abs_log_moneyness,
        maximum_dates,
    )
    if not dates:
        raise ValueError("no quote dates match the requested run")

    config = Research001Config(
        dataset_ids=(DATASET_ID,),
        start_date=min(dates),
        end_date=max(dates),
        underlyings=("SPY",),
        target_tenors=TARGET_TENORS,
        tenor_tolerances=TENOR_TOLERANCES,
    )
    config.write(output_dir / "config.json")
    started = time.perf_counter()
    _run_daily_batches(
        dates,
        database=options_database,
        rates=rates,
        distributions=distributions,
        specification=specification,
        paths=paths,
        max_moneyness=max_abs_log_moneyness,
        workers=workers,
        output_dir=output_dir,
        started=started,
    )
    successful_dates = [
        value
        for value in dates
        if (paths.expiry / f"expiry_{value.isoformat()}.parquet").exists()
    ]
    excluded_dates = [
        value
        for value in dates
        if (paths.exclusions / f"exclusion_{value.isoformat()}.json").exists()
    ]
    unresolved_dates = sorted(set(dates) - set(successful_dates) - set(excluded_dates))
    if unresolved_dates:
        raise RuntimeError(
            f"{len(unresolved_dates)} dates have neither output nor exclusion marker"
        )

    if not successful_dates:
        raise RuntimeError("no daily batches produced expiry metrics")

    expiry = _read_partitions(paths.expiry, successful_dates)
    expiry = adjacent_expiry_gaps(
        add_daily_stability(expiry, group_columns=("underlying", "expiration"))
    )
    tenor = build_nearest_tenor_panel(
        expiry,
        target_tenors=TARGET_TENORS,
        tenor_tolerances=TENOR_TOLERANCES,
        minimum_smile_points=config.minimum_smile_points,
    )
    tenor, returns, thresholds = _enrich_tenor_panel(
        tenor,
        options_database,
        carry_panel_path,
        option_implied_panel_path,
        specification,
    )

    expiry_path = write_parquet(
        output_dir / "atm_expiry_panel.parquet",
        expiry,
        sort_by=("underlying", "quote_date", "expiration"),
    )
    tenor_path = write_parquet(
        output_dir / "atm_tenor_panel.parquet",
        tenor,
        sort_by=("underlying", "target_tenor", "quote_date"),
    )
    write_parquet(
        output_dir / "underlying_return_panel.parquet",
        returns,
        sort_by=("underlying", "quote_date"),
    )
    _write_attrition(paths.attrition, dates, raw_count, scoped_count, output_dir)
    _write_daily_exclusions(paths.exclusions, excluded_dates, output_dir)
    _write_results(tenor, expiry, output_dir)
    generate_core_figures(tenor, expiry, output_dir / "figures")
    _copy_audit(audit_source_dir, output_dir)
    write_json(output_dir / "regime_thresholds.json", thresholds)
    persisted_expiry = pd.read_parquet(expiry_path)
    persisted_tenor = pd.read_parquet(tenor_path)
    write_json(
        output_dir / "run_manifest.json",
        {
            "research_id": "research_001a_spy_full_history_ncx",
            "iv_specification": specification,
            "formal_ncx_reconstruction": True,
            "date_range": [str(min(dates)), str(max(dates))],
            "requested_quote_dates": len(dates),
            "processed_quote_dates": len(successful_dates),
            "excluded_quote_dates": len(excluded_dates),
            "worker_count": workers,
            "max_abs_log_moneyness": max_abs_log_moneyness,
            "config_sha256": config.sha256,
            "dataset_manifest_sha256": file_sha256(dataset_manifest),
            "carry_inputs": {
                "rate_panel_sha256": file_sha256(rate_panel_path),
                "distribution_panel_sha256": file_sha256(distribution_panel_path),
                "carry_panel_sha256": file_sha256(carry_panel_path),
                "option_implied_panel_sha256": file_sha256(option_implied_panel_path),
            },
            "counts": {
                "raw_wide_rows": raw_count,
                "scoped_wide_rows": scoped_count,
                "expiry_panel_rows": len(expiry),
                "tenor_panel_rows": len(tenor),
            },
            "logical_panel_hashes": {
                "expiry": logical_frame_sha256(
                    persisted_expiry, sort_by=("underlying", "quote_date", "expiration")
                ),
                "tenor": logical_frame_sha256(
                    persisted_tenor,
                    sort_by=("underlying", "target_tenor", "quote_date"),
                ),
            },
            "elapsed_seconds": time.perf_counter() - started,
            "regime_thresholds": thresholds,
        },
    )
    _write_report(output_dir, specification)
    write_output_hashes(output_dir, output_dir / "output_hashes.json")


def _load_run_scope(
    options_database: Path,
    rates: pd.DataFrame,
    specification: str,
    max_abs_log_moneyness: float,
    maximum_dates: int | None,
) -> tuple[list[date], int, int]:
    with duckdb.connect(str(options_database), read_only=True) as connection:
        dates = [
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT quote_date FROM raw_wide ORDER BY quote_date"
            ).fetchall()
        ]
        if specification == "sofr_flat_projected_dividend_schedule":
            available = set(
                pd.to_datetime(rates.loc[rates["SOFR"].notna(), "date"]).dt.date
            )
            dates = [value for value in dates if value in available]
        if maximum_dates is not None:
            dates = dates[:maximum_dates]
        raw_count = connection.execute("SELECT count(*) FROM raw_wide").fetchone()[0]
        scoped_count = connection.execute(
            f"SELECT count(*) FROM raw_wide WHERE {_ELIGIBLE_QUOTES}",
            [max_abs_log_moneyness],
        ).fetchone()[0]
    return dates, raw_count, scoped_count


def _run_daily_batches(
    dates: list[date],
    *,
    database: Path,
    rates: pd.DataFrame,
    distributions: pd.DataFrame,
    specification: str,
    paths: _CheckpointPaths,
    max_moneyness: float,
    workers: int,
    output_dir: Path,
    started: float,
) -> None:
    failures = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
        initargs=(database, rates, distributions, specification, paths, max_moneyness),
    ) as executor:
        futures = {
            executor.submit(_process_date, value.isoformat()): value for value in dates
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            quote_date = futures[future]
            try:
                future.result()
            except Exception as error:  # the failure is persisted and raised after checkpoints complete
                failures.append(
                    {
                        "quote_date": quote_date,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
            if completed % 100 == 0 or completed == len(dates):
                elapsed = time.perf_counter() - started
                print(
                    f"{specification}: {completed}/{len(dates)} dates, {elapsed:.1f}s",
                    flush=True,
                )
    if failures:
        write_csv(output_dir / "batch_failures.csv", failures, sort_by=("quote_date",))
        raise RuntimeError(
            f"{len(failures)} daily NCX batches failed; checkpoints were retained"
        )

    write_csv(
        output_dir / "batch_failures.csv",
        (),
        columns=("quote_date", "error_type", "error"),
        sort_by=("quote_date",),
    )


def _enrich_tenor_panel(
    tenor: pd.DataFrame,
    options_database: Path,
    carry_panel_path: Path,
    option_implied_panel_path: Path,
    specification: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    tenor["iv_specification"] = specification
    carry = pd.read_parquet(carry_panel_path)
    carry["quote_date"] = pd.to_datetime(carry["quote_date"])
    carry["expiration"] = pd.to_datetime(carry["expiration"])
    diagnostics = pd.read_parquet(option_implied_panel_path)
    diagnostics["quote_date"] = pd.to_datetime(diagnostics["quote_date"])
    diagnostics["expiration"] = pd.to_datetime(diagnostics["expiration"])
    carry_columns = [
        "quote_date",
        "expiration",
        "trailing_12m_dividend_yield",
        "projected_dividend_count_to_expiry",
        "projected_dividend_cash_to_expiry",
        "dgs3mo",
        "sofr",
    ]
    tenor = tenor.merge(
        carry[carry_columns],
        on=["quote_date", "expiration"],
        how="left",
        validate="one_to_one",
    )
    tenor = tenor.merge(
        diagnostics[
            [
                "quote_date",
                "expiration",
                "option_implied_forward",
                "parity_pair_count",
                "option_implied_minus_baseline_forward",
            ]
        ],
        on=["quote_date", "expiration"],
        how="left",
        validate="one_to_one",
    )

    with duckdb.connect(str(options_database), read_only=True) as connection:
        spot = connection.execute(
            """SELECT 'SPY' AS underlying_symbol, quote_date, median(underlying_last) AS underlying_price
            FROM raw_wide GROUP BY quote_date ORDER BY quote_date"""
        ).fetchdf()
    returns = build_underlying_return_panel(spot)
    returns["past_realized_volatility"] = returns.groupby("underlying")[
        "underlying_return"
    ].transform(lambda values: values.rolling(21, min_periods=10).std() * (252.0**0.5))
    tenor = tenor.merge(
        returns[
            [
                "underlying",
                "quote_date",
                "underlying_return",
                "absolute_underlying_return",
                "past_realized_volatility",
            ]
        ],
        on=["underlying", "quote_date"],
        how="left",
        validate="many_to_one",
    )
    tenor = add_market_period(tenor)
    thresholds = freeze_regime_thresholds(tenor)
    tenor = add_volatility_regime(tenor, thresholds)
    tenor["high_vol_regime"] = (
        tenor["market_regime"].isin(["high", "extreme"]).astype(int)
    )
    tenor["liquidity_quintile"] = tenor.groupby("target_tenor")[
        "median_relative_price_spread"
    ].transform(_quintile)
    return tenor, returns, thresholds


def _initialize_worker(
    database: Path,
    rates: pd.DataFrame,
    distributions: pd.DataFrame,
    specification: str,
    paths: _CheckpointPaths,
    max_moneyness: float,
) -> None:
    global _worker
    _worker = _DailyWorker(
        connection=duckdb.connect(str(database), read_only=True),
        rates_by_date=rate_row_map(rates),
        distributions=distributions,
        specification=specification,
        paths=paths,
        max_moneyness=max_moneyness,
    )


def _process_date(date_text: str) -> dict[str, object]:
    worker = _worker
    if worker is None:
        raise RuntimeError("daily worker was not initialized")
    output = worker.paths.expiry / f"expiry_{date_text}.parquet"
    attrition_output = worker.paths.attrition / f"attrition_{date_text}.parquet"
    exclusion_output = worker.paths.exclusions / f"exclusion_{date_text}.json"
    if (output.exists() or exclusion_output.exists()) and attrition_output.exists():
        return {"quote_date": date_text, "status": "checkpoint"}
    wide = worker.connection.execute(
        f"""
        SELECT * FROM raw_wide
        WHERE quote_date = ? AND {_ELIGIBLE_QUOTES}
        ORDER BY expire_date, strike
        """,
        [date_text, worker.max_moneyness],
    ).fetchdf()
    if wide.empty:
        raise ValueError(f"no eligible rows for {date_text}")
    quote_date = date.fromisoformat(date_text)
    two_sided = wide[["c_bid", "c_ask"]].notna().all(axis=1) | wide[
        ["p_bid", "p_ask"]
    ].notna().all(axis=1)
    if not bool(two_sided.any()):
        raw_contract_count = len(wide) * 2
        attrition = pd.DataFrame.from_records(
            [
                {
                    "stage": "daily_eligibility",
                    "reason": "NO_TWO_SIDED_QUOTES",
                    "underlying": "SPY",
                    "quote_date": quote_date,
                    "expiration": None,
                    "raw_count": raw_contract_count,
                    "remaining_count": 0,
                }
            ]
        )
        write_parquet(
            attrition_output,
            attrition,
            sort_by=("stage", "reason", "quote_date", "expiration"),
        )
        write_json(
            exclusion_output,
            {
                "quote_date": date_text,
                "reason": "NO_TWO_SIDED_QUOTES",
                "raw_wide_rows": len(wide),
                "remaining_rows": 0,
            },
        )
        return {
            "quote_date": date_text,
            "status": "excluded",
            "reason": "NO_TWO_SIDED_QUOTES",
        }

    spot = float(wide["underlying_last"].median())
    carry = build_carry_assumptions(
        specification=worker.specification,
        quote_date=quote_date,
        spot=spot,
        rate_row=worker.rates_by_date[quote_date],
        distributions=worker.distributions,
    )
    standardized = standardize_optionsdx_wide(
        wide,
        underlying_symbol="SPY",
        source_file=f"spy_options_{quote_date.year}.parquet",
    )
    batch = run_daily_ncx_pipeline(
        standardized,
        dataset_id=DATASET_ID,
        carry_for_date=lambda _date: carry,
        interim_directory=worker.paths.stage2,
    )
    expiry, attrition = build_expiry_panel(
        batch.chain, dataset_id=DATASET_ID, minimum_smile_points=5
    )
    if expiry.empty:
        raise ValueError(f"no successful expiry metrics for {date_text}")
    expiry["carry_specification"] = worker.specification
    batch_attrition = pd.DataFrame.from_records(batch.attrition_records)
    combined = pd.concat([batch_attrition, attrition], ignore_index=True)
    write_parquet(
        attrition_output,
        combined,
        sort_by=("stage", "reason", "quote_date", "expiration"),
    )
    write_parquet(output, expiry, sort_by=("underlying", "quote_date", "expiration"))
    return {"quote_date": date_text, "status": "processed", "expiry_rows": len(expiry)}


def _read_partitions(directory: Path, dates: list[date]) -> pd.DataFrame:
    frames = [
        pd.read_parquet(directory / f"expiry_{value.isoformat()}.parquet")
        for value in dates
    ]
    return pd.concat(frames, ignore_index=True).sort_values(
        ["underlying", "quote_date", "expiration"], kind="mergesort", ignore_index=True
    )


def _write_attrition(
    attrition_dir: Path,
    dates: list[date],
    raw_count: int,
    scoped_count: int,
    output_dir: Path,
) -> None:
    frames = [
        pd.read_parquet(attrition_dir / f"attrition_{value.isoformat()}.parquet")
        for value in dates
    ]
    attrition = pd.concat(frames, ignore_index=True)
    attrition = pd.concat(
        [
            pd.DataFrame.from_records(
                [
                    {
                        "stage": "full_history_scope",
                        "reason": "dte_7_180_and_abs_log_moneyness_le_0.40",
                        "underlying": "SPY",
                        "quote_date": None,
                        "expiration": None,
                        "raw_count": raw_count * 2,
                        "remaining_count": scoped_count * 2,
                    }
                ]
            ),
            attrition,
        ],
        ignore_index=True,
    )
    write_csv(
        output_dir / "sample_attrition.csv",
        attrition.to_dict("records"),
        sort_by=("stage", "reason", "quote_date", "expiration"),
    )


def _write_daily_exclusions(
    exclusion_dir: Path, dates: list[date], output_dir: Path
) -> None:
    records = [
        json.loads(
            (exclusion_dir / f"exclusion_{value.isoformat()}.json").read_text(
                encoding="utf-8"
            )
        )
        for value in dates
    ]
    write_csv(
        output_dir / "daily_exclusions.csv",
        records,
        columns=("quote_date", "reason", "raw_wide_rows", "remaining_rows"),
        sort_by=("quote_date",),
    )


def _write_results(tenor: pd.DataFrame, expiry: pd.DataFrame, output_dir: Path) -> None:
    year = tenor.assign(year=pd.to_datetime(tenor["quote_date"]).dt.year)
    summaries = {
        "stability_by_tenor.csv": stability_summary(
            tenor, group_columns=("target_tenor",)
        ),
        "stability_by_year.csv": stability_summary(
            year,
            group_columns=("year", "target_tenor"),
            coverage_scope_columns=("year",),
        ),
        "stability_by_period.csv": stability_summary(
            tenor, group_columns=("market_period", "target_tenor")
        ),
        "stability_by_regime.csv": stability_summary(
            tenor, group_columns=("market_regime", "target_tenor")
        ),
        "stability_by_liquidity.csv": stability_summary(
            tenor, group_columns=("liquidity_quintile", "target_tenor")
        ),
    }
    for name, frame in summaries.items():
        write_csv(
            output_dir / name,
            frame.to_dict("records"),
            sort_by=tuple(frame.columns[: len(frame.columns) - 10]),
        )
    regressions = []
    for name, runner in (
        ("spy_contemporaneous_hac", run_spy_regression),
        ("quote_uncertainty_next_day_hac", run_quote_uncertainty_regression),
    ):
        try:
            _, records = runner(tenor)
            regressions.extend({**record, "model": name} for record in records)
        except (ValueError, KeyError) as error:
            regressions.append(
                {"model": name, "term": "MODEL_FAILURE", "error": str(error)}
            )
    write_csv(
        output_dir / "regression_results.csv", regressions, sort_by=("model", "term")
    )
    quote_rows = []
    robustness = []
    for target, group in tenor.groupby("target_tenor"):
        valid = group["noise_adjusted_move"].dropna()
        quote_rows.append(
            {
                "target_tenor": int(target),
                "observation_count": int(valid.size),
                "median_atm_iv_spread": group["atm_iv_spread"].median(),
                "median_absolute_atm_iv_change": group[
                    "absolute_atm_iv_change"
                ].median(),
                "median_noise_adjusted_move": valid.median(),
                "fraction_move_within_average_iv_spread": (valid <= 1.0).mean(),
                "spearman_spread_vs_next_move": group["atm_iv_spread"].corr(
                    group["absolute_atm_iv_change"].shift(-1), method="spearman"
                ),
            }
        )
        robustness.append(
            {
                "target_tenor": int(target),
                "median_absolute_atm_iv_change": group[
                    "absolute_atm_iv_change"
                ].median(),
                "median_absolute_relative_atm_iv_change": group[
                    "relative_atm_iv_change"
                ]
                .abs()
                .median(),
                "median_absolute_total_variance_change": group[
                    "delta_atm_total_variance"
                ]
                .abs()
                .median(),
                "median_absolute_relative_total_variance_change": (
                    group["delta_atm_total_variance"].abs()
                    / group["previous_atm_total_variance"]
                ).median(),
            }
        )
    write_csv(
        output_dir / "quote_uncertainty_results.csv",
        quote_rows,
        sort_by=("target_tenor",),
    )
    write_csv(
        output_dir / "robustness_results.csv", robustness, sort_by=("target_tenor",)
    )


def _copy_audit(source: Path, destination: Path) -> None:
    for name in (
        "audit_summary.csv",
        "audit_by_date.csv",
        "audit_by_expiry.csv",
        "audit_failures.csv",
        "schema_report.json",
        "data_quality_report.md",
    ):
        path = source / name
        if path.exists():
            shutil.copy2(path, destination / name)


def _write_report(output_dir: Path, specification: str) -> None:
    summary = pd.read_csv(output_dir / "stability_by_tenor.csv").set_index(
        "target_tenor"
    )
    regime = pd.read_csv(output_dir / "stability_by_regime.csv")
    target = regime.loc[regime["target_tenor"].eq(21)].set_index("market_regime")
    ratio = (
        target.loc["extreme", "median_absolute_atm_iv_change"]
        / target.loc["low", "median_absolute_atm_iv_change"]
        if {"extreme", "low"}.issubset(target.index)
        else float("nan")
    )
    ranking = (
        summary.sort_values("median_absolute_atm_iv_change").index.astype(int).tolist()
    )
    tenor_values = {
        target: (
            f"{100 * summary.loc[target, 'median_absolute_atm_iv_change']:.3f}"
            if target in summary.index
            else "NA"
        )
        for target in TARGET_TENORS
    }
    (output_dir / "research_001_report.md").write_text(
        "# Research 001A — full-history NCX reconstruction\n\n"
        f"Carry specification: `{specification}`.\n\n"
        "Raw bid/ask quotes were inverted with the NCX Stage 3.1 pipeline and passed to Stage 3.2 ATM analysis for every eligible 7–180D expiry.\n\n"
        "## Findings\n\n"
        f"- Median daily |ΔATMIV| (vol points): 21D={tenor_values[21]}, "
        f"45D={tenor_values[45]}, 90D={tenor_values[90]}, 150D={tenor_values[150]}.\n"
        f"- Stability ranking, most to least stable: {ranking}.\n"
        f"- Extreme-regime 21D instability is {ratio:.2f}× the low-regime median.\n\n"
        "Treasury constant-maturity yields remain financing proxies, and the projected SPY distribution schedule uses the last amount known at each quote date.\n",
        encoding="utf-8",
        newline="\n",
    )


def _quintile(values: pd.Series) -> pd.Series:
    ranked = values.rank(method="first")
    try:
        return pd.qcut(ranked, 5, labels=[1, 2, 3, 4, 5])
    except ValueError:
        return pd.Series(pd.NA, index=values.index, dtype="Int64")


if __name__ == "__main__":
    main()
