from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from ncx_derivatives.market_data import (
    CarryAssumptions,
    FlatDividendYieldCurve,
    FlatZeroRateCurve,
)
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
from .build_tenor_panel import add_daily_stability, build_nearest_tenor_panel
from .config import Research001Config
from .pipeline import run_daily_ncx_pipeline
from .plots import generate_core_figures
from .regressions import run_quote_uncertainty_regression, run_spy_regression
from .report import write_limitations, write_research_report

TARGET_TENORS = (21, 45, 90, 150)
TENOR_TOLERANCES = (7, 10, 15, 25)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SPY 2020-2022 Research 001 pilot")
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--interim-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rate", type=float, default=0.0)
    parser.add_argument("--dividend-yield", type=float, default=0.0)
    parser.add_argument("--specification", default="zero_rate_zero_dividend_diagnostic")
    parser.add_argument("--max-abs-log-moneyness", type=float, default=0.40)
    arguments = parser.parse_args()
    outputs = run_pilot(
        raw_csv=arguments.raw_csv,
        dataset_manifest=arguments.dataset_manifest,
        database=arguments.database,
        interim_dir=arguments.interim_dir,
        output_dir=arguments.output_dir,
        rate=arguments.rate,
        dividend_yield=arguments.dividend_yield,
        specification=arguments.specification,
        max_abs_log_moneyness=arguments.max_abs_log_moneyness,
    )
    print(json.dumps(outputs, indent=2, default=str))


def run_pilot(
    *,
    raw_csv: Path,
    dataset_manifest: Path,
    database: Path,
    interim_dir: Path,
    output_dir: Path,
    rate: float,
    dividend_yield: float,
    specification: str,
    max_abs_log_moneyness: float,
) -> dict[str, object]:
    interim_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage2_dir = interim_dir / "stage2_inputs" / specification
    stage2_dir.mkdir(parents=True, exist_ok=True)
    config = Research001Config(
        dataset_ids=("kylegraupe/spy-daily-eod-options-quotes-2020-2022:v1",),
        start_date=date(2020, 1, 1),
        end_date=date(2022, 12, 31),
        underlyings=("SPY",),
        target_tenors=TARGET_TENORS,
        tenor_tolerances=TENOR_TOLERANCES,
    )
    config_path = config.write(output_dir / "config.json")

    connection = duckdb.connect(str(database))
    _prepare_database(connection, raw_csv)
    audit_paths = _write_audit(connection, output_dir)
    _prepare_selected_expiries(connection, max_abs_log_moneyness)
    selected_dates = [row[0] for row in connection.execute(
        "SELECT DISTINCT quote_date FROM pilot_wide ORDER BY quote_date"
    ).fetchall()]

    carry = CarryAssumptions(
        risk_free_curve=FlatZeroRateCurve(rate),
        dividend_curve=FlatDividendYieldCurve(dividend_yield),
    )
    expiry_frames = []
    attrition_frames = []
    batch_attrition = []
    for quote_date in selected_dates:
        wide = connection.execute(
            "SELECT * EXCLUDE (target_tenor, tenor_mismatch) FROM pilot_wide WHERE quote_date = ? ORDER BY expire_date, strike",
            [quote_date],
        ).fetchdf()
        standardized = standardize_optionsdx_wide(
            wide,
            underlying_symbol="SPY",
            source_file=raw_csv.name,
        )
        batch = run_daily_ncx_pipeline(
            standardized,
            dataset_id="kylegraupe/spy-daily-eod-options-quotes-2020-2022:v1",
            carry_for_date=lambda _date, value=carry: value,
            interim_directory=stage2_dir,
        )
        expiry, attrition = build_expiry_panel(
            batch.chain,
            dataset_id="kylegraupe/spy-daily-eod-options-quotes-2020-2022:v1",
            minimum_smile_points=config.minimum_smile_points,
        )
        if not expiry.empty:
            expiry_frames.append(expiry)
        if not attrition.empty:
            attrition_frames.append(attrition)
        batch_attrition.extend(batch.attrition_records)

    expiry_panel = pd.concat(expiry_frames, ignore_index=True).sort_values(
        ["underlying", "quote_date", "expiration"], kind="mergesort", ignore_index=True
    )
    expiry_panel = add_daily_stability(
        expiry_panel,
        group_columns=("underlying", "expiration"),
    )
    tenor_panel = build_nearest_tenor_panel(
        expiry_panel,
        target_tenors=TARGET_TENORS,
        tenor_tolerances=TENOR_TOLERANCES,
        minimum_smile_points=config.minimum_smile_points,
    )
    tenor_panel["iv_specification"] = specification

    raw_spot = connection.execute(
        "SELECT 'SPY' AS underlying_symbol, quote_date, underlying_last AS underlying_price FROM raw_wide GROUP BY quote_date, underlying_last ORDER BY quote_date"
    ).fetchdf()
    return_panel = build_underlying_return_panel(raw_spot)
    return_panel = _add_realized_volatility(return_panel)
    tenor_panel = tenor_panel.merge(
        return_panel[["underlying", "quote_date", "underlying_return", "absolute_underlying_return", "past_realized_volatility"]],
        on=["underlying", "quote_date"],
        how="left",
        validate="many_to_one",
    )
    tenor_panel = add_market_period(tenor_panel)
    thresholds = freeze_regime_thresholds(tenor_panel)
    tenor_panel = add_volatility_regime(tenor_panel, thresholds)
    tenor_panel["high_vol_regime"] = tenor_panel["market_regime"].isin(["high", "extreme"]).astype(int)
    tenor_panel["liquidity_quintile"] = tenor_panel.groupby("target_tenor")["median_relative_price_spread"].transform(_quintile)

    expiry_path = write_parquet(
        output_dir / "atm_expiry_panel.parquet",
        expiry_panel,
        sort_by=("underlying", "quote_date", "expiration"),
    )
    tenor_path = write_parquet(
        output_dir / "atm_tenor_panel.parquet",
        tenor_panel,
        sort_by=("underlying", "target_tenor", "quote_date"),
    )
    returns_path = write_parquet(
        output_dir / "underlying_return_panel.parquet",
        return_panel,
        sort_by=("underlying", "quote_date"),
    )

    attrition = pd.concat(
        [pd.DataFrame.from_records(batch_attrition), *attrition_frames],
        ignore_index=True,
    )
    prefilter = connection.execute(
        "SELECT count(*) AS remaining_count FROM pilot_wide"
    ).fetchone()[0]
    raw_eligible = connection.execute(
        "SELECT count(*) FROM raw_wide WHERE dte BETWEEN 7 AND 180"
    ).fetchone()[0]
    attrition = pd.concat(
        [
            pd.DataFrame.from_records(
                [
                    {
                        "stage": "pilot_pre_iv_filter",
                        "reason": f"nearest_tenors_and_abs_log_moneyness_le_{max_abs_log_moneyness}",
                        "underlying": "SPY",
                        "quote_date": None,
                        "expiration": None,
                        "raw_count": raw_eligible * 2,
                        "remaining_count": prefilter * 2,
                    }
                ]
            ),
            attrition,
        ],
        ignore_index=True,
    )
    attrition_path = write_csv(
        output_dir / "sample_attrition.csv",
        attrition.to_dict("records"),
        sort_by=("stage", "reason", "quote_date", "expiration"),
    )

    by_tenor = stability_summary(tenor_panel, group_columns=("target_tenor",))
    by_year = stability_summary(
        tenor_panel.assign(year=pd.to_datetime(tenor_panel["quote_date"]).dt.year),
        group_columns=("year", "target_tenor"),
        coverage_scope_columns=("year",),
    )
    by_regime = stability_summary(
        tenor_panel,
        group_columns=("market_regime", "target_tenor"),
    )
    by_liquidity = stability_summary(
        tenor_panel,
        group_columns=("liquidity_quintile", "target_tenor"),
    )
    result_paths = {
        "stability_by_tenor": write_csv(output_dir / "stability_by_tenor.csv", by_tenor.to_dict("records"), sort_by=("target_tenor",)),
        "stability_by_year": write_csv(output_dir / "stability_by_year.csv", by_year.to_dict("records"), sort_by=("year", "target_tenor")),
        "stability_by_regime": write_csv(output_dir / "stability_by_regime.csv", by_regime.to_dict("records"), sort_by=("market_regime", "target_tenor")),
        "stability_by_liquidity": write_csv(output_dir / "stability_by_liquidity.csv", by_liquidity.to_dict("records"), sort_by=("liquidity_quintile", "target_tenor")),
    }

    regression_records = []
    for model_name, runner in (
        ("spy_contemporaneous_hac", run_spy_regression),
        ("quote_uncertainty_next_day_hac", run_quote_uncertainty_regression),
    ):
        try:
            _, records = runner(tenor_panel)
            records = tuple({**record, "model": model_name} for record in records)
            regression_records.extend(records)
        except (ValueError, KeyError) as error:
            regression_records.append({"model": model_name, "term": "MODEL_FAILURE", "error": str(error)})
    regression_path = write_csv(
        output_dir / "regression_results.csv",
        regression_records,
        sort_by=("model", "term"),
    )
    figure_paths = generate_core_figures(tenor_panel, expiry_panel, output_dir / "figures")
    write_json(output_dir / "regime_thresholds.json", thresholds)
    write_json(
        output_dir / "run_manifest.json",
        {
            "research_id": "research_001_atm_stability_pilot_a",
            "iv_specification": specification,
            "not_formal_ncx_baseline": specification != "baseline_rate_dividend_enrichment",
            "risk_free_rate": rate,
            "dividend_yield": dividend_yield,
            "max_abs_log_moneyness": max_abs_log_moneyness,
            "config_sha256": config.sha256,
            "raw_csv_sha256": file_sha256(raw_csv),
            "dataset_manifest_sha256": file_sha256(dataset_manifest),
            "logical_panel_hashes": {
                "expiry": logical_frame_sha256(expiry_panel, sort_by=("underlying", "quote_date", "expiration")),
                "tenor": logical_frame_sha256(tenor_panel, sort_by=("underlying", "target_tenor", "quote_date")),
            },
            "counts": {
                "raw_wide_rows": connection.execute("SELECT count(*) FROM raw_wide").fetchone()[0],
                "pilot_wide_rows": prefilter,
                "expiry_panel_rows": len(expiry_panel),
                "tenor_panel_rows": len(tenor_panel),
            },
            "regime_thresholds": thresholds,
        },
    )
    write_limitations(output_dir / "research_001_limitations.md")
    write_research_report(
        output_dir / "research_001_report.md",
        title="Research 001 Pilot A — SPY ATM Volatility Stability, 2020–2022",
        result_files={key: str(value) for key, value in result_paths.items()},
        notes=(
            "This run is a zero-rate/zero-dividend diagnostic unless explicitly labelled otherwise.",
            "The pilot prefilters to nearest target expiries and |log(K/S)| within the configured bound before IV inversion.",
            "Formal results require historical rate/dividend enrichment and the full expiry-level inversion.",
        ),
    )
    output_hashes = write_output_hashes(output_dir, output_dir / "output_hashes.json")
    connection.close()
    return {
        "config": config_path,
        "audit": audit_paths,
        "expiry_panel": expiry_path,
        "tenor_panel": tenor_path,
        "return_panel": returns_path,
        "sample_attrition": attrition_path,
        "regressions": regression_path,
        "figures": figure_paths,
        "output_hashes": output_hashes,
    }


def _prepare_database(connection: duckdb.DuckDBPyConnection, raw_csv: Path) -> None:
    existing = connection.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name='raw_wide'"
    ).fetchone()[0]
    if existing:
        return
    source = raw_csv.as_posix().replace("'", "''")
    numeric = (
        "underlying_last", "dte", "strike", "c_bid", "c_ask", "c_last", "c_volume",
        "c_iv", "c_delta", "c_gamma", "c_vega", "c_theta", "c_rho", "p_bid",
        "p_ask", "p_last", "p_volume", "p_iv", "p_delta", "p_gamma", "p_vega",
        "p_theta", "p_rho", "quote_time_hours",
    )
    casts = ",\n".join(f"try_cast(trim({name}) AS DOUBLE) AS {name}" for name in numeric)
    connection.execute(
        f"""
        CREATE TABLE raw_wide AS
        SELECT
            row_number() OVER () + 1 AS source_row,
            try_cast(trim(quote_date) AS DATE) AS quote_date,
            try_cast(trim(quote_readtime) AS TIMESTAMP) AS quote_readtime,
            try_cast(trim(expire_date) AS DATE) AS expire_date,
            {casts}
        FROM read_csv_auto('{source}', normalize_names=true, all_varchar=true)
        """
    )
    connection.execute("CREATE INDEX raw_date_expiry ON raw_wide(quote_date, expire_date)")


def _prepare_selected_expiries(connection, max_abs_log_moneyness: float) -> None:
    connection.execute("DROP TABLE IF EXISTS pilot_wide")
    target_values = ",".join(
        f"({target},{tolerance})" for target, tolerance in zip(TARGET_TENORS, TENOR_TOLERANCES)
    )
    connection.execute(
        f"""
        CREATE TABLE pilot_wide AS
        WITH expiries AS (
            SELECT DISTINCT quote_date, expire_date, CAST(round(dte) AS INTEGER) AS actual_dte
            FROM raw_wide
            WHERE quote_date BETWEEN DATE '2020-01-01' AND DATE '2022-12-31'
              AND dte BETWEEN 7 AND 180
        ), candidates AS (
            SELECT e.*, t.target_tenor, t.tolerance,
                   row_number() OVER (
                       PARTITION BY e.quote_date, t.target_tenor
                       ORDER BY abs(e.actual_dte-t.target_tenor), e.actual_dte, e.expire_date
                   ) AS choice_rank
            FROM expiries e
            CROSS JOIN (VALUES {target_values}) t(target_tenor,tolerance)
            WHERE abs(e.actual_dte-t.target_tenor) <= t.tolerance
        ), chosen AS (
            SELECT * FROM candidates WHERE choice_rank=1
        )
        SELECT r.*, c.target_tenor, r.dte-c.target_tenor AS tenor_mismatch
        FROM raw_wide r
        JOIN chosen c USING (quote_date, expire_date)
        WHERE r.strike > 0 AND r.underlying_last > 0
          AND abs(ln(r.strike/r.underlying_last)) <= {float(max_abs_log_moneyness)}
        ORDER BY r.quote_date, r.expire_date, r.strike
        """
    )
    connection.execute("CREATE INDEX pilot_date ON pilot_wide(quote_date)")


def _write_audit(connection, output_dir: Path) -> dict[str, Path]:
    summary = connection.execute(
        """
        SELECT
          count(*) AS row_count,
          count(DISTINCT quote_date) AS unique_quote_dates,
          count(DISTINCT expire_date) AS unique_expirations,
          count(DISTINCT strike) AS unique_strikes,
          min(quote_date) AS minimum_quote_date,
          max(quote_date) AS maximum_quote_date,
          sum(CASE WHEN expire_date < quote_date THEN 1 ELSE 0 END) AS expiration_before_quote,
          sum(CASE WHEN abs(dte-date_diff('day',quote_date,expire_date)) > 1 THEN 1 ELSE 0 END) AS dte_mismatch,
          sum(CASE WHEN dayofweek(quote_date) IN (0,6) THEN 1 ELSE 0 END) AS weekend_quote_rows,
          sum(CASE WHEN underlying_last IS NULL OR underlying_last <= 0 THEN 1 ELSE 0 END) AS invalid_spot,
          sum(CASE WHEN strike IS NULL OR strike <= 0 THEN 1 ELSE 0 END) AS invalid_strike,
          sum(CASE WHEN c_bid < 0 OR p_bid < 0 THEN 1 ELSE 0 END) AS negative_bid_rows,
          sum(CASE WHEN c_ask <= 0 OR p_ask <= 0 THEN 1 ELSE 0 END) AS non_positive_ask_rows,
          sum(CASE WHEN c_bid > c_ask OR p_bid > p_ask THEN 1 ELSE 0 END) AS crossed_market_rows,
          sum(CASE WHEN c_bid IS NULL OR p_bid IS NULL THEN 1 ELSE 0 END) AS missing_bid_rows,
          sum(CASE WHEN c_ask IS NULL OR p_ask IS NULL THEN 1 ELSE 0 END) AS missing_ask_rows,
          sum(CASE WHEN c_iv > 5 OR p_iv > 5 THEN 1 ELSE 0 END) AS possible_percent_iv_rows,
          sum(CASE WHEN c_vega < 0 OR p_vega < 0 THEN 1 ELSE 0 END) AS negative_vega_rows
        FROM raw_wide
        """
    ).fetchdf()
    summary_records = [
        {"metric": column, "value": summary.iloc[0][column]}
        for column in summary.columns
    ]
    summary_path = write_csv(output_dir / "audit_summary.csv", summary_records, columns=("metric", "value"), sort_by=("metric",))
    by_date = connection.execute(
        """
        SELECT quote_date, count(*) AS raw_rows, count(DISTINCT expire_date) AS expiries,
               count(DISTINCT strike) AS strikes, min(underlying_last) AS minimum_spot,
               max(underlying_last) AS maximum_spot,
               sum(CASE WHEN c_bid > c_ask OR p_bid > p_ask THEN 1 ELSE 0 END) AS crossed_market_rows,
               sum(CASE WHEN c_bid IS NULL OR c_ask IS NULL OR p_bid IS NULL OR p_ask IS NULL THEN 1 ELSE 0 END) AS missing_quote_rows
        FROM raw_wide GROUP BY quote_date ORDER BY quote_date
        """
    ).fetchdf()
    by_expiry = connection.execute(
        """
        SELECT quote_date, expire_date, min(dte) AS minimum_vendor_dte,
               max(dte) AS maximum_vendor_dte, count(*) AS raw_rows,
               count(DISTINCT strike) AS strikes,
               sum(CASE WHEN c_bid > c_ask OR p_bid > p_ask THEN 1 ELSE 0 END) AS crossed_market_rows
        FROM raw_wide GROUP BY quote_date, expire_date ORDER BY quote_date, expire_date
        """
    ).fetchdf()
    by_date_path = write_csv(output_dir / "audit_by_date.csv", by_date.to_dict("records"), sort_by=("quote_date",))
    by_expiry_path = write_csv(output_dir / "audit_by_expiry.csv", by_expiry.to_dict("records"), sort_by=("quote_date", "expire_date"))
    failure_path = output_dir / "audit_failures.csv"
    failure_sql = """
        SELECT * FROM (
          SELECT 'EXPIRATION_BEFORE_QUOTE' AS code, source_row, quote_date, expire_date, strike FROM raw_wide WHERE expire_date < quote_date
          UNION ALL SELECT 'DTE_MISMATCH', source_row, quote_date, expire_date, strike FROM raw_wide WHERE abs(dte-date_diff('day',quote_date,expire_date)) > 1
          UNION ALL SELECT 'WEEKEND_QUOTE_DATE', source_row, quote_date, expire_date, strike FROM raw_wide WHERE dayofweek(quote_date) IN (0,6)
          UNION ALL SELECT 'INVALID_SPOT', source_row, quote_date, expire_date, strike FROM raw_wide WHERE underlying_last IS NULL OR underlying_last <= 0
          UNION ALL SELECT 'INVALID_STRIKE', source_row, quote_date, expire_date, strike FROM raw_wide WHERE strike IS NULL OR strike <= 0
          UNION ALL SELECT 'NEGATIVE_BID', source_row, quote_date, expire_date, strike FROM raw_wide WHERE c_bid < 0 OR p_bid < 0
          UNION ALL SELECT 'NON_POSITIVE_ASK', source_row, quote_date, expire_date, strike FROM raw_wide WHERE c_ask <= 0 OR p_ask <= 0
          UNION ALL SELECT 'CROSSED_MARKET', source_row, quote_date, expire_date, strike FROM raw_wide WHERE c_bid > c_ask OR p_bid > p_ask
          UNION ALL SELECT 'MISSING_BID', source_row, quote_date, expire_date, strike FROM raw_wide WHERE c_bid IS NULL OR p_bid IS NULL
          UNION ALL SELECT 'MISSING_ASK', source_row, quote_date, expire_date, strike FROM raw_wide WHERE c_ask IS NULL OR p_ask IS NULL
          UNION ALL SELECT 'POSSIBLE_PERCENT_IV', source_row, quote_date, expire_date, strike FROM raw_wide WHERE c_iv > 5 OR p_iv > 5
          UNION ALL SELECT 'NEGATIVE_VENDOR_VEGA', source_row, quote_date, expire_date, strike FROM raw_wide WHERE c_vega < 0 OR p_vega < 0
        ) ORDER BY code, quote_date, expire_date, strike, source_row
    """
    escaped_failure = failure_path.as_posix().replace("'", "''")
    connection.execute(f"COPY ({failure_sql}) TO '{escaped_failure}' (HEADER, DELIMITER ',')")
    schema = connection.execute("PRAGMA table_info('raw_wide')").fetchdf()
    schema_path = write_json(
        output_dir / "schema_report.json",
        {
            "columns": schema.to_dict("records"),
            "raw_table": "raw_wide",
            "source_row_count": int(summary.iloc[0]["row_count"]),
        },
    )
    report_path = output_dir / "data_quality_report.md"
    report_path.write_text(
        "# SPY Pilot data quality report\n\n"
        f"Raw wide rows: {int(summary.iloc[0]['row_count']):,}\n\n"
        f"Coverage: {summary.iloc[0]['minimum_quote_date']} to {summary.iloc[0]['maximum_quote_date']}\n\n"
        "Audit failures are recorded without modifying the source CSV. The pilot NCX run applies a separately recorded pre-IV scope filter.\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "summary": summary_path,
        "by_date": by_date_path,
        "by_expiry": by_expiry_path,
        "failures": failure_path,
        "schema": schema_path,
        "report": report_path,
    }


def _add_realized_volatility(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel.sort_values(["underlying", "quote_date"], kind="mergesort").copy()
    data["past_realized_volatility"] = data.groupby("underlying")["underlying_return"].transform(
        lambda values: values.rolling(21, min_periods=10).std() * (252.0 ** 0.5)
    )
    return data


def _quintile(values: pd.Series) -> pd.Series:
    ranked = values.rank(method="first")
    try:
        return pd.qcut(ranked, 5, labels=[1, 2, 3, 4, 5])
    except ValueError:
        return pd.Series(pd.NA, index=values.index, dtype="Int64")


if __name__ == "__main__":
    main()
