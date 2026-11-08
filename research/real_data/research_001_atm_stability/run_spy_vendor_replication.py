from __future__ import annotations

import argparse
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
from research.real_data.common.returns import build_underlying_return_panel

from .analyse_spy import (
    add_market_period,
    add_volatility_regime,
    freeze_regime_thresholds,
    stability_summary,
)
from .build_tenor_panel import (
    add_daily_stability,
    adjacent_expiry_gaps,
    build_nearest_tenor_panel,
)
from .config import Research001Config
from .plots import generate_core_figures
from .report import write_limitations
from .run_spy_pilot import _write_audit

TARGET_TENORS = (21, 45, 90, 150)
TENOR_TOLERANCES = (7, 10, 15, 25)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SPY 2010-2023 vendor-IV replication")
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    run_replication(
        parquet_dir=arguments.parquet_dir,
        dataset_manifest=arguments.dataset_manifest,
        database=arguments.database,
        output_dir=arguments.output_dir,
    )


def run_replication(
    *,
    parquet_dir: Path,
    dataset_manifest: Path,
    database: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    database.parent.mkdir(parents=True, exist_ok=True)
    config = Research001Config(
        dataset_ids=("dudesurfin/spy-options-eod-volatility-surface-2010-2023:v2",),
        start_date=date(2010, 1, 1),
        end_date=date(2023, 12, 31),
        underlyings=("SPY",),
        target_tenors=TARGET_TENORS,
        tenor_tolerances=TENOR_TOLERANCES,
    )
    config.write(output_dir / "config.json")
    connection = duckdb.connect(str(database))
    _prepare_raw_table(connection, parquet_dir)
    _write_audit(connection, output_dir)
    expiry = _vendor_expiry_panel(connection)
    expiry = add_daily_stability(expiry, group_columns=("underlying", "expiration"))
    expiry = adjacent_expiry_gaps(expiry)
    tenor = build_nearest_tenor_panel(
        expiry,
        target_tenors=TARGET_TENORS,
        tenor_tolerances=TENOR_TOLERANCES,
        minimum_smile_points=config.minimum_smile_points,
    )
    tenor["iv_specification"] = "vendor_iv_replication"

    spot = connection.execute(
        "SELECT 'SPY' AS underlying_symbol, quote_date, underlying_last AS underlying_price FROM raw_wide GROUP BY quote_date, underlying_last ORDER BY quote_date"
    ).fetchdf()
    returns = build_underlying_return_panel(spot)
    returns["past_realized_volatility"] = returns.groupby("underlying")["underlying_return"].transform(
        lambda values: values.rolling(21, min_periods=10).std() * (252.0 ** 0.5)
    )
    tenor = tenor.merge(
        returns[["underlying", "quote_date", "underlying_return", "absolute_underlying_return", "past_realized_volatility"]],
        on=["underlying", "quote_date"],
        how="left",
        validate="many_to_one",
    )
    tenor = add_market_period(tenor)
    thresholds = freeze_regime_thresholds(tenor)
    tenor = add_volatility_regime(tenor, thresholds)
    tenor["high_vol_regime"] = tenor["market_regime"].isin(["high", "extreme"]).astype(int)
    tenor["liquidity_quintile"] = tenor.groupby("target_tenor")["median_relative_price_spread"].transform(_quintile)

    write_parquet(output_dir / "atm_expiry_panel.parquet", expiry, sort_by=("underlying", "quote_date", "expiration"))
    write_parquet(output_dir / "atm_tenor_panel.parquet", tenor, sort_by=("underlying", "target_tenor", "quote_date"))
    write_parquet(output_dir / "underlying_return_panel.parquet", returns, sort_by=("underlying", "quote_date"))
    persisted_expiry = pd.read_parquet(output_dir / "atm_expiry_panel.parquet")
    persisted_tenor = pd.read_parquet(output_dir / "atm_tenor_panel.parquet")
    _write_summaries(tenor, expiry, output_dir)
    generate_core_figures(tenor, expiry, output_dir / "figures")
    write_json(output_dir / "regime_thresholds.json", thresholds)
    write_json(
        output_dir / "run_manifest.json",
        {
            "research_id": "research_001a_spy_vendor_iv_replication",
            "iv_specification": "vendor_iv_replication",
            "not_ncx_reconstructed_iv": True,
            "config_sha256": config.sha256,
            "dataset_manifest_sha256": file_sha256(dataset_manifest),
            "source_files": [
                {"name": path.name, "sha256": file_sha256(path), "size_bytes": path.stat().st_size}
                for path in sorted(parquet_dir.glob("*.parquet"))
            ],
            "counts": {
                "raw_wide_rows": connection.execute("SELECT count(*) FROM raw_wide").fetchone()[0],
                "expiry_panel_rows": len(expiry),
                "tenor_panel_rows": len(tenor),
            },
            "logical_panel_hashes": {
                "expiry": logical_frame_sha256(persisted_expiry, sort_by=("underlying", "quote_date", "expiration")),
                "tenor": logical_frame_sha256(persisted_tenor, sort_by=("underlying", "target_tenor", "quote_date")),
            },
            "regime_thresholds": thresholds,
        },
    )
    write_limitations(output_dir / "research_001_limitations.md")
    _write_report(tenor, output_dir / "research_001_report.md")
    write_output_hashes(output_dir, output_dir / "output_hashes.json")
    connection.close()


def _prepare_raw_table(connection, parquet_dir: Path) -> None:
    if connection.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name='raw_wide'"
    ).fetchone()[0]:
        return
    glob = (parquet_dir / "*.parquet").as_posix().replace("'", "''")
    numeric = (
        "underlying_last", "dte", "strike", "c_bid", "c_ask", "c_last", "c_volume",
        "c_iv", "c_delta", "c_gamma", "c_vega", "c_theta", "c_rho", "p_bid",
        "p_ask", "p_last", "p_volume", "p_iv", "p_delta", "p_gamma", "p_vega",
        "p_theta", "p_rho", "quote_time_hours",
    )
    selections = ",\n".join(f'"[{name.upper()}]" AS {name}' for name in numeric)
    connection.execute(
        f"""
        CREATE TABLE raw_wide AS
        SELECT
          row_number() OVER () + 1 AS source_row,
          try_cast("[QUOTE_DATE]" AS DATE) AS quote_date,
          try_cast("[QUOTE_READTIME]" AS TIMESTAMP) AS quote_readtime,
          try_cast("[EXPIRE_DATE]" AS DATE) AS expire_date,
          {selections}
        FROM read_parquet('{glob}', union_by_name=true)
        """
    )
    connection.execute("CREATE INDEX raw_date_expiry ON raw_wide(quote_date, expire_date)")


def _vendor_expiry_panel(connection) -> pd.DataFrame:
    return connection.execute(
        """
        WITH points AS (
          SELECT quote_date, expire_date, round(dte)::INTEGER AS actual_dte,
                 underlying_last AS spot, strike,
                 ln(strike/underlying_last) AS k,
                 CASE WHEN strike >= underlying_last THEN c_iv ELSE p_iv END AS iv,
                 CASE WHEN strike >= underlying_last THEN c_bid ELSE p_bid END AS option_bid,
                 CASE WHEN strike >= underlying_last THEN c_ask ELSE p_ask END AS option_ask,
                 CASE WHEN strike >= underlying_last THEN c_volume ELSE p_volume END AS option_volume,
                 CASE WHEN strike >= underlying_last THEN c_vega ELSE p_vega END AS option_vega
          FROM raw_wide
          WHERE dte BETWEEN 7 AND 180 AND strike > 0 AND underlying_last > 0
        ), valid AS (
          SELECT *,
                 CASE WHEN option_bid IS NOT NULL AND option_ask IS NOT NULL THEN option_ask-option_bid END AS price_spread,
                 CASE WHEN option_bid+option_ask > 0 THEN 2*(option_ask-option_bid)/(option_bid+option_ask) END AS relative_price_spread
          FROM points WHERE iv > 0 AND iv < 5
        ), brackets AS (
          SELECT quote_date, expire_date, actual_dte, spot,
                 max(k) FILTER (WHERE k <= 0) AS left_k,
                 min(k) FILTER (WHERE k >= 0) AS right_k,
                 count(*) AS selected_point_count,
                 median(price_spread) AS median_price_spread,
                 median(relative_price_spread) AS median_relative_price_spread,
                 sum(coalesce(option_volume,0)) AS total_volume,
                 median(option_vega) AS median_vega
          FROM valid GROUP BY quote_date, expire_date, actual_dte, spot
        ), joined AS (
          SELECT b.*, l.iv AS left_iv, r.iv AS right_iv,
                 CASE
                   WHEN b.left_k IS NULL OR b.right_k IS NULL THEN NULL
                   WHEN b.left_k=b.right_k THEN l.iv
                   ELSE sqrt(
                     (1-(-b.left_k/(b.right_k-b.left_k)))*l.iv*l.iv
                     + (-b.left_k/(b.right_k-b.left_k))*r.iv*r.iv
                   )
                 END AS atm_mid_iv
          FROM brackets b
          LEFT JOIN valid l ON l.quote_date=b.quote_date AND l.expire_date=b.expire_date AND l.k=b.left_k
          LEFT JOIN valid r ON r.quote_date=b.quote_date AND r.expire_date=b.expire_date AND r.k=b.right_k
        )
        SELECT
          'dudesurfin/spy-options-eod-volatility-surface-2010-2023:v2' AS dataset_id,
          'SPY' AS underlying,
          quote_date,
          quote_date + INTERVAL 21 HOUR AS valuation_timestamp,
          expire_date AS expiration,
          actual_dte,
          spot,
          spot AS forward,
          NULL::DOUBLE AS risk_free_discount_factor,
          NULL::DOUBLE AS dividend_discount_factor,
          atm_mid_iv,
          NULL::DOUBLE AS atm_bid_iv,
          NULL::DOUBLE AS atm_ask_iv,
          atm_mid_iv*atm_mid_iv*actual_dte/365.0 AS atm_total_variance,
          CASE WHEN atm_mid_iv IS NOT NULL AND selected_point_count>=5 THEN 'SUCCESS' ELSE 'FAILED' END AS atm_mid_status,
          'NOT_AVAILABLE_VENDOR_REPLICATION' AS atm_bid_status,
          'NOT_AVAILABLE_VENDOR_REPLICATION' AS atm_ask_status,
          CASE WHEN left_k=right_k THEN 'OBSERVED' ELSE 'LINEAR_TOTAL_VARIANCE' END AS atm_method,
          left_k AS nearest_left_k,
          right_k AS nearest_right_k,
          right_k-left_k AS atm_bracket_span,
          selected_point_count,
          NULL::BIGINT AS excluded_quote_count,
          median_price_spread,
          median_relative_price_spread,
          total_volume,
          NULL::DOUBLE AS total_open_interest,
          median_vega,
          NULL::DOUBLE AS local_skew,
          NULL::DOUBLE AS curvature,
          NULL::DOUBLE AS call_25d_iv,
          NULL::DOUBLE AS put_25d_iv,
          NULL::DOUBLE AS rr25,
          NULL::DOUBLE AS bf25,
          NULL::DOUBLE AS atm_iv_spread,
          NULL::DOUBLE AS relative_atm_iv_spread
        FROM joined
        WHERE atm_mid_iv IS NOT NULL AND selected_point_count>=5
        ORDER BY quote_date, expire_date
        """
    ).fetchdf()


def _write_summaries(tenor, expiry, output_dir):
    by_tenor = stability_summary(tenor, group_columns=("target_tenor",))
    yearly = tenor.assign(year=pd.to_datetime(tenor["quote_date"]).dt.year)
    by_year = stability_summary(yearly, group_columns=("year", "target_tenor"), coverage_scope_columns=("year",))
    by_period = stability_summary(tenor, group_columns=("market_period", "target_tenor"))
    by_regime = stability_summary(tenor, group_columns=("market_regime", "target_tenor"))
    by_liquidity = stability_summary(tenor, group_columns=("liquidity_quintile", "target_tenor"))
    for name, frame, sort_by in (
        ("stability_by_tenor.csv", by_tenor, ("target_tenor",)),
        ("stability_by_year.csv", by_year, ("year", "target_tenor")),
        ("stability_by_period.csv", by_period, ("market_period", "target_tenor")),
        ("stability_by_regime.csv", by_regime, ("market_regime", "target_tenor")),
        ("stability_by_liquidity.csv", by_liquidity, ("liquidity_quintile", "target_tenor")),
    ):
        write_csv(output_dir / name, frame.to_dict("records"), sort_by=sort_by)
    robustness = []
    for target, group in tenor.groupby("target_tenor"):
        robustness.append(
            {
                "target_tenor": int(target),
                "median_absolute_atm_iv_change": group["absolute_atm_iv_change"].median(),
                "median_absolute_relative_atm_iv_change": group["relative_atm_iv_change"].abs().median(),
                "median_absolute_total_variance_change": group["delta_atm_total_variance"].abs().median(),
                "median_absolute_relative_total_variance_change": (
                    group["delta_atm_total_variance"].abs() / group["previous_atm_total_variance"]
                ).median(),
            }
        )
    write_csv(output_dir / "robustness_results.csv", robustness, sort_by=("target_tenor",))


def _write_report(tenor, path):
    summary = stability_summary(tenor, group_columns=("target_tenor",)).set_index("target_tenor")
    regime = stability_summary(tenor, group_columns=("market_regime", "target_tenor"))
    target = regime.loc[regime["target_tenor"] == 21].set_index("market_regime")
    ratio = target.loc["extreme", "median_absolute_atm_iv_change"] / target.loc["low", "median_absolute_atm_iv_change"]
    ranking = summary.sort_values("median_absolute_atm_iv_change").index.astype(int).tolist()
    path.write_text(
        "# Research 001A — SPY vendor-IV replication, 2010–2023\n\n"
        "## Status\n\n"
        "This is the preregistered `vendor_iv_replication`, not the formal NCX rate/dividend-enriched result. "
        "Vendor OTM IV is interpolated in variance at spot ATM.\n\n"
        "## Findings\n\n"
        f"- Median daily |ΔATMIV|: 21D={summary.loc[21,'median_absolute_atm_iv_change']*100:.3f}, "
        f"45D={summary.loc[45,'median_absolute_atm_iv_change']*100:.3f}, "
        f"90D={summary.loc[90,'median_absolute_atm_iv_change']*100:.3f}, "
        f"150D={summary.loc[150,'median_absolute_atm_iv_change']*100:.3f} volatility points.\n"
        f"- Stability ranking from most to least stable is {ranking}.\n"
        f"- Extreme-regime 21D median instability is {ratio:.2f}× the low-regime median.\n\n"
        "## Limits\n\n"
        "- Vendor IV model details are not fully documented and vendor IV is not treated as truth.\n"
        "- Bid/ask IV uncertainty cannot be reconstructed from vendor IV alone.\n"
        "- Formal conclusions remain conditional on NCX reconstruction with historical carry enrichment.\n",
        encoding="utf-8",
        newline="\n",
    )


def _quintile(values):
    ranked = values.rank(method="first")
    return pd.qcut(ranked, 5, labels=[1, 2, 3, 4, 5])


if __name__ == "__main__":
    main()
