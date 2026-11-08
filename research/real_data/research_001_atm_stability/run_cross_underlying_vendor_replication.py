from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from research.real_data.common.deterministic_io import (
    file_sha256,
    logical_frame_sha256,
    write_csv,
    write_json,
    write_output_hashes,
    write_parquet,
)
from research.real_data.common.kaggle_manifest import (
    build_dataset_manifest,
    write_dataset_manifest,
)
from research.real_data.common.regression import fit_clustered_ols
from research.real_data.common.returns import build_underlying_return_panel

from .analyse_cross_underlying import (
    add_underlying_class,
    cross_underlying_summary,
    etf_stock_summary,
)
from .build_tenor_panel import (
    add_daily_stability,
    adjacent_expiry_gaps,
    build_nearest_tenor_panel,
)
from .config import Research001Config

TARGET_TENORS = (21, 45, 90, 150)
TENOR_TOLERANCES = (7, 10, 15, 25)
SPLIT_DATES = (
    ("AAPL", "2020-08-31"),
    ("NVDA", "2021-07-20"),
    ("TSLA", "2020-08-31"),
    ("TSLA", "2022-08-25"),
)

DATASETS = {
    "SPY": {
        "directory": "spy_2020_2022",
        "files": ("spy_2020_2022.csv",),
        "slug": "kylegraupe/spy-daily-eod-options-quotes-2020-2022",
    },
    "QQQ": {
        "directory": "qqq_2020_2022",
        "files": ("qqq_2020_2022.csv",),
        "slug": "kylegraupe/qqq-daily-option-chains-q1-2020-to-q4-2022",
    },
    "AAPL": {
        "directory": "aapl_2016_2023",
        "files": ("aapl_2016_2020.csv", "aapl_2021_2023.csv"),
        "slug": "kylegraupe/aapl-options-data-2016-2020",
    },
    "NVDA": {
        "directory": "nvda_2020_2022",
        "files": ("nvda_2020_2022.csv",),
        "slug": "kylegraupe/nvda-daily-option-chains-q1-2020-to-q4-2022",
    },
    "TSLA": {
        "directory": "tsla_2019_2022",
        "files": ("tsla_2019_2022.csv",),
        "slug": "kylegraupe/tsla-daily-eod-options-quotes-2019-2022",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Research 001B vendor-IV replication")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_replication(raw_root=args.raw_root, database=args.database, output_dir=args.output_dir)


def run_replication(*, raw_root: Path, database: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    database.parent.mkdir(parents=True, exist_ok=True)
    config = Research001Config(
        dataset_ids=tuple(f"{item['slug']}:current" for item in DATASETS.values()),
        start_date=pd.Timestamp("2020-01-01").date(),
        end_date=pd.Timestamp("2022-12-31").date(),
        underlyings=tuple(DATASETS),
        target_tenors=TARGET_TENORS,
        tenor_tolerances=TENOR_TOLERANCES,
        split_policy="exclude_cross_split_changes",
    )
    config.write(output_dir / "config.json")
    connection = duckdb.connect(str(database))
    _prepare_raw_table(connection, raw_root)
    audit = _write_audit(connection, output_dir)
    _write_split_audit(connection, output_dir)
    _write_dataset_manifests(raw_root, audit)

    expiry = adjacent_expiry_gaps(_vendor_expiry_panel(connection))
    tenor = build_nearest_tenor_panel(
        expiry,
        target_tenors=TARGET_TENORS,
        tenor_tolerances=TENOR_TOLERANCES,
        minimum_smile_points=config.minimum_smile_points,
    )
    stability_inputs = tenor.drop(
        columns=[
            column
            for column in (
                "previous_observation_date", "observation_gap_days", "is_consecutive_observation",
                "previous_atm_mid_iv", "delta_atm_iv", "absolute_atm_iv_change",
                "squared_atm_iv_change", "relative_atm_iv_change",
                "previous_atm_total_variance", "delta_atm_total_variance", "crosses_split",
            )
            if column in tenor
        ]
    )
    tenor = add_daily_stability(
        stability_inputs,
        group_columns=("underlying", "target_tenor"),
        split_dates=SPLIT_DATES,
    )
    tenor["iv_specification"] = "vendor_iv_replication"
    tenor = add_underlying_class(tenor)

    spot = connection.execute(
        """SELECT underlying AS underlying_symbol, quote_date,
        median(underlying_last) AS underlying_price FROM raw_wide
        GROUP BY underlying, quote_date ORDER BY underlying, quote_date"""
    ).fetchdf()
    returns = build_underlying_return_panel(spot)
    returns["past_realized_volatility"] = returns.groupby("underlying")["underlying_return"].transform(
        lambda values: values.rolling(21, min_periods=10).std() * np.sqrt(252.0)
    )
    tenor = tenor.merge(
        returns[["underlying", "quote_date", "underlying_return", "absolute_underlying_return", "past_realized_volatility"]],
        on=["underlying", "quote_date"], how="left", validate="many_to_one",
    )
    tenor["rolling_zscore_change"] = tenor.groupby(["underlying", "target_tenor"])["delta_atm_iv"].transform(
        lambda values: (values - values.rolling(63, min_periods=20).mean()) / values.rolling(63, min_periods=20).std()
    )

    write_parquet(output_dir / "atm_expiry_panel.parquet", expiry, sort_by=("underlying", "quote_date", "expiration"))
    write_parquet(output_dir / "atm_tenor_panel.parquet", tenor, sort_by=("underlying", "target_tenor", "quote_date"))
    write_parquet(output_dir / "underlying_return_panel.parquet", returns, sort_by=("underlying", "quote_date"))
    _write_summaries(tenor, output_dir)
    _write_attrition(connection, expiry, tenor, output_dir)
    _write_regression(tenor, output_dir)
    _write_report(tenor, audit, output_dir / "research_001_report.md")
    _write_limitations(output_dir / "research_001_limitations.md")

    source_files = []
    manifest_hashes = {}
    for symbol, spec in DATASETS.items():
        directory = raw_root / spec["directory"]
        manifest = directory / "dataset_manifest.json"
        manifest_hashes[symbol] = file_sha256(manifest)
        for filename in spec["files"]:
            path = directory / filename
            source_files.append({"underlying": symbol, "name": filename, "size_bytes": path.stat().st_size, "sha256": file_sha256(path)})
    write_json(
        output_dir / "run_manifest.json",
        {
            "research_id": "research_001b_cross_underlying_vendor_iv_replication",
            "iv_specification": "vendor_iv_replication",
            "not_ncx_reconstructed_iv": True,
            "config_sha256": config.sha256,
            "dataset_manifest_sha256": manifest_hashes,
            "source_files": source_files,
            "counts": {
                "raw_wide_rows": int(connection.execute("SELECT count(*) FROM raw_wide").fetchone()[0]),
                "expiry_panel_rows": len(expiry),
                "tenor_panel_rows": len(tenor),
            },
            "logical_panel_hashes": {
                "expiry": logical_frame_sha256(expiry, sort_by=("underlying", "quote_date", "expiration")),
                "tenor": logical_frame_sha256(tenor, sort_by=("underlying", "target_tenor", "quote_date")),
            },
            "split_dates_excluded": list(SPLIT_DATES),
        },
    )
    write_output_hashes(output_dir, output_dir / "output_hashes.json")
    connection.close()


def _prepare_raw_table(connection: duckdb.DuckDBPyConnection, raw_root: Path) -> None:
    if connection.execute("SELECT count(*) FROM information_schema.tables WHERE table_name='raw_wide'").fetchone()[0]:
        return
    numeric = (
        "underlying_last", "dte", "strike", "c_bid", "c_ask", "c_last", "c_volume",
        "c_iv", "c_delta", "c_gamma", "c_vega", "c_theta", "c_rho", "p_bid",
        "p_ask", "p_last", "p_volume", "p_iv", "p_delta", "p_gamma", "p_vega",
        "p_theta", "p_rho", "quote_time_hours",
    )
    casts = ",\n".join(f"try_cast(trim({name}) AS DOUBLE) AS {name}" for name in numeric)
    selections = []
    for symbol, spec in DATASETS.items():
        for filename in spec["files"]:
            source = (raw_root / spec["directory"] / filename).as_posix().replace("'", "''")
            selections.append(
                f"""SELECT '{symbol}' AS underlying, '{spec['slug']}' AS dataset_id,
                '{filename}' AS source_file, row_number() OVER () + 1 AS source_row,
                try_cast(trim(quote_date) AS DATE) AS quote_date,
                try_cast(trim(quote_readtime) AS TIMESTAMP) AS quote_readtime,
                try_cast(trim(expire_date) AS DATE) AS expire_date,
                {casts}
                FROM read_csv_auto('{source}', normalize_names=true, all_varchar=true, sample_size=200000)
                WHERE try_cast(trim(quote_date) AS DATE) BETWEEN DATE '2020-01-01' AND DATE '2022-12-31'"""
            )
    connection.execute("CREATE TABLE raw_wide AS " + " UNION ALL ".join(selections))
    connection.execute("CREATE INDEX raw_symbol_date_expiry ON raw_wide(underlying, quote_date, expire_date)")


def _vendor_expiry_panel(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return connection.execute(
        """
        WITH points AS (
          SELECT underlying, dataset_id, quote_date, expire_date, round(dte)::INTEGER AS actual_dte,
                 underlying_last AS spot, strike, ln(strike/underlying_last) AS k,
                 CASE WHEN strike >= underlying_last THEN c_iv ELSE p_iv END AS iv,
                 CASE WHEN strike >= underlying_last THEN c_bid ELSE p_bid END AS option_bid,
                 CASE WHEN strike >= underlying_last THEN c_ask ELSE p_ask END AS option_ask,
                 CASE WHEN strike >= underlying_last THEN c_volume ELSE p_volume END AS option_volume,
                 CASE WHEN strike >= underlying_last THEN c_vega ELSE p_vega END AS option_vega
          FROM raw_wide WHERE dte BETWEEN 7 AND 180 AND strike > 0 AND underlying_last > 0
        ), valid AS (
          SELECT *, option_ask-option_bid AS price_spread,
                 CASE WHEN option_bid+option_ask > 0 THEN 2*(option_ask-option_bid)/(option_bid+option_ask) END AS relative_price_spread
          FROM points WHERE iv > 0 AND iv < 5
        ), brackets AS (
          SELECT underlying, dataset_id, quote_date, expire_date, actual_dte, spot,
                 max(k) FILTER (WHERE k <= 0) AS left_k, min(k) FILTER (WHERE k >= 0) AS right_k,
                 count(*) AS selected_point_count, median(price_spread) AS median_price_spread,
                 median(relative_price_spread) AS median_relative_price_spread,
                 sum(coalesce(option_volume,0)) AS total_volume, median(option_vega) AS median_vega
          FROM valid GROUP BY underlying, dataset_id, quote_date, expire_date, actual_dte, spot
        ), joined AS (
          SELECT b.*, l.iv AS left_iv, r.iv AS right_iv,
                 CASE WHEN b.left_k IS NULL OR b.right_k IS NULL THEN NULL
                      WHEN b.left_k=b.right_k THEN l.iv
                      ELSE sqrt((1-(-b.left_k/(b.right_k-b.left_k)))*l.iv*l.iv
                         + (-b.left_k/(b.right_k-b.left_k))*r.iv*r.iv) END AS atm_mid_iv
          FROM brackets b
          LEFT JOIN valid l ON l.underlying=b.underlying AND l.quote_date=b.quote_date AND l.expire_date=b.expire_date AND l.k=b.left_k
          LEFT JOIN valid r ON r.underlying=b.underlying AND r.quote_date=b.quote_date AND r.expire_date=b.expire_date AND r.k=b.right_k
        )
        SELECT dataset_id, underlying, quote_date, quote_date + INTERVAL 21 HOUR AS valuation_timestamp,
          expire_date AS expiration, actual_dte, spot, spot AS forward,
          NULL::DOUBLE AS risk_free_discount_factor, NULL::DOUBLE AS dividend_discount_factor,
          atm_mid_iv, NULL::DOUBLE AS atm_bid_iv, NULL::DOUBLE AS atm_ask_iv,
          atm_mid_iv*atm_mid_iv*actual_dte/365.0 AS atm_total_variance,
          'SUCCESS' AS atm_mid_status, 'NOT_AVAILABLE_VENDOR_REPLICATION' AS atm_bid_status,
          'NOT_AVAILABLE_VENDOR_REPLICATION' AS atm_ask_status,
          CASE WHEN left_k=right_k THEN 'OBSERVED' ELSE 'LINEAR_TOTAL_VARIANCE' END AS atm_method,
          left_k AS nearest_left_k, right_k AS nearest_right_k, right_k-left_k AS atm_bracket_span,
          selected_point_count, NULL::BIGINT AS excluded_quote_count,
          median_price_spread, median_relative_price_spread, total_volume,
          NULL::DOUBLE AS total_open_interest, median_vega,
          NULL::DOUBLE AS local_skew, NULL::DOUBLE AS curvature,
          NULL::DOUBLE AS call_25d_iv, NULL::DOUBLE AS put_25d_iv,
          NULL::DOUBLE AS rr25, NULL::DOUBLE AS bf25,
          NULL::DOUBLE AS atm_iv_spread, NULL::DOUBLE AS relative_atm_iv_spread
        FROM joined WHERE atm_mid_iv IS NOT NULL AND selected_point_count>=5
        ORDER BY underlying, quote_date, expire_date
        """
    ).fetchdf()


def _write_audit(connection: duckdb.DuckDBPyConnection, output_dir: Path) -> pd.DataFrame:
    audit = connection.execute(
        """
        SELECT underlying, count(*) AS row_count, count(DISTINCT quote_date) AS unique_quote_dates,
          count(DISTINCT expire_date) AS unique_expirations, count(DISTINCT strike) AS unique_strikes,
          min(quote_date) AS minimum_quote_date, max(quote_date) AS maximum_quote_date,
          sum(expire_date < quote_date)::BIGINT AS expiration_before_quote,
          sum(abs(dte-date_diff('day',quote_date,expire_date)) > 1)::BIGINT AS dte_mismatch,
          sum(dayofweek(quote_date) IN (0,6))::BIGINT AS weekend_quote_rows,
          sum(underlying_last IS NULL OR underlying_last<=0)::BIGINT AS invalid_spot,
          sum(strike IS NULL OR strike<=0)::BIGINT AS invalid_strike,
          sum(c_bid<0 OR p_bid<0)::BIGINT AS negative_bid_rows,
          sum(c_ask<=0 OR p_ask<=0)::BIGINT AS non_positive_ask_rows,
          sum(c_bid>c_ask OR p_bid>p_ask)::BIGINT AS crossed_market_rows,
          sum(c_bid IS NULL OR p_bid IS NULL)::BIGINT AS missing_bid_rows,
          sum(c_ask IS NULL OR p_ask IS NULL)::BIGINT AS missing_ask_rows,
          sum(c_iv>5 OR p_iv>5)::BIGINT AS possible_percent_iv_rows,
          sum(c_vega<0 OR p_vega<0)::BIGINT AS negative_vega_rows,
          count(DISTINCT quote_date) FILTER (WHERE quote_date IN (
            SELECT quote_date FROM raw_wide daily
            WHERE daily.underlying=raw_wide.underlying
            GROUP BY quote_date HAVING count(DISTINCT underlying_last)>1
          )) AS dates_with_multiple_spot_values
        FROM raw_wide GROUP BY underlying ORDER BY underlying
        """
    ).fetchdf()
    write_csv(output_dir / "audit_summary.csv", audit.to_dict("records"), sort_by=("underlying",))
    by_date = connection.execute(
        """SELECT underlying, quote_date, count(*) AS raw_rows, count(DISTINCT expire_date) AS expiries,
          count(DISTINCT strike) AS strikes, min(underlying_last) AS minimum_spot,
          max(underlying_last) AS maximum_spot, sum(c_bid>c_ask OR p_bid>p_ask)::BIGINT AS crossed_market_rows
          FROM raw_wide GROUP BY underlying,quote_date ORDER BY underlying,quote_date"""
    ).fetchdf()
    write_csv(output_dir / "audit_by_date.csv", by_date.to_dict("records"), sort_by=("underlying", "quote_date"))
    failure_path = output_dir / "audit_failures.csv"
    escaped = failure_path.as_posix().replace("'", "''")
    connection.execute(
        f"""COPY (SELECT underlying,source_file,source_row,quote_date,expire_date,strike,code FROM (
          SELECT *, unnest(list_filter([
            CASE WHEN expire_date<quote_date THEN 'EXPIRATION_BEFORE_QUOTE' END,
            CASE WHEN abs(dte-date_diff('day',quote_date,expire_date))>1 THEN 'DTE_MISMATCH' END,
            CASE WHEN c_bid>c_ask OR p_bid>p_ask THEN 'CROSSED_MARKET' END,
            CASE WHEN c_bid IS NULL OR p_bid IS NULL THEN 'MISSING_BID' END,
            CASE WHEN c_ask IS NULL OR p_ask IS NULL THEN 'MISSING_ASK' END,
            CASE WHEN c_iv>5 OR p_iv>5 THEN 'POSSIBLE_PERCENT_IV' END
          ], x -> x IS NOT NULL)) AS code FROM raw_wide
        ) ORDER BY underlying,code,quote_date,expire_date,strike,source_row)
        TO '{escaped}' (HEADER, DELIMITER ',')"""
    )
    schema = connection.execute("PRAGMA table_info('raw_wide')").fetchdf()
    write_json(output_dir / "schema_report.json", {"columns": schema.to_dict("records"), "raw_table": "raw_wide"})
    return audit


def _write_split_audit(connection: duckdb.DuckDBPyConnection, output_dir: Path) -> None:
    rows = []
    fields = (
        "underlying", "announced_split_date", "previous_quote_date", "first_quote_date",
        "previous_spot", "first_spot", "spot_ratio", "previous_median_strike",
        "first_median_strike", "median_strike_ratio",
    )
    for symbol, split_date in SPLIT_DATES:
        result = connection.execute(
            """
            WITH daily AS (SELECT quote_date, median(underlying_last) AS spot, median(strike) AS median_strike
              FROM raw_wide WHERE underlying=? GROUP BY quote_date),
            before AS (SELECT * FROM daily WHERE quote_date < ? ORDER BY quote_date DESC LIMIT 1),
            after AS (SELECT * FROM daily WHERE quote_date >= ? ORDER BY quote_date LIMIT 1)
            SELECT before.quote_date, after.quote_date, before.spot, after.spot,
              after.spot/before.spot, before.median_strike, after.median_strike,
              after.median_strike/before.median_strike FROM before CROSS JOIN after
            """,
            [symbol, split_date, split_date],
        ).fetchone()
        if result:
            rows.append(dict(zip(fields, (symbol, split_date, *result))))
    write_csv(output_dir / "split_audit.csv", rows, sort_by=("underlying", "announced_split_date"))


def _write_dataset_manifests(raw_root: Path, audit: pd.DataFrame) -> None:
    audit_by_symbol = audit.set_index("underlying")
    for symbol, spec in DATASETS.items():
        root = raw_root / spec["directory"]
        metadata = json.loads((root / "kaggle_metadata.json").read_text(encoding="utf-8"))
        row = audit_by_symbol.loc[symbol]
        manifest = build_dataset_manifest(
            root,
            dataset_slug=spec["slug"],
            dataset_title=metadata.get("title", spec["slug"]),
            uploader=metadata.get("ownerName", "Kyle Graupe"),
            kaggle_version=str(metadata.get("currentVersionNumber", "unknown")),
            claimed_original_source="Not named on the Kaggle dataset page",
            license_name=metadata.get("licenseName", "unknown"),
            readme_snapshot=metadata.get("description") or metadata.get("subtitle"),
            schema={"column_count": 33, "shape": "wide call/put row", "normalized_adapter": "optionsdx_kaggle"},
            date_coverage={"start": str(row["minimum_quote_date"]), "end": str(row["maximum_quote_date"]), "observed_quote_dates": int(row["unique_quote_dates"])},
            download_timestamp=datetime.now(timezone.utc),
        )
        write_dataset_manifest(root, manifest)


def _write_attrition(connection, expiry, tenor, output_dir):
    records = []
    raw = connection.execute("SELECT underlying,count(*) FROM raw_wide GROUP BY underlying").fetchall()
    valid = connection.execute(
        """SELECT underlying,count(*) FROM raw_wide WHERE dte BETWEEN 7 AND 180 AND strike>0
        AND underlying_last>0 AND (CASE WHEN strike>=underlying_last THEN c_iv ELSE p_iv END) BETWEEN 0 AND 5
        GROUP BY underlying"""
    ).fetchall()
    for stage, values in (("raw_rows", raw), ("valid_vendor_iv_points", valid)):
        for symbol, count in values:
            records.append({"stage": stage, "reason": "remaining", "underlying": symbol, "raw_count": int(count), "remaining_count": int(count)})
    for symbol, count in expiry.groupby("underlying").size().items():
        records.append({"stage": "successful_atm_metrics", "reason": "remaining", "underlying": symbol, "raw_count": int(count), "remaining_count": int(count)})
    for symbol, count in tenor.groupby("underlying").size().items():
        records.append({"stage": "nearest_tenor_observations", "reason": "remaining", "underlying": symbol, "raw_count": int(count), "remaining_count": int(count)})
    write_csv(output_dir / "sample_attrition.csv", records, sort_by=("underlying", "stage"))


def _write_summaries(tenor: pd.DataFrame, output_dir: Path) -> None:
    underlying = cross_underlying_summary(tenor)
    classes = etf_stock_summary(tenor)
    write_csv(output_dir / "stability_by_underlying.csv", underlying.to_dict("records"), sort_by=("underlying", "target_tenor"))
    write_csv(output_dir / "stability_etf_vs_stock.csv", classes.to_dict("records"), sort_by=("underlying_class", "target_tenor"))
    coverage = tenor.groupby(["underlying", "target_tenor"], as_index=False).agg(
        first_date=("quote_date", "min"), last_date=("quote_date", "max"), observations=("quote_date", "size"),
        change_observations=("absolute_atm_iv_change", "count"), crosses_split=("crosses_split", "sum"),
    )
    write_csv(output_dir / "coverage_by_underlying.csv", coverage.to_dict("records"), sort_by=("underlying", "target_tenor"))


def _write_regression(tenor: pd.DataFrame, output_dir: Path) -> None:
    data = tenor.copy()
    data["log_dte"] = np.log(data["actual_dte"])
    required = [
        "absolute_atm_iv_change", "log_dte", "median_relative_price_spread",
        "absolute_underlying_return", "underlying", "quote_date", "target_tenor",
    ]
    data = data.dropna(subset=required).copy()
    formula = (
        "absolute_atm_iv_change ~ log_dte + median_relative_price_spread + "
        "absolute_underlying_return + C(underlying) + C(quote_date) + C(target_tenor)"
    )
    _, records = fit_clustered_ols(data, formula=formula, cluster_column="underlying")
    table = [dict(record, model="vendor_iv_cross_underlying_one_way_cluster") for record in records]
    write_csv(output_dir / "regression_results.csv", table, sort_by=("term",))


def _write_report(tenor: pd.DataFrame, audit: pd.DataFrame, path: Path) -> None:
    summary = cross_underlying_summary(tenor)
    ranking = summary.groupby("underlying", as_index=False)["median_absolute_atm_iv_change"].mean().sort_values("median_absolute_atm_iv_change")
    qqq = audit.loc[audit["underlying"].eq("QQQ")].iloc[0]
    path.write_text(
        "# Research 001B — cross-underlying vendor-IV replication\n\n"
        "## Status\n\nThis is an exploratory same-uploader vendor-IV replication, not the formal NCX carry-enriched result.\n\n"
        "## Findings\n\n"
        f"- Average-of-tenor stability ranking (most stable first): {ranking['underlying'].tolist()}.\n"
        f"- QQQ actually covers {qqq['minimum_quote_date']} to {qqq['maximum_quote_date']}; its comparable window is 2021–2022 despite the title.\n"
        "- Split-crossing changes are excluded on the four preregistered split dates.\n\n"
        "## Interpretation\n\nETF-versus-stock comparisons are descriptive because the universe has only five underlyings and vendor IV model details are unavailable.\n",
        encoding="utf-8", newline="\n",
    )


def _write_limitations(path: Path) -> None:
    path.write_text(
        "# Research 001B limitations\n\n"
        "- Kaggle is a distribution platform; the uploader pages do not name the original data vendor.\n"
        "- Vendor IV is used for this cross-underlying replication and is not treated as truth.\n"
        "- QQQ has no 2020 observations in the downloaded file.\n"
        "- Bid/ask IV uncertainty is unavailable in vendor-IV replication.\n"
        "- The five-name universe and pandemic-heavy window make inference exploratory.\n"
        "- One-way clustering with five underlyings is fragile; regression p-values are not confirmatory.\n"
        "- Snapshot time is uploader-described as 4:00 pm EST but not independently timestamp-verified here.\n",
        encoding="utf-8", newline="\n",
    )


if __name__ == "__main__":
    main()
