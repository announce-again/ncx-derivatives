from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from research.real_data.common.deterministic_io import (
    file_sha256,
    logical_frame_sha256,
    write_json,
    write_output_hashes,
    write_parquet,
)

from .historical_carry import (
    RATE_COLUMNS,
    SUPPORTED_CARRY_SPECIFICATIONS,
    build_carry_assumptions,
    load_fred_rate_panel,
    load_spy_distributions,
    projected_dividend_schedule,
    rate_row_map,
    trailing_dividend_cash,
)

FRED_DOWNLOAD_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv?"
    "id=DGS1MO,DGS3MO,DGS6MO,DGS1,DGS2,DFF,SOFR&cosd=2009-01-01&coed=2024-01-31"
)
STATE_STREET_DISTRIBUTION_URL = (
    "https://www.ssga.com/library-content/products/fund-data/etfs/us/"
    "spdr-etf-historical-distributions.xlsx"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Research 001 historical carry enrichment")
    parser.add_argument("--fred-raw-dir", type=Path, required=True)
    parser.add_argument("--distribution-workbook", type=Path, required=True)
    parser.add_argument("--options-database", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build_historical_carry(
        fred_raw_dir=args.fred_raw_dir,
        distribution_workbook=args.distribution_workbook,
        options_database=args.options_database,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
    )


def build_historical_carry(
    *,
    fred_raw_dir: Path,
    distribution_workbook: Path,
    options_database: Path,
    processed_dir: Path,
    output_dir: Path,
) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    rates = load_fred_rate_panel(fred_raw_dir)
    distributions = load_spy_distributions(distribution_workbook)
    rate_map = rate_row_map(rates)

    connection = duckdb.connect(str(options_database), read_only=True)
    expiries = connection.execute(
        """
        SELECT quote_date, expire_date AS expiration, round(dte)::INTEGER AS actual_dte,
               median(underlying_last) AS spot
        FROM raw_wide
        WHERE dte BETWEEN 7 AND 180 AND underlying_last > 0
        GROUP BY quote_date, expire_date, round(dte)::INTEGER
        ORDER BY quote_date, expire_date
        """
    ).fetchdf()
    records = []
    missing_rate_dates = set()
    for timestamp, daily_expiries in expiries.groupby("quote_date", sort=True):
        quote_date = pd.Timestamp(timestamp).date()
        rate_row = rate_map.get(quote_date)
        if rate_row is None:
            missing_rate_dates.add(quote_date)
            continue
        spot = float(daily_expiries.iloc[0]["spot"])
        carries = {}
        for specification in SUPPORTED_CARRY_SPECIFICATIONS:
            try:
                carries[specification] = build_carry_assumptions(
                    specification=specification,
                    quote_date=quote_date,
                    spot=spot,
                    rate_row=rate_row,
                    distributions=distributions,
                )
            except ValueError:
                carries[specification] = None
        trailing_cash = trailing_dividend_cash(distributions, quote_date)
        projected_dates, projected_amounts = projected_dividend_schedule(
            distributions, quote_date=quote_date
        )
        for row in daily_expiries.itertuples(index=False):
            expiration = pd.Timestamp(row.expiration).date()
            maturity = (expiration - quote_date).days / 365.0
            record = {
                "quote_date": quote_date,
                "expiration": expiration,
                "actual_dte": (expiration - quote_date).days,
                "time_to_maturity": maturity,
                "spot": spot,
                "trailing_12m_cash_dividend": trailing_cash,
                "trailing_12m_dividend_yield": trailing_cash / spot,
                "projected_dividend_count_to_expiry": sum(value <= expiration for value in projected_dates),
                "projected_dividend_cash_to_expiry": sum(
                    amount for value, amount in zip(projected_dates, projected_amounts) if value <= expiration
                ),
            }
            for series in RATE_COLUMNS:
                value = rate_row.get(series)
                record[series.lower()] = None if pd.isna(value) else float(value)
                record[f"{series.lower()}_staleness_days"] = rate_row.get(f"{series}_staleness_days")
            for specification, carry in carries.items():
                record[f"{specification}_available"] = carry is not None
                record[f"{specification}_risk_free_discount_factor"] = (
                    None if carry is None else carry.risk_free_discount_factor(maturity)
                )
                record[f"{specification}_dividend_discount_factor"] = (
                    None if carry is None else carry.dividend_discount_factor(maturity)
                )
                record[f"{specification}_forward"] = (
                    None if carry is None else carry.forward_price(spot, maturity)
                )
            records.append(record)
    if missing_rate_dates:
        raise ValueError(f"missing rate rows for quote dates: {sorted(missing_rate_dates)[:5]}")
    carry_panel = pd.DataFrame.from_records(records).sort_values(
        ["quote_date", "expiration"], kind="mergesort", ignore_index=True
    )

    rate_path = write_parquet(processed_dir / "historical_rate_panel.parquet", rates, sort_by=("date",))
    distribution_path = write_parquet(
        processed_dir / "spy_distribution_history.parquet",
        distributions,
        sort_by=("ex_date",),
    )
    carry_path = write_parquet(
        processed_dir / "carry_expiry_panel.parquet",
        carry_panel,
        sort_by=("quote_date", "expiration"),
    )
    implied = _option_implied_forward_diagnostic(connection, carry_panel)
    implied_path = write_parquet(
        processed_dir / "option_implied_forward_diagnostic.parquet",
        implied,
        sort_by=("quote_date", "expiration"),
    )
    connection.close()

    raw_files = [
        path
        for path in sorted(fred_raw_dir.iterdir())
        if path.is_file()
    ] + [distribution_workbook]
    source_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fred_download_url": FRED_DOWNLOAD_URL,
        "state_street_distribution_url": STATE_STREET_DISTRIBUTION_URL,
        "rate_series": list(RATE_COLUMNS),
        "treasury_units": "percent, quoted on an investment basis; treated as continuously compounded zero-rate proxies",
        "sofr_policy": "flat overnight-rate diagnostic for available dates only; not a term OIS curve",
        "dividend_policy": {
            "baseline": "future quarterly ex-dates with each amount projected from the last distribution known by quote date",
            "alternative": "trailing-365-day cash distributions divided by contemporaneous spot",
            "diagnostic": "realized future distribution amounts; contains look-ahead and is never the baseline",
        },
        "raw_files": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in raw_files
        ],
        "coverage": {
            "rate_start": str(pd.to_datetime(rates["date"]).min().date()),
            "rate_end": str(pd.to_datetime(rates["date"]).max().date()),
            "spy_distribution_start": str(pd.to_datetime(distributions["ex_date"]).min().date()),
            "spy_distribution_end": str(pd.to_datetime(distributions["ex_date"]).max().date()),
            "spy_distributions_2010_2023": int(
                pd.to_datetime(distributions["ex_date"]).between("2010-01-01", "2023-12-31").sum()
            ),
        },
    }
    manifest_path = write_json(output_dir / "carry_source_manifest.json", source_manifest)
    coverage = _coverage_report(rates, distributions, carry_panel, implied)
    write_json(output_dir / "carry_coverage_report.json", coverage)
    report = output_dir / "historical_carry_methodology.md"
    report.write_text(
        "# Historical carry enrichment\n\n"
        "## Risk-free curve\n\n"
        "The baseline linearly interpolates DGS1MO, DGS3MO, DGS6MO, DGS1, and DGS2 yields in maturity, "
        "with flat endpoint extrapolation, then uses `D_r(t,T)=exp(-r(t,T)T)`. Treasury constant-maturity "
        "yields are investment-basis market yields, not option-financing zero rates; the construction is a proxy.\n\n"
        "The robustness set contains a flat DGS3MO curve and a flat SOFR diagnostic where SOFR exists. "
        "The SOFR diagnostic is not represented as a term OIS curve.\n\n"
        "## Dividends\n\n"
        "The baseline uses State Street SPY quarterly ex-dates. Each future payment is projected using the most "
        "recent cash distribution known by the quote date. Its present value is converted to an equivalent "
        "dividend discount factor `(S-PV(dividends))/S`, which preserves quarterly timing without using future amounts.\n\n"
        "Alternative specifications use trailing-12-month cash yield and realized future cash distributions. "
        "The realized schedule contains look-ahead and is diagnostic only.\n\n"
        "## Option-implied forward\n\n"
        "A put-call-parity forward is calculated from paired call/put midpoints near spot. SPY options are American, "
        "so early exercise and quote noise make this a diagnostic rather than a carry input.\n",
        encoding="utf-8",
        newline="\n",
    )
    write_json(
        output_dir / "carry_run_manifest.json",
        {
            "source_manifest_sha256": file_sha256(manifest_path),
            "input_options_database": str(options_database),
            "outputs": {
                "rates": {"path": str(rate_path), "logical_sha256": logical_frame_sha256(rates, sort_by=("date",))},
                "distributions": {"path": str(distribution_path), "logical_sha256": logical_frame_sha256(distributions, sort_by=("ex_date",))},
                "carry": {"path": str(carry_path), "logical_sha256": logical_frame_sha256(carry_panel, sort_by=("quote_date", "expiration"))},
                "option_implied": {"path": str(implied_path), "logical_sha256": logical_frame_sha256(implied, sort_by=("quote_date", "expiration"))},
            },
            "record_counts": {
                "rates": len(rates),
                "distributions": len(distributions),
                "carry_expiries": len(carry_panel),
                "option_implied_expiries": len(implied),
            },
        },
    )
    write_output_hashes(output_dir, output_dir / "output_hashes.json")


def _option_implied_forward_diagnostic(
    connection: duckdb.DuckDBPyConnection,
    carry_panel: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "quote_date", "expiration", "spot", "time_to_maturity",
        "treasury_projected_dividend_schedule_risk_free_discount_factor",
        "treasury_projected_dividend_schedule_forward",
    ]
    frames = []
    for year in range(2010, 2024):
        registered = carry_panel.loc[
            pd.to_datetime(carry_panel["quote_date"]).dt.year.eq(year), columns
        ].copy()
        connection.register("carry_for_parity", registered)
        result = connection.execute(
        """
        WITH candidates AS (
          SELECT r.quote_date, r.expire_date AS expiration, c.spot, c.time_to_maturity,
            c.treasury_projected_dividend_schedule_risk_free_discount_factor AS risk_free_discount_factor,
            c.treasury_projected_dividend_schedule_forward AS baseline_forward,
            r.strike,
            r.strike + (((r.c_bid+r.c_ask)/2.0)-((r.p_bid+r.p_ask)/2.0)) /
              c.treasury_projected_dividend_schedule_risk_free_discount_factor AS parity_forward
          FROM raw_wide r JOIN carry_for_parity c
            ON r.quote_date=c.quote_date AND r.expire_date=c.expiration
          WHERE r.dte BETWEEN 7 AND 180 AND r.strike>0 AND r.underlying_last>0
            AND abs(ln(r.strike/r.underlying_last))<=0.10
            AND r.c_bid>=0 AND r.p_bid>=0 AND r.c_ask>0 AND r.p_ask>0
            AND r.c_bid<=r.c_ask AND r.p_bid<=r.p_ask
        ), bounded AS (
          SELECT * FROM candidates WHERE parity_forward BETWEEN 0.75*spot AND 1.25*spot
        ), medians AS (
          SELECT quote_date,expiration,spot,time_to_maturity,risk_free_discount_factor,baseline_forward,
            median(parity_forward) AS option_implied_forward, count(*) AS parity_pair_count
          FROM bounded GROUP BY ALL
        )
        SELECT m.*,
          median(abs(b.parity_forward-m.option_implied_forward)) AS parity_forward_mad,
          m.option_implied_forward*m.risk_free_discount_factor/m.spot AS option_implied_dividend_discount_factor,
          CASE WHEN m.time_to_maturity>0 AND m.option_implied_forward*m.risk_free_discount_factor/m.spot>0
            THEN -ln(m.option_implied_forward*m.risk_free_discount_factor/m.spot)/m.time_to_maturity END
            AS option_implied_dividend_yield,
          m.option_implied_forward-m.baseline_forward AS option_implied_minus_baseline_forward
        FROM medians m JOIN bounded b USING (quote_date,expiration)
        GROUP BY ALL ORDER BY m.quote_date,m.expiration
        """,
        ).fetchdf()
        connection.unregister("carry_for_parity")
        frames.append(result)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["quote_date", "expiration"], kind="mergesort", ignore_index=True
    )


def _coverage_report(
    rates: pd.DataFrame,
    distributions: pd.DataFrame,
    carry: pd.DataFrame,
    implied: pd.DataFrame,
) -> dict[str, object]:
    quote_dates = pd.to_datetime(carry["quote_date"])
    return {
        "quote_date_count": int(quote_dates.nunique()),
        "carry_expiry_count": len(carry),
        "treasury_baseline_available_rate": float(carry["treasury_projected_dividend_schedule_available"].mean()),
        "flat_3m_available_rate": float(carry["flat_3m_treasury_projected_dividend_schedule_available"].mean()),
        "sofr_available_rate": float(carry["sofr_flat_projected_dividend_schedule_available"].mean()),
        "sofr_first_available_date": str(pd.to_datetime(rates.loc[rates["SOFR"].notna(), "date"]).min().date()),
        "maximum_treasury_staleness_days": int(max(rates[f"{series}_staleness_days"].max() for series in RATE_COLUMNS[:5])),
        "spy_distribution_count": len(distributions),
        "option_implied_forward_coverage": float(len(implied) / len(carry)),
        "median_absolute_option_implied_forward_difference": float(implied["option_implied_minus_baseline_forward"].abs().median()),
    }


if __name__ == "__main__":
    main()
