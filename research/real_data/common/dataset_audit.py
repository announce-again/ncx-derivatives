from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from research.real_data.common._optional import require_pandas

from .calendar import missing_business_date_ranges
from .deterministic_io import write_csv, write_json


@dataclass(frozen=True, slots=True)
class DatasetAuditResult:
    summary: tuple[dict[str, Any], ...]
    by_date: tuple[dict[str, Any], ...]
    by_expiry: tuple[dict[str, Any], ...]
    failures: tuple[dict[str, Any], ...]
    schema_report: dict[str, Any]


def audit_standardized_options(frame) -> DatasetAuditResult:
    """Audit the canonical long table without modifying or dropping rows."""

    pandas = require_pandas()
    required = {
        "underlying_symbol", "quote_date", "underlying_price", "expiration",
        "vendor_dte", "strike", "option_type", "bid", "ask",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"standardized table is missing columns: {missing}")
    data = frame.copy()
    for column in ("quote_date", "expiration"):
        data[column] = pandas.to_datetime(data[column], errors="coerce")
    numeric = (
        "underlying_price", "vendor_dte", "time_to_maturity", "strike", "bid",
        "ask", "last", "volume", "open_interest", "vendor_iv", "vendor_delta",
        "vendor_gamma", "vendor_vega", "vendor_theta", "vendor_rho",
    )
    for column in numeric:
        if column in data:
            data[column] = pandas.to_numeric(data[column], errors="coerce")

    calendar_dte = (data["expiration"] - data["quote_date"]).dt.days
    flags: list[tuple[str, Any, str]] = []

    def add(code: str, mask, detail: str) -> None:
        flags.append((code, mask.fillna(False), detail))

    add("MISSING_QUOTE_DATE", data["quote_date"].isna(), "quote date is missing or invalid")
    add("MISSING_EXPIRATION", data["expiration"].isna(), "expiration is missing or invalid")
    add("EXPIRATION_BEFORE_QUOTE", data["expiration"] < data["quote_date"], "expiration precedes quote date")
    add("WEEKEND_QUOTE_DATE", data["quote_date"].dt.day_of_week >= 5, "quote date is Saturday or Sunday")
    add("DTE_MISMATCH", (data["vendor_dte"] - calendar_dte).abs() > 1.0, "vendor DTE differs from calendar DTE by more than one day")
    finite_spot = data["underlying_price"].map(_is_finite)
    finite_strike = data["strike"].map(_is_finite)
    add("INVALID_SPOT", ~(finite_spot & (data["underlying_price"] > 0)), "underlying price is not finite and positive")
    add("INVALID_STRIKE", ~(finite_strike & (data["strike"] > 0)), "strike is not finite and positive")
    add("NEGATIVE_BID", data["bid"] < 0, "bid is negative")
    add("NON_POSITIVE_ASK", data["ask"] <= 0, "ask is non-positive")
    add("CROSSED_MARKET", data["bid"] > data["ask"], "bid exceeds ask")
    add("ZERO_MARKET", (data["bid"] == 0) & (data["ask"] == 0), "bid and ask are both zero")
    add("MISSING_BID", data["bid"].isna(), "bid is missing")
    add("MISSING_ASK", data["ask"].isna(), "ask is missing")
    add("NON_FINITE_BID", data["bid"].notna() & ~data["bid"].map(_is_finite), "bid is non-finite")
    add("NON_FINITE_ASK", data["ask"].notna() & ~data["ask"].map(_is_finite), "ask is non-finite")
    call_upper = (data["option_type"] == "call") & (data["ask"] > data["underlying_price"])
    put_upper = (data["option_type"] == "put") & (data["ask"] > data["strike"])
    add("PRICE_ABOVE_BASIC_UPPER_BOUND", call_upper | put_upper, "ask exceeds the undiscounted basic upper bound")
    add("EXTREME_STRIKE_SPOT_RATIO", (data["strike"] / data["underlying_price"] < 0.05) | (data["strike"] / data["underlying_price"] > 20), "strike/spot lies outside [0.05, 20]")
    if "vendor_iv" in data:
        add("INVALID_VENDOR_IV", data["vendor_iv"].notna() & (data["vendor_iv"] <= 0), "vendor IV is non-positive")
        add("POSSIBLE_PERCENT_IV", data["vendor_iv"] > 5.0, "vendor IV may be percentage-scaled")
    if "vendor_delta" in data:
        bad_call_delta = (data["option_type"] == "call") & data["vendor_delta"].notna() & ~data["vendor_delta"].between(0, 1)
        bad_put_delta = (data["option_type"] == "put") & data["vendor_delta"].notna() & ~data["vendor_delta"].between(-1, 0)
        add("VENDOR_DELTA_OUT_OF_RANGE", bad_call_delta | bad_put_delta, "vendor delta is inconsistent with option type")
    if "vendor_vega" in data:
        add("NEGATIVE_VENDOR_VEGA", data["vendor_vega"] < 0, "vendor vega is negative")
    analytic_columns = [
        column for column in ("vendor_iv", "vendor_delta", "vendor_gamma", "vendor_vega", "vendor_theta", "vendor_rho")
        if column in data
    ]
    if analytic_columns:
        non_finite_analytic = data[analytic_columns].apply(
            lambda column: column.notna() & ~column.map(_is_finite)
        ).any(axis=1)
        add("NON_FINITE_VENDOR_ANALYTIC", non_finite_analytic, "one or more populated vendor analytics are non-finite")

    duplicate_subset = ["underlying_symbol", "quote_date", "expiration", "strike", "option_type"]
    add("DUPLICATE_CONTRACT_ROW", data.duplicated(duplicate_subset, keep=False), "duplicate underlying-date-expiry-strike-type row")
    paired_count = data.groupby(
        ["underlying_symbol", "quote_date", "expiration", "strike"], dropna=False
    )["option_type"].transform("nunique")
    add("UNPAIRED_CALL_PUT_GRID", paired_count != 2, "strike does not have both call and put records")
    spot_counts = data.groupby(["underlying_symbol", "quote_date"], dropna=False)["underlying_price"].transform("nunique")
    add("NON_UNIQUE_DAILY_SPOT", spot_counts > 1, "underlying price is not unique within date")
    stale_order = data.sort_values(
        ["underlying_symbol", "expiration", "strike", "option_type", "quote_date"],
        kind="mergesort",
    )
    stale_group = stale_order.groupby(
        ["underlying_symbol", "expiration", "strike", "option_type"], sort=False
    )
    repeated_market = (
        stale_order["bid"].eq(stale_group["bid"].shift(1))
        & stale_order["ask"].eq(stale_group["ask"].shift(1))
    )
    moved_spot = (
        stale_order["underlying_price"] / stale_group["underlying_price"].shift(1) - 1.0
    ).abs() > 0.002
    stale_mask = (repeated_market & moved_spot).reindex(data.index, fill_value=False)
    add("STALE_LOOKING_REPEAT", stale_mask, "bid/ask repeated while spot moved by more than 0.2%")
    daily_spot = data.drop_duplicates(["underlying_symbol", "quote_date"]).sort_values(
        ["underlying_symbol", "quote_date"], kind="mergesort"
    )
    daily_spot["_large_jump"] = daily_spot.groupby("underlying_symbol")["underlying_price"].pct_change().abs() > 0.35
    jump_keys = set(
        zip(
            daily_spot.loc[daily_spot["_large_jump"], "underlying_symbol"],
            daily_spot.loc[daily_spot["_large_jump"], "quote_date"],
        )
    )
    large_jump = data.apply(
        lambda row: (row["underlying_symbol"], row["quote_date"]) in jump_keys,
        axis=1,
    )
    add("LARGE_UNDERLYING_JUMP", large_jump, "absolute unadjusted daily spot return exceeds 35%; inspect split adjustment")
    if {"valuation_timestamp", "source_file"}.issubset(data.columns):
        timestamp_count = data.groupby(
            ["source_file", "quote_date"], dropna=False
        )["valuation_timestamp"].transform("nunique")
        add("INCONSISTENT_FILE_SNAPSHOT_TIME", timestamp_count > 1, "one source file/date has multiple valuation timestamps")

    failure_records = []
    for code, mask, detail in flags:
        for index in data.index[mask]:
            row = data.loc[index]
            failure_records.append(
                {
                    "code": code,
                    "detail": detail,
                    "underlying": row.get("underlying_symbol"),
                    "quote_date": _date_value(row.get("quote_date")),
                    "expiration": _date_value(row.get("expiration")),
                    "strike": row.get("strike"),
                    "option_type": row.get("option_type"),
                    "source_file": row.get("source_file"),
                    "source_row": row.get("source_row"),
                }
            )
    failure_records.sort(key=lambda item: tuple(str(item.get(key, "")) for key in ("code", "underlying", "quote_date", "expiration", "strike", "option_type", "source_file", "source_row")))

    duplicate_count = int(data.duplicated().sum())
    summary = [
        {"metric": "row_count", "value": len(data)},
        {"metric": "column_count", "value": len(data.columns)},
        {"metric": "unique_quote_dates", "value": int(data["quote_date"].nunique())},
        {"metric": "unique_expirations", "value": int(data["expiration"].nunique())},
        {"metric": "unique_strikes", "value": int(data["strike"].nunique())},
        {"metric": "file_count", "value": int(data["source_file"].nunique()) if "source_file" in data else 0},
        {"metric": "missing_value_count", "value": int(data.isna().sum().sum())},
        {
            "metric": "missing_weekday_date_count",
            "value": sum(
                len(missing_business_date_ranges(group["quote_date"].dropna()))
                for _, group in data.groupby("underlying_symbol")
            ),
        },
        {"metric": "duplicate_full_rows", "value": duplicate_count},
        {"metric": "audit_failure_count", "value": len(failure_records)},
    ]
    for code, mask, _ in flags:
        summary.append({"metric": code.lower(), "value": int(mask.sum())})

    by_date = _group_summary(data, ["underlying_symbol", "quote_date"], flags)
    by_expiry = _group_summary(data, ["underlying_symbol", "quote_date", "expiration"], flags)
    schema_report = {
        "columns": [
            {
                "name": str(column),
                "dtype": str(data[column].dtype),
                "missing_count": int(data[column].isna().sum()),
            }
            for column in data.columns
        ],
        "date_range": {
            "minimum": _date_value(data["quote_date"].min()),
            "maximum": _date_value(data["quote_date"].max()),
        },
        "expiration_range": {
            "minimum": _date_value(data["expiration"].min()),
            "maximum": _date_value(data["expiration"].max()),
        },
    }
    return DatasetAuditResult(tuple(summary), tuple(by_date), tuple(by_expiry), tuple(failure_records), schema_report)


def write_audit_outputs(result: DatasetAuditResult, output_directory: str | Path) -> dict[str, Path]:
    root = Path(output_directory)
    paths = {
        "summary": write_csv(root / "audit_summary.csv", result.summary, columns=("metric", "value"), sort_by=("metric",)),
        "by_date": write_csv(root / "audit_by_date.csv", result.by_date, sort_by=("underlying_symbol", "quote_date")),
        "by_expiry": write_csv(root / "audit_by_expiry.csv", result.by_expiry, sort_by=("underlying_symbol", "quote_date", "expiration")),
        "failures": write_csv(root / "audit_failures.csv", result.failures, sort_by=("code", "underlying", "quote_date", "expiration", "strike", "option_type")),
        "schema": write_json(root / "schema_report.json", result.schema_report),
    }
    report = root / "data_quality_report.md"
    report.write_text(_quality_report(result), encoding="utf-8", newline="\n")
    paths["report"] = report
    return paths


def _group_summary(data, keys, flags):
    pandas = require_pandas()
    working = data.copy()
    for code, mask, _ in flags:
        working[code.lower()] = mask.astype("int64")
    aggregations = {"row_count": ("strike", "size"), "unique_strikes": ("strike", "nunique")}
    aggregations.update({code.lower(): (code.lower(), "sum") for code, _, _ in flags})
    grouped = working.groupby(keys, dropna=False).agg(**aggregations).reset_index()
    for key in keys:
        if pandas.api.types.is_datetime64_any_dtype(grouped[key]):
            grouped[key] = grouped[key].dt.strftime("%Y-%m-%d")
    return grouped.to_dict("records")


def _date_value(value) -> str | None:
    if value is None or require_pandas().isna(value):
        return None
    return value.date().isoformat() if hasattr(value, "date") else str(value)


def _is_finite(value) -> bool:
    try:
        return isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _quality_report(result: DatasetAuditResult) -> str:
    values = {item["metric"]: item["value"] for item in result.summary}
    return (
        "# Data quality report\n\n"
        f"Rows audited: {values.get('row_count', 0)}\n\n"
        f"Recorded audit failures: {values.get('audit_failure_count', 0)}\n\n"
        "Failures are non-destructive diagnostics. No source row was silently dropped. "
        "Review `audit_failures.csv` before declaring a dataset eligible.\n"
    )
