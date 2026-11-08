from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from math import exp, isfinite
from pathlib import Path

import pandas as pd

from ncx_derivatives.market_data import (
    CarryAssumptions,
    FlatDividendYieldCurve,
    FlatZeroRateCurve,
)

TREASURY_MATURITIES = {
    "DGS1MO": 1.0 / 12.0,
    "DGS3MO": 0.25,
    "DGS6MO": 0.50,
    "DGS1": 1.0,
    "DGS2": 2.0,
}
RATE_COLUMNS = tuple(TREASURY_MATURITIES) + ("DFF", "SOFR")
SUPPORTED_CARRY_SPECIFICATIONS = (
    "treasury_projected_dividend_schedule",
    "treasury_trailing_dividend_yield",
    "flat_3m_treasury_projected_dividend_schedule",
    "sofr_flat_projected_dividend_schedule",
    "treasury_realized_dividend_schedule_diagnostic",
)


@dataclass(frozen=True, slots=True)
class InterpolatedZeroRateCurve:
    """Linear interpolation of continuously compounded zero-rate proxies."""

    maturities: tuple[float, ...]
    rates: tuple[float, ...]
    interpolation_policy: str = "linear_zero_rate_flat_extrapolation"

    def __post_init__(self) -> None:
        if len(self.maturities) != len(self.rates) or not self.maturities:
            raise ValueError("maturities and rates must be non-empty and have equal length")
        if any(not isfinite(value) or value <= 0.0 for value in self.maturities):
            raise ValueError("maturities must be positive and finite")
        if tuple(sorted(self.maturities)) != self.maturities or len(set(self.maturities)) != len(self.maturities):
            raise ValueError("maturities must be strictly increasing")
        if any(not isfinite(value) for value in self.rates):
            raise ValueError("rates must be finite")

    def zero_rate(self, maturity: float) -> float:
        _validate_maturity(maturity)
        if maturity <= self.maturities[0]:
            return self.rates[0]
        if maturity >= self.maturities[-1]:
            return self.rates[-1]
        right = bisect_right(self.maturities, maturity)
        left = right - 1
        span = self.maturities[right] - self.maturities[left]
        weight = (maturity - self.maturities[left]) / span
        return self.rates[left] + weight * (self.rates[right] - self.rates[left])

    def discount_factor(self, maturity: float) -> float:
        return exp(-self.zero_rate(maturity) * maturity)


@dataclass(frozen=True, slots=True)
class CashDividendScheduleCurve:
    """Equivalent dividend discount factor from a dated cash-dividend schedule."""

    quote_date: date
    spot: float
    ex_dates: tuple[date, ...]
    cash_amounts: tuple[float, ...]
    risk_free_curve: object
    interpolation_policy: str = "dated_cash_dividend_schedule"

    def __post_init__(self) -> None:
        if not isinstance(self.quote_date, date):
            raise ValueError("quote_date must be a date")
        if not isfinite(self.spot) or self.spot <= 0.0:
            raise ValueError("spot must be positive and finite")
        if len(self.ex_dates) != len(self.cash_amounts):
            raise ValueError("ex_dates and cash_amounts must have equal length")
        if tuple(sorted(self.ex_dates)) != self.ex_dates:
            raise ValueError("ex_dates must be sorted")
        if any(value < 0.0 or not isfinite(value) for value in self.cash_amounts):
            raise ValueError("cash_amounts must be finite and non-negative")
        if not hasattr(self.risk_free_curve, "discount_factor"):
            raise ValueError("risk_free_curve must provide discount_factor")

    def discount_factor(self, maturity: float) -> float:
        _validate_maturity(maturity)
        horizon = self.quote_date + timedelta(days=round(maturity * 365.0))
        present_value = 0.0
        for ex_date, amount in zip(self.ex_dates, self.cash_amounts):
            if ex_date <= self.quote_date:
                continue
            if ex_date > horizon:
                break
            dividend_maturity = (ex_date - self.quote_date).days / 365.0
            present_value += amount * self.risk_free_curve.discount_factor(dividend_maturity)
        return max((self.spot - present_value) / self.spot, 1.0e-8)


def load_fred_rate_panel(raw_directory: str | Path) -> pd.DataFrame:
    root = Path(raw_directory)
    daily = pd.read_csv(root / "daily.csv", parse_dates=["observation_date"])
    fed_funds = pd.read_csv(root / "daily,_7-day.csv", parse_dates=["observation_date"])
    data = daily.merge(fed_funds, on="observation_date", how="outer", validate="one_to_one")
    data = data.rename(columns={"observation_date": "date"}).sort_values("date", kind="mergesort")
    calendar = pd.DataFrame({"date": pd.date_range("2009-01-01", "2024-01-31", freq="D")})
    data = calendar.merge(data, on="date", how="left", validate="one_to_one")
    for column in RATE_COLUMNS:
        if column not in data:
            data[column] = float("nan")
        data[column] = pd.to_numeric(data[column], errors="coerce")
        observed_date = data["date"].where(data[column].notna()).ffill()
        data[f"{column}_observation_date"] = observed_date
        data[f"{column}_staleness_days"] = (data["date"] - observed_date).dt.days
        data[column] = data[column].ffill(limit=7)
    return data.loc[data["date"].between("2009-01-01", "2024-01-31")].reset_index(drop=True)


def load_spy_distributions(workbook: str | Path) -> pd.DataFrame:
    raw = pd.read_excel(workbook, sheet_name="dividend")
    data = raw.loc[raw["TICKER"].astype(str).str.strip().eq("SPY")].copy()
    data["ex_date"] = pd.to_datetime(data["EX-DATE"], errors="coerce").dt.normalize()
    data["record_date"] = pd.to_datetime(data["RECORD DATE"], errors="coerce").dt.normalize()
    data["payable_date"] = pd.to_datetime(data["PAYABLE DATE"], errors="coerce").dt.normalize()
    data["cash_dividend"] = pd.to_numeric(data["DIVIDEND ($)"], errors="coerce")
    data["short_term_capital_gain"] = pd.to_numeric(data["SHORT TERM CAPITAL GAIN ($)"], errors="coerce")
    data["long_term_capital_gain"] = pd.to_numeric(data["LONG TERM CAPITAL GAIN ($)"], errors="coerce")
    result = data.loc[
        data["ex_date"].notna() & data["cash_dividend"].notna(),
        ["ex_date", "record_date", "payable_date", "cash_dividend", "short_term_capital_gain", "long_term_capital_gain", "FREQUENCY"],
    ].rename(columns={"FREQUENCY": "frequency"})
    return result.sort_values("ex_date", kind="mergesort", ignore_index=True)


def rate_row_map(rate_panel: pd.DataFrame) -> dict[date, dict[str, object]]:
    records = rate_panel.copy()
    records["date"] = pd.to_datetime(records["date"]).dt.date
    return {record.pop("date"): record for record in records.to_dict("records")}


def treasury_curve(rate_row: Mapping[str, object]) -> InterpolatedZeroRateCurve:
    points = []
    for series, maturity in TREASURY_MATURITIES.items():
        value = _finite_or_none(rate_row.get(series))
        if value is not None:
            points.append((maturity, value / 100.0))
    if len(points) < 2:
        raise ValueError("at least two Treasury curve points are required")
    maturities, rates = zip(*points)
    return InterpolatedZeroRateCurve(tuple(maturities), tuple(rates))


def build_carry_assumptions(
    *,
    specification: str,
    quote_date: date,
    spot: float,
    rate_row: Mapping[str, object],
    distributions: pd.DataFrame,
) -> CarryAssumptions:
    if specification not in SUPPORTED_CARRY_SPECIFICATIONS:
        raise ValueError(f"unsupported carry specification: {specification}")
    treasury = treasury_curve(rate_row)
    if specification == "flat_3m_treasury_projected_dividend_schedule":
        rate = _required_rate(rate_row, "DGS3MO")
        risk_curve = FlatZeroRateCurve(rate / 100.0)
    elif specification == "sofr_flat_projected_dividend_schedule":
        rate = _required_rate(rate_row, "SOFR")
        risk_curve = FlatZeroRateCurve(rate / 100.0)
    else:
        risk_curve = treasury

    if specification == "treasury_trailing_dividend_yield":
        trailing_cash = trailing_dividend_cash(distributions, quote_date)
        dividend_curve = FlatDividendYieldCurve(trailing_cash / spot)
    else:
        realized = specification == "treasury_realized_dividend_schedule_diagnostic"
        ex_dates, amounts = projected_dividend_schedule(
            distributions,
            quote_date=quote_date,
            realized_amounts=realized,
        )
        dividend_curve = CashDividendScheduleCurve(
            quote_date=quote_date,
            spot=spot,
            ex_dates=ex_dates,
            cash_amounts=amounts,
            risk_free_curve=risk_curve,
        )
    return CarryAssumptions(risk_free_curve=risk_curve, dividend_curve=dividend_curve)


def projected_dividend_schedule(
    distributions: pd.DataFrame,
    *,
    quote_date: date,
    realized_amounts: bool = False,
    maximum_horizon_days: int = 370,
) -> tuple[tuple[date, ...], tuple[float, ...]]:
    dates = pd.to_datetime(distributions["ex_date"]).dt.date
    known = distributions.loc[dates <= quote_date]
    if known.empty:
        raise ValueError(f"no known SPY distribution on or before {quote_date}")
    last_known_amount = float(known.iloc[-1]["cash_dividend"])
    horizon = quote_date + timedelta(days=maximum_horizon_days)
    future = distributions.loc[(dates > quote_date) & (dates <= horizon)].copy()
    future_dates = tuple(pd.to_datetime(future["ex_date"]).dt.date)
    if realized_amounts:
        future_amounts = tuple(float(value) for value in future["cash_dividend"])
    else:
        future_amounts = tuple(last_known_amount for _ in future_dates)
    return future_dates, future_amounts


def trailing_dividend_cash(distributions: pd.DataFrame, quote_date: date) -> float:
    dates = pd.to_datetime(distributions["ex_date"]).dt.date
    start = quote_date - timedelta(days=365)
    values = distributions.loc[(dates > start) & (dates <= quote_date), "cash_dividend"]
    return float(pd.to_numeric(values, errors="coerce").sum())


def carry_record(
    *,
    quote_date: date,
    expiration: date,
    spot: float,
    rate_row: Mapping[str, object],
    distributions: pd.DataFrame,
) -> dict[str, object]:
    maturity = (expiration - quote_date).days / 365.0
    record: dict[str, object] = {
        "quote_date": quote_date,
        "expiration": expiration,
        "actual_dte": (expiration - quote_date).days,
        "time_to_maturity": maturity,
        "spot": spot,
    }
    for series in RATE_COLUMNS:
        value = _finite_or_none(rate_row.get(series))
        record[series.lower()] = value
        record[f"{series.lower()}_staleness_days"] = rate_row.get(f"{series}_staleness_days")
    for specification in SUPPORTED_CARRY_SPECIFICATIONS:
        try:
            carry = build_carry_assumptions(
                specification=specification,
                quote_date=quote_date,
                spot=spot,
                rate_row=rate_row,
                distributions=distributions,
            )
        except ValueError:
            record[f"{specification}_available"] = False
            record[f"{specification}_risk_free_discount_factor"] = None
            record[f"{specification}_dividend_discount_factor"] = None
            record[f"{specification}_forward"] = None
            continue
        record[f"{specification}_available"] = True
        record[f"{specification}_risk_free_discount_factor"] = carry.risk_free_discount_factor(maturity)
        record[f"{specification}_dividend_discount_factor"] = carry.dividend_discount_factor(maturity)
        record[f"{specification}_forward"] = carry.forward_price(spot, maturity)
    record["trailing_12m_cash_dividend"] = trailing_dividend_cash(distributions, quote_date)
    record["trailing_12m_dividend_yield"] = record["trailing_12m_cash_dividend"] / spot
    projected_dates, projected_amounts = projected_dividend_schedule(distributions, quote_date=quote_date)
    record["projected_dividend_count_to_expiry"] = sum(value <= expiration for value in projected_dates)
    record["projected_dividend_cash_to_expiry"] = sum(
        amount for value, amount in zip(projected_dates, projected_amounts) if value <= expiration
    )
    return record


def _required_rate(rate_row: Mapping[str, object], series: str) -> float:
    value = _finite_or_none(rate_row.get(series))
    if value is None:
        raise ValueError(f"missing {series} rate")
    return value


def _finite_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _validate_maturity(maturity: float) -> None:
    if not isfinite(maturity) or maturity < 0.0:
        raise ValueError("maturity must be non-negative and finite")
