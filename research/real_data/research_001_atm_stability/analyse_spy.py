from __future__ import annotations

from research.real_data.common._optional import require_pandas

MARKET_PERIODS = (
    ("2010-2012", "2010-01-01", "2012-12-31"),
    ("2013-2016", "2013-01-01", "2016-12-31"),
    ("2017-2019", "2017-01-01", "2019-12-31"),
    ("2020", "2020-01-01", "2020-12-31"),
    ("2021-2023", "2021-01-01", "2023-12-31"),
)


def add_market_period(panel):
    pandas = require_pandas()
    data = panel.copy()
    dates = pandas.to_datetime(data["quote_date"])
    data["market_period"] = None
    for label, start, end in MARKET_PERIODS:
        data.loc[dates.between(start, end), "market_period"] = label
    return data


def freeze_regime_thresholds(panel, *, level_column: str = "atm_mid_iv") -> dict[str, float]:
    values = panel[level_column].dropna()
    if values.empty:
        raise ValueError("cannot freeze regimes from an empty volatility series")
    quantiles = values.quantile([0.50, 0.80, 0.95])
    return {
        "medium": float(quantiles.loc[0.50]),
        "high": float(quantiles.loc[0.80]),
        "extreme": float(quantiles.loc[0.95]),
    }


def add_volatility_regime(panel, thresholds: dict[str, float], *, level_column: str = "atm_mid_iv"):
    pandas = require_pandas()
    required = {"medium", "high", "extreme"}
    if set(thresholds) != required:
        raise ValueError(f"thresholds must contain exactly {sorted(required)}")
    if not thresholds["medium"] <= thresholds["high"] <= thresholds["extreme"]:
        raise ValueError("regime thresholds must be monotonic")
    data = panel.copy()
    data["market_regime"] = pandas.cut(
        data[level_column],
        bins=[float("-inf"), thresholds["medium"], thresholds["high"], thresholds["extreme"], float("inf")],
        labels=["low", "medium", "high", "extreme"],
        include_lowest=True,
    )
    return data


def stability_summary(
    panel,
    *,
    group_columns: tuple[str, ...],
    coverage_scope_columns: tuple[str, ...] = (),
):
    data = panel.copy()
    observed_dates = data.groupby(list(group_columns), dropna=False)["quote_date"].transform("nunique")
    if coverage_scope_columns:
        total_dates = data.groupby(
            list(coverage_scope_columns), dropna=False
        )["quote_date"].transform("nunique").clip(lower=1)
    else:
        total_dates = max(int(data["quote_date"].nunique()), 1)
    data["_coverage"] = observed_dates / total_dates
    grouped = data.groupby(list(group_columns), dropna=False, observed=True)
    result = grouped.agg(
        count=("atm_mid_iv", "count"),
        coverage=("_coverage", "first"),
        mean_atm_iv=("atm_mid_iv", "mean"),
        median_atm_iv=("atm_mid_iv", "median"),
        atm_iv_standard_deviation=("atm_mid_iv", "std"),
        median_atm_iv_spread=("atm_iv_spread", "median"),
        median_absolute_atm_iv_change=("absolute_atm_iv_change", "median"),
        p90_absolute_atm_iv_change=("absolute_atm_iv_change", lambda values: values.quantile(0.90)),
        p95_absolute_atm_iv_change=("absolute_atm_iv_change", lambda values: values.quantile(0.95)),
        median_relative_instability=("relative_atm_iv_change", lambda values: values.abs().median()),
    ).reset_index()
    return result.sort_values(list(group_columns), kind="mergesort", ignore_index=True)
