from __future__ import annotations

from .analyse_spy import stability_summary

ETF_SYMBOLS = frozenset({"SPY", "QQQ"})


def add_underlying_class(panel):
    data = panel.copy()
    data["underlying_class"] = data["underlying"].map(
        lambda value: "ETF" if str(value).upper() in ETF_SYMBOLS else "stock"
    )
    return data


def cross_underlying_summary(panel):
    return stability_summary(
        panel,
        group_columns=("underlying", "target_tenor"),
        coverage_scope_columns=("underlying",),
    )


def etf_stock_summary(panel):
    return stability_summary(add_underlying_class(panel), group_columns=("underlying_class", "target_tenor"))
