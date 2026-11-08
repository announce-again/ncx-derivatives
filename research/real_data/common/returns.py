from __future__ import annotations

from collections.abc import Iterable

from research.real_data.common._optional import require_pandas


def build_underlying_return_panel(
    standardized_options,
    *,
    split_dates: Iterable[tuple[str, object]] = (),
):
    """Build snapshot-aligned unadjusted returns and suppress split crossings."""

    pandas = require_pandas()
    required = {"underlying_symbol", "quote_date", "underlying_price"}
    missing = sorted(required.difference(standardized_options.columns))
    if missing:
        raise ValueError(f"standardized options are missing columns: {missing}")
    daily = standardized_options.loc[:, list(required)].copy()
    daily["quote_date"] = pandas.to_datetime(daily["quote_date"], errors="coerce")
    uniqueness = daily.groupby(["underlying_symbol", "quote_date"])["underlying_price"].nunique()
    if (uniqueness > 1).any():
        raise ValueError("underlying price must be unique within symbol/date")
    daily = daily.drop_duplicates(["underlying_symbol", "quote_date"]).sort_values(
        ["underlying_symbol", "quote_date"], kind="mergesort", ignore_index=True
    )
    group = daily.groupby("underlying_symbol", sort=False)
    daily["previous_underlying_price"] = group["underlying_price"].shift(1)
    daily["underlying_return"] = daily["underlying_price"] / daily["previous_underlying_price"] - 1.0
    split_set = {(symbol.upper(), pandas.Timestamp(value).normalize()) for symbol, value in split_dates}
    daily["crosses_split"] = [
        (str(symbol).upper(), date.normalize()) in split_set
        for symbol, date in zip(daily["underlying_symbol"], daily["quote_date"])
    ]
    daily.loc[daily["crosses_split"], "underlying_return"] = float("nan")
    daily["absolute_underlying_return"] = daily["underlying_return"].abs()
    return daily.rename(columns={"underlying_symbol": "underlying"})
