from __future__ import annotations

from collections.abc import Iterable

from research.real_data.common._optional import require_pandas


def add_previous_observation_date(
    frame,
    *,
    group_columns: tuple[str, ...],
    date_column: str = "quote_date",
    maximum_calendar_gap: int = 7,
):
    """Mark adjacent observations without pretending to be an exchange calendar."""

    pandas = require_pandas()
    data = frame.copy()
    data[date_column] = pandas.to_datetime(data[date_column], errors="coerce")
    order = [*group_columns, date_column]
    data = data.sort_values(order, kind="mergesort").reset_index(drop=True)
    grouped = data.groupby(list(group_columns), sort=False, dropna=False)
    data["previous_observation_date"] = grouped[date_column].shift(1)
    data["observation_gap_days"] = (
        data[date_column] - data["previous_observation_date"]
    ).dt.days
    data["is_consecutive_observation"] = data["observation_gap_days"].between(
        1, maximum_calendar_gap
    )
    return data


def missing_business_date_ranges(dates: Iterable[object]):
    """Return weekday gaps; this is a diagnostic, not an exchange calendar."""

    pandas = require_pandas()
    observed = pandas.DatetimeIndex(pandas.to_datetime(list(dates))).normalize().unique().sort_values()
    if len(observed) < 2:
        return ()
    expected = pandas.bdate_range(observed.min(), observed.max())
    missing = expected.difference(observed)
    return tuple(value.date() for value in missing)
