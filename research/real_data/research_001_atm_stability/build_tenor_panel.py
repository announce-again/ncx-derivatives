from __future__ import annotations

from collections.abc import Iterable

from research.real_data.common._optional import require_pandas
from research.real_data.common.calendar import add_previous_observation_date


def build_nearest_tenor_panel(
    expiry_panel,
    *,
    target_tenors: tuple[int, ...] = (21, 45, 90, 150),
    tenor_tolerances: tuple[int, ...] = (7, 10, 15, 25),
    minimum_dte: int = 7,
    maximum_dte: int = 180,
    minimum_smile_points: int = 5,
):
    """Select the closest eligible expiry, breaking ties toward shorter DTE."""

    pandas = require_pandas()
    if len(target_tenors) != len(tenor_tolerances):
        raise ValueError("target_tenors and tenor_tolerances must have equal length")
    required = {
        "underlying", "quote_date", "expiration", "actual_dte", "atm_mid_iv",
        "atm_total_variance", "atm_mid_status", "selected_point_count",
    }
    missing = sorted(required.difference(expiry_panel.columns))
    if missing:
        raise ValueError(f"expiry panel is missing columns: {missing}")
    data = expiry_panel.copy()
    data["quote_date"] = pandas.to_datetime(data["quote_date"], errors="coerce")
    data["expiration"] = pandas.to_datetime(data["expiration"], errors="coerce")
    eligible = (
        data["actual_dte"].between(minimum_dte, maximum_dte)
        & data["atm_mid_iv"].notna()
        & data["atm_mid_status"].eq("SUCCESS")
        & data["selected_point_count"].ge(minimum_smile_points)
    )
    data = data.loc[eligible].copy()
    selected = []
    for target, tolerance in zip(target_tenors, tenor_tolerances):
        candidates = data.copy()
        candidates["target_tenor"] = target
        candidates["tenor_mismatch"] = candidates["actual_dte"] - target
        candidates["absolute_tenor_mismatch"] = candidates["tenor_mismatch"].abs()
        candidates = candidates.loc[candidates["absolute_tenor_mismatch"] <= tolerance]
        candidates = candidates.sort_values(
            ["underlying", "quote_date", "absolute_tenor_mismatch", "actual_dte", "expiration"],
            kind="mergesort",
        )
        candidates = candidates.drop_duplicates(["underlying", "quote_date", "target_tenor"], keep="first")
        selected.append(candidates)
    if not selected:
        return pandas.DataFrame()
    panel = pandas.concat(selected, ignore_index=True).sort_values(
        ["underlying", "target_tenor", "quote_date"], kind="mergesort", ignore_index=True
    )
    if {"atm_ask_iv", "atm_bid_iv"}.issubset(panel.columns):
        panel["atm_iv_spread"] = panel["atm_ask_iv"] - panel["atm_bid_iv"]
        panel["relative_atm_iv_spread"] = panel["atm_iv_spread"] / panel["atm_mid_iv"]
    return add_daily_stability(panel, group_columns=("underlying", "target_tenor"))


def add_daily_stability(
    panel,
    *,
    group_columns: tuple[str, ...],
    split_dates: Iterable[tuple[str, object]] = (),
):
    pandas = require_pandas()
    data = add_previous_observation_date(panel, group_columns=group_columns)
    grouped = data.groupby(list(group_columns), sort=False, dropna=False)
    data["previous_atm_mid_iv"] = grouped["atm_mid_iv"].shift(1)
    data["delta_atm_iv"] = data["atm_mid_iv"] - data["previous_atm_mid_iv"]
    data["absolute_atm_iv_change"] = data["delta_atm_iv"].abs()
    data["squared_atm_iv_change"] = data["delta_atm_iv"] ** 2
    data["relative_atm_iv_change"] = data["delta_atm_iv"] / data["previous_atm_mid_iv"]
    data["previous_atm_total_variance"] = grouped["atm_total_variance"].shift(1)
    data["delta_atm_total_variance"] = data["atm_total_variance"] - data["previous_atm_total_variance"]
    if "atm_iv_spread" in data:
        data["previous_atm_iv_spread"] = grouped["atm_iv_spread"].shift(1)
        average_spread = 0.5 * (data["atm_iv_spread"] + data["previous_atm_iv_spread"])
        data["noise_adjusted_move"] = data["absolute_atm_iv_change"] / average_spread.where(average_spread > 0)
    invalid = ~data["is_consecutive_observation"]
    split_set = {(symbol.upper(), pandas.Timestamp(value).normalize()) for symbol, value in split_dates}
    data["crosses_split"] = [
        (str(symbol).upper(), pandas.Timestamp(current).normalize()) in split_set
        for symbol, current in zip(data["underlying"], data["quote_date"])
    ]
    invalid |= data["crosses_split"]
    change_columns = (
        "delta_atm_iv", "absolute_atm_iv_change", "squared_atm_iv_change",
        "relative_atm_iv_change", "delta_atm_total_variance", "noise_adjusted_move",
    )
    for column in change_columns:
        if column in data:
            data.loc[invalid, column] = float("nan")
    return data.sort_values([*group_columns, "quote_date"], kind="mergesort", ignore_index=True)


def adjacent_expiry_gaps(expiry_panel):
    data = expiry_panel.copy().sort_values(
        ["underlying", "quote_date", "actual_dte", "expiration"], kind="mergesort"
    )
    group = data.groupby(["underlying", "quote_date"], sort=False, dropna=False)
    data["next_expiration"] = group["expiration"].shift(-1)
    data["next_actual_dte"] = group["actual_dte"].shift(-1)
    data["adjacent_expiry_gap"] = (group["atm_mid_iv"].shift(-1) - data["atm_mid_iv"]).abs()
    data["adjacent_variance_gap"] = (group["atm_total_variance"].shift(-1) - data["atm_total_variance"]).abs()
    return data
