"""Reproduce Stage 3.6 from the frozen Research 001A SPY expiry and return panels."""

from __future__ import annotations

import argparse
import json
from math import isfinite, log, sqrt
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from research.real_data.common.deterministic_io import (
    file_sha256,
    write_csv,
    write_json,
    write_parquet,
)
from research.real_data.research_001_atm_stability.compare_constant_maturity import (
    build_constant_maturity_panel,
)

_SESSION = ["underlying", "quote_date"]
_KEY = [*_SESSION, "target_tenor"]
_EXPIRY = [*_SESSION, "expiration"]
_NORMAL = NormalDist()
_EVENTS = (
    ("2020-03-03", "FOMC March 3 emergency decision"),
    ("2020-03-15", "FOMC March 15 emergency decision"),
    ("2022-06-15", "FOMC June 2022 decision"),
)


def realized_variance(sessions: pd.DataFrame, tenors: tuple[int, ...]) -> pd.DataFrame:
    """Match each target horizon to subsequent close-to-close log returns."""
    results = []
    for symbol, group in sessions.groupby("underlying", sort=True):
        group = group.sort_values("quote_date")
        dates = group["quote_date"].to_numpy(dtype="datetime64[D]")
        if len(dates) != len(np.unique(dates)) or np.isnat(dates).any():
            raise ValueError("session dates must be unique and present")
        prices = group["underlying_price"].to_numpy(dtype=float)
        logged = np.log(np.where((prices > 0) & np.isfinite(prices), prices, np.nan))
        returns = np.diff(logged, prepend=np.nan)
        long_gaps = np.diff(dates) > np.timedelta64(7, "D")
        invalid = ~np.isfinite(returns) | group["crosses_split"].to_numpy(dtype=bool)
        invalid[1:] |= long_gaps
        gap_after_session = np.append(long_gaps, False)
        returns[invalid] = np.nan
        cumulative = np.cumsum(np.where(invalid, 0.0, returns**2))
        missing = np.cumsum(invalid)
        indexes = np.arange(len(dates))
        for tenor in tenors:
            if isinstance(tenor, bool) or not isinstance(tenor, int) or tenor <= 0:
                raise ValueError("tenors must be positive integers")
            horizons = dates + np.timedelta64(tenor, "D")
            end = np.searchsorted(dates, horizons, side="right") - 1
            horizon_in_gap = (horizons > dates[end]) & gap_after_session[end]
            valid = (
                (end > indexes)
                & (horizons <= dates[-1])
                & (missing[end] == missing)
                & ~horizon_in_gap
            )
            annualized = np.where(
                valid, (cumulative[end] - cumulative) * 365 / tenor, np.nan
            )
            results.append(
                pd.DataFrame(
                    {
                        "underlying": symbol,
                        "quote_date": pd.to_datetime(dates),
                        "target_tenor": tenor,
                        "log_spot_return": returns,
                        "realized_end_date": pd.to_datetime(dates[end]).where(valid),
                        "realized_return_count": np.where(valid, end - indexes, np.nan),
                        "forward_realized_variance": annualized,
                    }
                )
            )
    return pd.concat(results, ignore_index=True).sort_values(_KEY, ignore_index=True)


def term_dynamics(constant: pd.DataFrame, realized: pd.DataFrame) -> pd.DataFrame:
    data = constant.merge(realized, on=_KEY, validate="one_to_one").sort_values(_KEY)
    front = data.loc[
        data["target_tenor"].eq(data["target_tenor"].min()), _SESSION + ["atm_mid_iv"]
    ].copy()
    grouped = front.groupby("underlying", sort=False)["atm_mid_iv"]
    for name, quantile in (("low", 0.25), ("high", 0.75)):
        front[name] = grouped.transform(
            lambda values: (
                values.shift().rolling(252, min_periods=126).quantile(quantile)
            )
        )
    front["regime"] = np.select(
        [
            front[["atm_mid_iv", "low", "high"]].isna().any(axis=1),
            front["atm_mid_iv"].le(front["low"]),
            front["atm_mid_iv"].ge(front["high"]),
        ],
        ["unclassified", "low", "high"],
        default="middle",
    )
    data = data.merge(front[_SESSION + ["regime"]], on=_SESSION, validate="many_to_one")
    data["implied_variance"] = data["atm_mid_iv"] ** 2
    data["realized_volatility"] = np.sqrt(data["forward_realized_variance"])
    data["variance_risk_premium_proxy"] = (
        data["implied_variance"] - data["forward_realized_variance"]
    )
    by_session = data.groupby(_SESSION, sort=False)
    previous_tenor = by_session["target_tenor"].shift()
    data["previous_target_tenor"] = previous_tenor
    data["implied_forward_variance"] = (
        (data["atm_total_variance"] - by_session["atm_total_variance"].shift())
        * 365
        / (data["target_tenor"] - previous_tenor)
    )
    data = data.sort_values(
        ["underlying", "target_tenor", "quote_date"], ignore_index=True
    )
    by_tenor = data.groupby(["underlying", "target_tenor"], sort=False)
    data["previous_quote_date"] = by_tenor["quote_date"].shift()
    data["previous_atm_iv"] = by_tenor["atm_mid_iv"].shift()
    for column in ("atm_mid_iv", "implied_forward_variance"):
        data[f"change_{column}"] = (
            by_tenor[column].diff().where(data["log_spot_return"].notna())
        )
    return data


def smile_dynamics(expiry: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    """Compare same-expiry smile changes and two prior-smile ATM predictions."""
    calendar = sessions[_SESSION].sort_values(_SESSION).copy()
    calendar["previous_date"] = calendar.groupby("underlying")["quote_date"].shift()
    returns = realized_variance(sessions, (1,))[_SESSION + ["log_spot_return"]]
    columns = [
        "actual_dte",
        "forward",
        "dividend_discount_factor",
        "atm_mid_iv",
        "call_25d_iv",
        "put_25d_iv",
        "rr25",
        "bf25",
        "local_skew",
    ]
    current = (
        expiry[_EXPIRY + columns]
        .merge(calendar, on=_SESSION)
        .merge(returns, on=_SESSION)
    )
    previous = expiry[_EXPIRY + columns].rename(
        columns={
            "quote_date": "previous_date",
            **{name: f"previous_{name}" for name in columns},
        }
    )
    paired = current.merge(
        previous,
        on=["underlying", "previous_date", "expiration"],
        validate="many_to_one",
    )
    paired = paired.loc[paired["log_spot_return"].notna()].copy()
    for name in ("atm_mid_iv", "rr25", "bf25", "local_skew"):
        paired[f"change_{name}"] = paired[name] - paired[f"previous_{name}"]
    paired["dte_band"] = pd.cut(
        paired["actual_dte"],
        [0, 30, 60, 120, 180],
        labels=["01-30", "31-60", "61-120", "121-180"],
    )
    predictions = [_sticky_predictions(row) for row in paired.itertuples(index=False)]
    paired[["sticky_strike_iv", "sticky_delta_iv"]] = pd.DataFrame(
        predictions, index=paired.index, columns=[0, 1]
    )
    for method in ("sticky_strike", "sticky_delta"):
        paired[f"{method}_error"] = paired[f"{method}_iv"] - paired["atm_mid_iv"]
    return paired


def _sticky_predictions(row):
    old_atm, old_put, old_call = (
        row.previous_atm_mid_iv,
        row.previous_put_25d_iv,
        row.previous_call_25d_iv,
    )
    old_t, t = row.previous_actual_dte / 365, row.actual_dte / 365
    old_forward, new_forward = row.previous_forward, row.forward
    discount = row.previous_dividend_discount_factor
    if not all(
        isfinite(x) and x > 0
        for x in (
            old_atm,
            old_put,
            old_call,
            old_t,
            t,
            old_forward,
            new_forward,
            discount,
        )
    ):
        return None, None
    call_delta = 0.25 / discount
    if not 0 < call_delta < 0.5:
        return None, None
    put_delta = 1 - call_delta
    atm_delta = _NORMAL.cdf(old_atm * sqrt(old_t) / 2)
    if not call_delta < atm_delta < put_delta:
        return None, None

    put_w, atm_w, call_w = (iv**2 * old_t for iv in (old_put, old_atm, old_call))
    put_k = put_w / 2 - sqrt(put_w) * _NORMAL.inv_cdf(put_delta)
    call_k = call_w / 2 - sqrt(call_w) * _NORMAL.inv_cdf(call_delta)
    strike_iv = None
    if put_k < 0 < call_k:
        query = log(new_forward / old_forward)
        if put_k <= query <= call_k:
            variance = np.interp(query, (put_k, 0.0, call_k), (put_w, atm_w, call_w))
            strike_iv = sqrt(variance / old_t)

    # The prior smile is linear in IV versus normalized call-equivalent delta.
    # At today's ATM-forward strike its delta depends on the forecast itself.
    scale = sqrt(t) / 2
    low, high = min(old_put, old_atm, old_call), max(old_put, old_atm, old_call)
    high = min(high, _NORMAL.inv_cdf(put_delta) / scale)
    if low > high:
        return strike_iv, None
    deltas = (call_delta, atm_delta, put_delta)
    vols = (old_call, old_atm, old_put)
    slopes = [
        (y2 - y1) / (x2 - x1)
        for (x1, x2), (y1, y2) in zip(zip(deltas, deltas[1:]), zip(vols, vols[1:]))
    ]
    if max(slopes) * scale / sqrt(2 * np.pi) >= 1:
        return strike_iv, None  # Monotonicity of the root residual is unverified.

    def residual(iv):
        return iv - np.interp(_NORMAL.cdf(iv * scale), deltas, vols)

    if residual(low) > 1e-12 or residual(high) < -1e-12:
        return strike_iv, None
    for _ in range(32):
        middle = (low + high) / 2
        if residual(middle) < 0:
            low = middle
        else:
            high = middle
    return strike_iv, (low + high) / 2


def event_study(
    term: pd.DataFrame, events: tuple[tuple[str, str], ...]
) -> pd.DataFrame:
    records = []
    for symbol, group in term.groupby("underlying", sort=True):
        for day, name in events:
            date = pd.Timestamp(day)
            eligible = group.loc[
                group["quote_date"].between(date, date + pd.Timedelta(days=7))
            ]
            if eligible.empty:
                continue
            session = eligible["quote_date"].min()
            for row in eligible.loc[eligible["quote_date"].eq(session)].itertuples():
                elapsed = (session - row.previous_quote_date).days
                records.append(
                    {
                        "underlying": symbol,
                        "event_date": date,
                        "event": name,
                        "quote_date": session,
                        "target_tenor": row.target_tenor,
                        "log_spot_return": row.log_spot_return,
                        "observed_squared_log_return": row.log_spot_return**2,
                        "pre_event_window_variance_proxy": row.previous_atm_iv**2
                        * elapsed
                        / 365,
                        "atm_mid_iv": row.atm_mid_iv,
                        "change_atm_mid_iv": row.change_atm_mid_iv,
                        "implied_forward_variance": row.implied_forward_variance,
                        "change_implied_forward_variance": row.change_implied_forward_variance,
                    }
                )
    return pd.DataFrame.from_records(
        records,
        columns=(
            "underlying",
            "event_date",
            "event",
            "quote_date",
            "target_tenor",
            "log_spot_return",
            "observed_squared_log_return",
            "pre_event_window_variance_proxy",
            "atm_mid_iv",
            "change_atm_mid_iv",
            "implied_forward_variance",
            "change_implied_forward_variance",
        ),
    )


def _summaries(term, smile):
    term_rows = []
    for (symbol, tenor, regime), group in term.groupby(
        ["underlying", "target_tenor", "regime"], sort=True
    ):
        matched = group.dropna(subset=["variance_risk_premium_proxy"])
        term_rows.append(
            {
                "underlying": symbol,
                "target_tenor": tenor,
                "regime": regime,
                "matched_observations": len(matched),
                "median_implied_volatility": matched["atm_mid_iv"].median(),
                "median_forward_realized_volatility": matched[
                    "realized_volatility"
                ].median(),
                "mean_variance_risk_premium_proxy": matched[
                    "variance_risk_premium_proxy"
                ].mean(),
                "median_variance_risk_premium_proxy": matched[
                    "variance_risk_premium_proxy"
                ].median(),
                "median_implied_forward_variance": group[
                    "implied_forward_variance"
                ].median(),
                "median_abs_change_atm_iv": group["change_atm_mid_iv"].abs().median(),
            }
        )
    smile_rows = []
    for (symbol, band), group in smile.groupby(
        ["underlying", "dte_band"], sort=True, observed=True
    ):
        common = group.dropna(subset=["sticky_strike_error", "sticky_delta_error"])
        row = {
            "underlying": symbol,
            "dte_band": str(band),
            "same_expiry_observations": len(group),
            "sticky_common_observations": len(common),
            "sticky_strike_mae": common["sticky_strike_error"].abs().mean(),
            "sticky_delta_mae": common["sticky_delta_error"].abs().mean(),
        }
        for metric in ("atm_mid_iv", "rr25", "bf25", "local_skew"):
            paired = group[["log_spot_return", f"change_{metric}"]].dropna()
            variance = paired["log_spot_return"].var()
            row[f"{metric}_response_observations"] = len(paired)
            row[f"{metric}_response_slope"] = (
                paired.cov().iloc[0, 1] / variance if variance > 0 else np.nan
            )
        smile_rows.append(row)
    return term_rows, smile_rows


def run(
    root: Path, output: Path, events: tuple[tuple[str, str], ...] = _EVENTS
) -> None:
    config_path = root / "config.json"
    expiry_path, return_path = (
        root / "atm_expiry_panel.parquet",
        root / "underlying_return_panel.parquet",
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expiry, sessions = pd.read_parquet(expiry_path), pd.read_parquet(return_path)
    for frame in (expiry, sessions):
        frame["quote_date"] = pd.to_datetime(frame["quote_date"]).dt.normalize()
    expiry["expiration"] = pd.to_datetime(expiry["expiration"]).dt.normalize()
    eligible = expiry.loc[
        expiry["actual_dte"].between(config["minimum_dte"], config["maximum_dte"])
        & expiry["atm_mid_status"].eq("SUCCESS")
        & expiry["selected_point_count"].ge(config["minimum_smile_points"])
        & expiry["atm_total_variance"].notna()
    ]
    tenors = tuple(config["target_tenors"])
    constant = build_constant_maturity_panel(eligible, sessions, tenors)
    term = term_dynamics(constant, realized_variance(sessions, tenors))
    smile = smile_dynamics(eligible, sessions)
    event = event_study(term, events)
    term_rows, smile_rows = _summaries(term, smile)
    paths = (
        write_parquet(output / "term_dynamics.parquet", term, sort_by=_KEY),
        write_parquet(output / "smile_dynamics.parquet", smile, sort_by=_EXPIRY),
        write_csv(output / "term_regime_summary.csv", term_rows),
        write_csv(output / "smile_response_summary.csv", smile_rows),
        write_csv(
            output / "events.csv",
            event.to_dict("records"),
            columns=tuple(event.columns),
        ),
    )
    write_json(
        output / "manifest.json",
        {
            "events": events,
            "inputs": {
                path.name: file_sha256(path)
                for path in (config_path, expiry_path, return_path)
            },
            "outputs": {path.name: file_sha256(path) for path in paths},
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-001a", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--events-csv",
        type=Path,
        help="Override the default FOMC examples with event_date,event rows",
    )
    args = parser.parse_args()
    events = _EVENTS
    if args.events_csv is not None:
        events = tuple(
            pd.read_csv(args.events_csv)[["event_date", "event"]].itertuples(
                index=False, name=None
            )
        )
    run(args.research_001a, args.output_dir, events)


if __name__ == "__main__":
    main()
