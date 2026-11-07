"""Compare Research 001A ATM designs on paired SPY sessions and carry inputs."""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path

import pandas

from ncx_derivatives.volatility import interpolate_total_variance
from research.real_data.common.deterministic_io import (
    file_sha256,
    write_csv,
    write_json,
    write_parquet,
)

_SESSION = ["underlying", "quote_date"]
_KEY = [*_SESSION, "target_tenor"]
_EXPIRY = [*_SESSION, "expiration"]
_METHODS = ("fixed_expiry", "nearest_tenor", "constant_maturity")


def build_constant_maturity_panel(expiry, sessions, target_tenors):
    """Query ACT/365F ATM total variance for every session and target tenor."""
    nodes = {
        key: tuple(zip(group["actual_dte"] / 365.0, group["atm_total_variance"]))
        for key, group in expiry.groupby(_SESSION, sort=False)
    }
    records = []
    for underlying, quote_date in sessions[_SESSION].itertuples(index=False, name=None):
        curve = nodes.get((underlying, quote_date), ())
        for tenor in target_tenors:
            maturity = tenor / 365.0
            result = interpolate_total_variance(curve, maturity)
            variance = result.total_variance
            records.append(
                {
                    "underlying": underlying,
                    "quote_date": quote_date,
                    "target_tenor": tenor,
                    "atm_mid_iv": None
                    if variance is None
                    else sqrt(variance / maturity),
                    "atm_total_variance": variance,
                    "lower_dte": None
                    if result.lower_maturity is None
                    else round(result.lower_maturity * 365),
                    "upper_dte": None
                    if result.upper_maturity is None
                    else round(result.upper_maturity * 365),
                    "upper_weight": result.upper_weight,
                    "interpolation_status": (
                        "NO_ELIGIBLE_EXPIRIES"
                        if not curve
                        else result.failure_reason.value
                        if result.failure_reason
                        else "EXACT"
                        if result.lower_maturity == result.upper_maturity
                        else "INTERPOLATED"
                    ),
                }
            )
    return pandas.DataFrame.from_records(records)


def compare(baseline: Path, alternative: Path, output: Path) -> None:
    roots = {"baseline": baseline, "trailing_dividend": alternative}
    config = json.loads((baseline / "config.json").read_text(encoding="utf-8"))
    if config != json.loads((alternative / "config.json").read_text(encoding="utf-8")):
        raise ValueError(
            "carry specifications must use the same research configuration"
        )
    targets = config["target_tenors"]
    comparisons = {}
    inputs = {}
    for specification, root in roots.items():
        paths = [
            root / name
            for name in (
                "config.json",
                "atm_expiry_panel.parquet",
                "atm_tenor_panel.parquet",
                "underlying_return_panel.parquet",
            )
        ]
        inputs[specification] = {path.name: file_sha256(path) for path in paths}
        expiry = _read_panel(paths[1], _EXPIRY)
        nearest = _read_panel(paths[2], _KEY)
        sessions = _read_panel(paths[3], _SESSION)[_SESSION + ["crosses_split"]]
        if specification == "baseline":
            baseline_sessions = sessions
        elif not sessions.equals(baseline_sessions):
            raise ValueError(
                "carry specifications must share the same observation calendar"
            )
        eligible = expiry.loc[
            expiry["actual_dte"].between(config["minimum_dte"], config["maximum_dte"])
            & expiry["atm_mid_status"].eq("SUCCESS")
            & expiry["selected_point_count"].ge(config["minimum_smile_points"])
            & expiry["atm_mid_iv"].notna()
            & expiry["atm_total_variance"].notna()
        ]
        constant = build_constant_maturity_panel(eligible, sessions, targets)
        comparisons[specification] = _paired_changes(
            sessions, eligible, nearest, constant
        )

    complete = {
        name: frame.dropna(subset=[f"{method}_change" for method in _METHODS])
        for name, frame in comparisons.items()
    }
    common = complete["baseline"][_KEY].merge(
        complete["trailing_dividend"][_KEY], on=_KEY
    )
    balanced_dates = common.groupby(_SESSION)["target_tenor"].nunique().eq(len(targets))
    balanced = common.merge(
        balanced_dates[balanced_dates].reset_index()[_SESSION], on=_SESSION
    )

    summary = []
    rollovers = []
    for name, frame in comparisons.items():
        for target in targets:
            group = frame.loc[frame["target_tenor"].eq(target)]
            paired = group.merge(common, on=_KEY)
            aligned = group.merge(balanced, on=_KEY)
            summary.append(_summarize(name, target, group, paired, aligned))
            for rollover, subset in paired.groupby("nearest_rollover", sort=True):
                rollovers.append(
                    {
                        "specification": name,
                        "target_tenor": target,
                        "nearest_rollover": rollover,
                        "observations": len(subset),
                        **_median_changes(subset),
                        "median_abs_roll_component_vp": 100
                        * subset["roll_component"].abs().median(),
                    }
                )
    output.mkdir(parents=True, exist_ok=True)
    panel = pandas.concat(
        [frame.assign(specification=name) for name, frame in comparisons.items()]
    )
    written = [
        write_parquet(
            output / "paired_observations.parquet",
            panel,
            sort_by=("specification", *_KEY),
        ),
        write_csv(output / "design_comparison.csv", summary, columns=tuple(summary[0])),
        write_csv(
            output / "rollover_comparison.csv",
            rollovers,
            columns=(
                "specification",
                "target_tenor",
                "nearest_rollover",
                "observations",
                *(f"{method}_median_abs_change_vp" for method in _METHODS),
                "median_abs_roll_component_vp",
            ),
        ),
    ]
    write_json(
        output / "manifest.json",
        {
            "config_sha256": config["config_sha256"],
            "input_sha256": inputs,
            "output_sha256": {path.name: file_sha256(path) for path in written},
        },
    )


def _read_panel(path, key):
    frame = pandas.read_parquet(path)
    for column in ("quote_date", "expiration"):
        if column in frame:
            frame[column] = pandas.to_datetime(
                frame[column], errors="raise"
            ).dt.normalize()
    if frame[key].isna().any().any() or frame.duplicated(key).any():
        raise ValueError(f"{path} must have unique, non-missing keys {key}")
    return frame.sort_values(key, ignore_index=True)


def _paired_changes(sessions, expiry, nearest, constant):
    sessions = sessions.copy()
    sessions["previous_date"] = sessions.groupby("underlying")["quote_date"].shift()
    frame = constant.rename(columns={"atm_mid_iv": "constant_maturity_iv"}).merge(
        sessions, on=_SESSION
    )
    selected = nearest[_KEY + ["expiration", "atm_mid_iv"]].rename(
        columns={"atm_mid_iv": "nearest_tenor_iv"}
    )
    frame = frame.merge(selected, on=_KEY, how="left", validate="one_to_one")
    frame = frame.merge(
        selected.rename(
            columns={
                "quote_date": "previous_date",
                "expiration": "previous_expiration",
                "nearest_tenor_iv": "previous_nearest_tenor_iv",
            }
        ),
        on=["underlying", "previous_date", "target_tenor"],
        how="left",
        validate="one_to_one",
    )
    frame = frame.merge(
        expiry[_EXPIRY + ["atm_mid_iv"]].rename(
            columns={
                "quote_date": "previous_date",
                "atm_mid_iv": "previous_fixed_expiry_iv",
            }
        ),
        on=["underlying", "previous_date", "expiration"],
        how="left",
        validate="many_to_one",
    )
    frame = frame.merge(
        constant[_KEY + ["atm_mid_iv"]].rename(
            columns={
                "quote_date": "previous_date",
                "atm_mid_iv": "previous_constant_maturity_iv",
            }
        ),
        on=["underlying", "previous_date", "target_tenor"],
        how="left",
        validate="one_to_one",
    )
    comparable = ~frame["crosses_split"] & frame["previous_date"].notna()
    for method in _METHODS:
        current = "nearest_tenor_iv" if method == "fixed_expiry" else f"{method}_iv"
        frame[f"{method}_change"] = (
            frame[current] - frame[f"previous_{method}_iv"]
        ).where(comparable)
    has_both = frame["expiration"].notna() & frame["previous_expiration"].notna()
    frame["nearest_rollover"] = (
        frame["expiration"]
        .ne(frame["previous_expiration"])
        .where(has_both)
        .astype("boolean")
    )
    frame["roll_component"] = (
        frame["nearest_tenor_change"] - frame["fixed_expiry_change"]
    )
    return frame


def _median_changes(frame, *, prefix=""):
    return {
        f"{prefix}{method}_median_abs_change_vp": 100
        * frame[f"{method}_change"].abs().median()
        for method in _METHODS
    }


def _summarize(name, target, group, paired, balanced):
    valid = group["constant_maturity_iv"].notna()
    return {
        "specification": name,
        "target_tenor": target,
        "quote_sessions": len(group),
        "nearest_level_observations": int(group["nearest_tenor_iv"].count()),
        "constant_level_observations": int(valid.sum()),
        "constant_coverage": valid.mean(),
        "constant_exact": int(group["interpolation_status"].eq("EXACT").sum()),
        "constant_median_bracket_days": (group["upper_dte"] - group["lower_dte"])
        .where(valid)
        .median(),
        "common_pair_observations": len(paired),
        **_median_changes(paired),
        "balanced_pair_observations": len(balanced),
        **_median_changes(balanced, prefix="balanced_"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--trailing-dividend", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    compare(args.baseline, args.trailing_dividend, args.output_dir)


if __name__ == "__main__":
    main()
