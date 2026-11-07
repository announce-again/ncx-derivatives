import pandas as pd
import pytest

from research.real_data.research_001_atm_stability.compare_constant_maturity import (
    _paired_changes,
    build_constant_maturity_panel,
)


def test_constant_maturity_does_not_bridge_missing_sessions_or_confuse_rollovers():
    dates = pd.date_range("2020-01-06", periods=4)
    front, back = pd.Timestamp("2020-01-27"), pd.Timestamp("2020-01-31")
    sessions = pd.DataFrame(
        {"underlying": "SPY", "quote_date": dates, "crosses_split": False}
    )
    expiry = pd.DataFrame(
        [
            ("SPY", dates[0], front, 21, 0.30),
            ("SPY", dates[2], front, 19, 0.32),
            ("SPY", dates[2], back, 23, 0.27),
            ("SPY", dates[3], front, 18, 0.31),
            ("SPY", dates[3], back, 22, 0.29),
        ],
        columns=["underlying", "quote_date", "expiration", "actual_dte", "atm_mid_iv"],
    )
    expiry["atm_total_variance"] = (
        expiry["atm_mid_iv"] ** 2 * expiry["actual_dte"] / 365.0
    )
    nearest = expiry.iloc[[0, 1, 4]].assign(target_tenor=21)

    constant = build_constant_maturity_panel(expiry, sessions, [21])
    assert constant["interpolation_status"].tolist() == [
        "EXACT",
        "NO_ELIGIBLE_EXPIRIES",
        "INTERPOLATED",
        "INTERPOLATED",
    ]
    assert constant.loc[2, "upper_weight"] == pytest.approx(0.5)
    assert constant.loc[2, "atm_total_variance"] == pytest.approx(
        (0.32**2 * 19 + 0.27**2 * 23) / 730
    )
    assert constant.loc[0, "atm_mid_iv"] == pytest.approx(0.30)

    paired = _paired_changes(sessions, expiry, nearest, constant)
    assert pd.isna(paired.loc[2, "constant_maturity_change"])
    assert paired.loc[2, "previous_date"] == dates[1]
    assert paired.loc[3, "nearest_rollover"]
    assert paired.loc[3, "fixed_expiry_change"] == pytest.approx(0.02)
    assert paired.loc[3, "nearest_tenor_change"] == pytest.approx(-0.03)
    assert paired.loc[3, "roll_component"] == pytest.approx(-0.05)
