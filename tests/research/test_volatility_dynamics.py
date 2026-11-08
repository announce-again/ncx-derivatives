from math import exp
from types import SimpleNamespace

import pandas as pd
import pytest

from research.real_data.volatility_dynamics import (
    _sticky_predictions,
    realized_variance,
)


def test_forward_realized_variance_matches_horizon_and_excludes_split_crossings():
    sessions = pd.DataFrame(
        {
            "underlying": "SPY",
            "quote_date": pd.to_datetime(
                ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
            ),
            "underlying_price": [100, 100 * exp(0.01), 100 * exp(0.03), 50],
            "crosses_split": [False, False, False, True],
        }
    )
    realized = realized_variance(sessions, (4,))
    assert realized.loc[0, "realized_return_count"] == 2
    assert realized.loc[0, "forward_realized_variance"] == pytest.approx(
        (0.01**2 + 0.02**2) * 365 / 4
    )
    assert realized.loc[0, "realized_end_date"] == pd.Timestamp("2020-01-06")
    assert pd.isna(realized.loc[1, "forward_realized_variance"])
    assert pd.isna(realized.loc[3, "log_spot_return"])


def test_forward_realized_variance_excludes_horizons_inside_long_observation_gaps():
    sessions = pd.DataFrame(
        {
            "underlying": "SPY",
            "quote_date": pd.to_datetime(
                ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-14"]
            ),
            "underlying_price": [100, 101, 103, 104],
            "crosses_split": False,
        }
    )
    realized = realized_variance(sessions, (3, 4, 12)).set_index(
        ["quote_date", "target_tenor"]
    )
    start = pd.Timestamp("2020-01-02")
    values = realized["forward_realized_variance"]
    assert values.loc[(start, 3)] > 0  # Horizon during a regular weekend.
    assert values.loc[(start, 4)] > 0  # Horizon at the last close before the gap.
    assert pd.isna(values.loc[(pd.Timestamp("2020-01-03"), 4)])
    assert pd.isna(values.loc[(start, 12)])


def test_sticky_predictions_recover_flat_smile_without_wing_extrapolation():
    row = SimpleNamespace(
        previous_atm_mid_iv=0.2,
        previous_put_25d_iv=0.2,
        previous_call_25d_iv=0.2,
        previous_actual_dte=31,
        actual_dte=30,
        previous_forward=100.0,
        forward=101.0,
        previous_dividend_discount_factor=1.0,
    )
    assert _sticky_predictions(row) == pytest.approx((0.2, 0.2))
    row.forward = 200.0
    strike, delta = _sticky_predictions(row)
    assert strike is None
    assert delta == pytest.approx(0.2)
