"""Enrich canonical quotes with rates, forwards, and research fields."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import VALUATION_DATE, carry_assumptions, sample_snapshot  # noqa: E402

from ncx_derivatives.market_data import enrich_option_chain_snapshot  # noqa: E402


def _fmt(value: float | None) -> str:
    if value is None:
        return "None"
    return f"{value:.4f}"


def main() -> None:
    snapshot = sample_snapshot()
    enriched = enrich_option_chain_snapshot(
        snapshot=snapshot,
        carry=carry_assumptions(),
        valuation_date=VALUATION_DATE,
    )

    print(f"Underlying: {snapshot.underlying_symbol}")
    print(
        "type  strike  midpoint  spread       T       df       forward  "
        "S/K     F/K     intrinsic  time_value  bounds"
    )
    for item in enriched:
        contract = item.quote.contract
        bounds = item.no_arbitrage_bounds
        bounds_text = (
            "None"
            if bounds is None
            else f"[{bounds.lower_bound:.4f}, {bounds.upper_bound:.4f}]"
        )
        print(
            f"{contract.option_type.value:4} "
            f"{contract.strike:7.2f} "
            f"{_fmt(item.midpoint):>8} "
            f"{_fmt(item.relative_spread):>8} "
            f"{item.time_to_maturity:7.4f} "
            f"{item.risk_free_discount_factor:8.4f} "
            f"{item.forward_price:8.4f} "
            f"{item.spot_moneyness:7.4f} "
            f"{item.forward_moneyness:7.4f} "
            f"{item.intrinsic_value:9.4f} "
            f"{_fmt(item.time_value):>10} "
            f"{bounds_text}"
        )


if __name__ == "__main__":
    main()
