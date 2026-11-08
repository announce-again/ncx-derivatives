"""Detect static-arbitrage issues without repairing prices."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import VALUATION_DATE, OptionType, carry_assumptions, quote  # noqa: E402

from ncx_derivatives.market_data import (  # noqa: E402
    diagnose_static_arbitrage,
    enrich_option_quote,
)


def main() -> None:
    quotes = (
        quote(OptionType.CALL, 90.0, 11.90, 12.10),
        quote(OptionType.CALL, 100.0, 13.90, 14.10),
        quote(OptionType.CALL, 110.0, 8.90, 9.10),
        quote(OptionType.PUT, 100.0, 5.90, 6.10),
    )
    carry = carry_assumptions()
    enriched = tuple(
        enrich_option_quote(
            quote=item,
            valuation_timestamp=item.quote_timestamp,
            valuation_date=VALUATION_DATE,
            spot=100.0,
            carry=carry,
        )
        for item in quotes
    )
    report = diagnose_static_arbitrage(enriched)

    for diagnostic in report.diagnostics:
        location = "/".join(diagnostic.location)
        print(
            f"{diagnostic.code.value} "
            f"violation={diagnostic.violation_amount:.6f} "
            f"location={location}"
        )
        print(f"  context={dict(diagnostic.context)}")

    assert enriched[1].midpoint == 14.0
    print()
    print("Original enriched midpoint remains unchanged:", enriched[1].midpoint)


if __name__ == "__main__":
    main()
