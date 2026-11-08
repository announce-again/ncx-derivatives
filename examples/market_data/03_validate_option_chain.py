"""Report validation issues without removing any data."""

import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import VALUATION_TIMESTAMP, OptionType, manual_snapshot, quote  # noqa: E402

from ncx_derivatives.market_data import (  # noqa: E402
    OptionChainSnapshot,
    validate_option_chain_snapshot,
)


def main() -> None:
    base = manual_snapshot()
    good = quote(OptionType.CALL, 90.0, 11.90, 12.10)
    crossed = quote(OptionType.CALL, 95.0, 9.20, 9.00)
    future = quote(
        OptionType.PUT,
        95.0,
        3.30,
        3.50,
        timestamp=VALUATION_TIMESTAMP + timedelta(minutes=1),
    )
    duplicate = quote(OptionType.CALL, 90.0, 11.80, 12.00)
    snapshot = OptionChainSnapshot(
        underlying_symbol=base.underlying_symbol,
        as_of=base.as_of,
        quotes=(good, crossed, future, duplicate),
        underlying_quote=base.underlying_quote,
        metadata=base.metadata,
    )

    report = validate_option_chain_snapshot(snapshot)

    print(f"Quotes inspected: {len(snapshot.quotes)}")
    print(f"Issues reported: {len(report.issues)}")
    print("Validation reports issues; it does not delete quotes.")
    print()

    for issue in report.issues:
        location = "/".join(issue.location)
        print(f"[{issue.severity.value}] {issue.code} {location}")
        print(f"  {issue.message}")
        if issue.context:
            print(f"  context={dict(issue.context)}")


if __name__ == "__main__":
    main()
