"""Apply configurable cleaning and inspect accepted/rejected partitions."""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    VALUATION_DATE,
    carry_assumptions,
    cleaning_config,
    sample_snapshot,
)

from ncx_derivatives.market_data import (  # noqa: E402
    clean_enriched_option_quotes,
    enrich_option_chain_snapshot,
)


def main() -> None:
    snapshot = sample_snapshot()
    enriched = enrich_option_chain_snapshot(
        snapshot=snapshot,
        carry=carry_assumptions(),
        valuation_date=VALUATION_DATE,
    )
    result = clean_enriched_option_quotes(enriched, cleaning_config())

    print(f"Accepted: {result.accepted_count}")
    print(f"Rejected: {result.rejected_count}")
    print()

    print("Rejected quotes:")
    for rejected in result.rejected:
        contract = rejected.quote.contract
        print(f"K={contract.strike:g} {contract.option_type.value}")
        for diagnostic in rejected.diagnostics:
            print(f"  - {diagnostic.reason.value}: {diagnostic.message}")

    reason_counts = Counter(diagnostic.reason.value for diagnostic in result.diagnostics)
    print()
    print("Diagnostics by reason:")
    for reason, count in sorted(reason_counts.items()):
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
