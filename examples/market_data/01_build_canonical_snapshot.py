"""Build canonical market-data objects without CSV ingestion."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import manual_snapshot  # noqa: E402


def main() -> None:
    snapshot = manual_snapshot()

    print(f"Snapshot symbol: {snapshot.underlying_symbol}")
    print(f"Quote count: {len(snapshot.quotes)}")
    print()

    print("Pairing keys:")
    for quote in snapshot.quotes:
        contract = quote.contract
        print(
            f"  {contract.option_type.value:4} "
            f"K={contract.strike:6.2f} -> {contract.pairing_key}"
        )

    print()
    print("Deterministic sorted contracts:")
    for contract in snapshot.contracts:
        print(
            f"  {contract.expiration.isoformat()} "
            f"{contract.option_type.value:4} "
            f"K={contract.strike:6.2f}"
        )


if __name__ == "__main__":
    main()
