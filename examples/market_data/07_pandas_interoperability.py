"""Use pandas at the boundary while preserving canonical objects."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import VALUATION_DATE, carry_assumptions, sample_snapshot  # noqa: E402

from ncx_derivatives.market_data import (  # noqa: E402
    OptionChainSnapshot,
    UnderlyingQuote,
    enrich_option_chain_snapshot,
    enriched_quotes_to_dataframe,
    option_chain_from_dataframe,
    option_chain_to_dataframe,
)


def main() -> None:
    source_snapshot = sample_snapshot()
    underlying = source_snapshot.underlying_quote
    snapshot = OptionChainSnapshot(
        underlying_symbol=source_snapshot.underlying_symbol,
        as_of=source_snapshot.as_of,
        quotes=source_snapshot.quotes,
        underlying_quote=(
            None
            if underlying is None
            else UnderlyingQuote(
                symbol=underlying.symbol,
                quote_timestamp=underlying.quote_timestamp,
                price=underlying.price,
                bid=underlying.bid,
                ask=underlying.ask,
                bid_venue=underlying.bid_venue,
                ask_venue=underlying.ask_venue,
            )
        ),
        metadata=source_snapshot.metadata,
    )
    frame = option_chain_to_dataframe(snapshot)
    calls = frame[frame["option_type"] == "call"]
    frame["open_interest"] = frame["open_interest"].astype("Int64")
    restored = option_chain_from_dataframe(frame)

    enriched = enrich_option_chain_snapshot(
        snapshot=snapshot,
        carry=carry_assumptions(),
        valuation_date=VALUATION_DATE,
    )
    enriched_frame = enriched_quotes_to_dataframe(enriched)
    summary = (
        enriched_frame
        .groupby("expiration")
        .agg(
            median_spread=("relative_spread", "median"),
            quote_count=("strike", "size"),
        )
    )

    print("DataFrame rows:", len(frame))
    print("Call rows:", len(calls))
    print()
    print(summary)

    assert restored == snapshot
    print()
    print("Round-trip restored == snapshot")


if __name__ == "__main__":
    main()
