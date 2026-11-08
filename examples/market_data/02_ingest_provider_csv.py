"""Ingest a synthetic provider-style CSV into canonical snapshots."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import ingest_sample_csv, sample_csv_path  # noqa: E402


def main() -> None:
    result = ingest_sample_csv()

    print(f"Input: {sample_csv_path()}")
    print(f"Rows read: {len(result.raw_records)}")
    print(f"Rows accepted: {result.successful_row_count}")
    print(f"Rows failed: {result.failed_row_count}")
    print(f"Snapshots created: {len(result.snapshots)}")

    for index, snapshot in enumerate(result.snapshots, start=1):
        print(
            f"  Snapshot {index}: "
            f"{snapshot.as_of.isoformat()} "
            f"quotes={len(snapshot.quotes)}"
        )


if __name__ == "__main__":
    main()
