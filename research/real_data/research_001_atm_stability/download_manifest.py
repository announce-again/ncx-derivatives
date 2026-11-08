from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from research.real_data.common.kaggle_manifest import (
    build_dataset_manifest,
    write_dataset_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze provenance for an already-downloaded Kaggle dataset")
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--uploader", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--claimed-source", required=True)
    parser.add_argument("--license", required=True, dest="license_name")
    parser.add_argument("--download-timestamp", required=True)
    parser.add_argument("--date-start", required=True)
    parser.add_argument("--date-end", required=True)
    parser.add_argument("--schema-report", type=Path)
    parser.add_argument("--readme", type=Path)
    arguments = parser.parse_args()
    readme = arguments.readme.read_text(encoding="utf-8") if arguments.readme else None
    schema = (
        json.loads(arguments.schema_report.read_text(encoding="utf-8"))
        if arguments.schema_report
        else {}
    )
    manifest = build_dataset_manifest(
        arguments.dataset_directory,
        dataset_slug=arguments.slug,
        dataset_title=arguments.title,
        uploader=arguments.uploader,
        kaggle_version=arguments.version,
        claimed_original_source=arguments.claimed_source,
        license_name=arguments.license_name,
        readme_snapshot=readme,
        schema=schema,
        date_coverage={"start": arguments.date_start, "end": arguments.date_end},
        download_timestamp=datetime.fromisoformat(arguments.download_timestamp),
    )
    write_dataset_manifest(arguments.dataset_directory, manifest)


if __name__ == "__main__":
    main()
