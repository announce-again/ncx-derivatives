from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from .deterministic_io import file_sha256, write_json


def build_dataset_manifest(
    dataset_directory: str | Path,
    *,
    dataset_slug: str,
    dataset_title: str,
    uploader: str,
    kaggle_version: str,
    claimed_original_source: str,
    license_name: str,
    readme_snapshot: str | None = None,
    schema: Mapping[str, object] | None = None,
    date_coverage: Mapping[str, object] | None = None,
    download_timestamp: datetime | None = None,
) -> dict[str, object]:
    root = Path(dataset_directory)
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "dataset_manifest.json":
            continue
        files.append(
            {
                "name": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    timestamp = download_timestamp or datetime.now(timezone.utc)
    return {
        "dataset_slug": dataset_slug,
        "dataset_title": dataset_title,
        "uploader": uploader,
        "kaggle_version": kaggle_version,
        "download_timestamp": timestamp.isoformat(),
        "claimed_original_source": claimed_original_source,
        "license": license_name,
        "files": files,
        "date_coverage": dict(date_coverage or {}),
        "schema": dict(schema or {}),
        "readme_snapshot": readme_snapshot,
    }


def write_dataset_manifest(directory: str | Path, manifest: Mapping[str, object]) -> Path:
    return write_json(Path(directory) / "dataset_manifest.json", dict(manifest))
