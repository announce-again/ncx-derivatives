from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import date, datetime
from enum import Enum
from hashlib import sha256
from math import isnan
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

_TEMPORARY_PREFIX = ".ncx-write-"


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def write_json(path: str | Path, value: object) -> Path:
    output = Path(path)
    with _atomic_output(output) as temporary:
        temporary.write_bytes(canonical_json_bytes(value) + b"\n")
    return output


def write_csv(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
    *,
    columns: Sequence[str] | None = None,
    sort_by: Sequence[str] = (),
) -> Path:
    rows = [dict(record) for record in records]
    if columns is None:
        columns = tuple(sorted({key for row in rows for key in row}))
    if sort_by:
        rows.sort(key=lambda row: tuple(_sort_value(row.get(key)) for key in sort_by))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in columns})
    return output


def write_output_hashes(directory: str | Path, path: str | Path) -> Path:
    root = Path(directory)
    output = Path(path)
    entries = []
    for candidate in sorted(item for item in root.rglob("*") if item.is_file()):
        if candidate.name.startswith(_TEMPORARY_PREFIX) or candidate.resolve() == output.resolve():
            continue
        entries.append(
            {
                "path": candidate.relative_to(root).as_posix(),
                "size_bytes": candidate.stat().st_size,
                "sha256": file_sha256(candidate),
            }
        )
    return write_json(output, {"files": entries})


def logical_frame_sha256(frame, *, sort_by: Sequence[str]) -> str:
    """Hash logical records independently of a Parquet engine's byte layout."""

    ordered = frame.sort_values(list(sort_by), kind="mergesort", ignore_index=True)
    records = ordered.where(ordered.notna(), None).to_dict("records")
    return sha256(canonical_json_bytes(records)).hexdigest()


def write_parquet(path: str | Path, frame, *, sort_by: Sequence[str]) -> Path:
    output = Path(path)
    ordered = frame.sort_values(list(sort_by), kind="mergesort", ignore_index=True)
    try:
        with _atomic_output(output) as temporary:
            ordered.to_parquet(temporary, index=False)
    except ImportError as error:
        raise RuntimeError(
            "writing Research 001 Parquet artifacts requires pyarrow or fastparquet"
        ) from error
    return output


@contextmanager
def _atomic_output(output: Path) -> Iterator[Path]:
    """Publish complete artifacts so interrupted writes can be retried."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=output.parent,
        prefix=_TEMPORARY_PREFIX,
        suffix=".tmp",
        delete=False,
    ) as file:
        temporary = Path(file.name)
    try:
        yield temporary
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if value.__class__.__name__ in {"NAType", "NaTType"}:
        return None
    item = getattr(value, "item", None)
    if callable(item):
        scalar = item()
        if scalar is not value:
            return _json_value(scalar)
    if isinstance(value, float) and isnan(value):
        return None
    return value


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if value.__class__.__name__ in {"NAType", "NaTType"}:
        return ""
    if isinstance(value, float) and isnan(value):
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (list, tuple, dict)):
        return canonical_json_bytes(value).decode("utf-8")
    return value


def _sort_value(value: Any) -> tuple[int, str]:
    return (1, "") if value is None else (0, str(_csv_value(value)))
