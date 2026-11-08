from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from hashlib import sha256
from pathlib import Path

from ncx_derivatives.volatility import SmileIvSource


@dataclass(frozen=True, slots=True)
class Research001Config:
    dataset_ids: tuple[str, ...]
    start_date: date
    end_date: date
    underlyings: tuple[str, ...]
    minimum_dte: int = 7
    maximum_dte: int = 180
    target_tenors: tuple[int, ...] = (21, 45, 90, 150)
    tenor_tolerances: tuple[int, ...] = (7, 10, 15, 25)
    iv_sources: tuple[SmileIvSource, ...] = (
        SmileIvSource.BID,
        SmileIvSource.MIDPOINT,
        SmileIvSource.ASK,
    )
    minimum_smile_points: int = 5
    liquidity_policy: str = "baseline"
    split_policy: str = "exclude_cross_split"

    def __post_init__(self) -> None:
        if self.start_date > self.end_date:
            raise ValueError("start_date must not be after end_date")
        if not 0 <= self.minimum_dte <= self.maximum_dte:
            raise ValueError("DTE bounds must satisfy 0 <= minimum <= maximum")
        if not self.dataset_ids or not self.underlyings:
            raise ValueError("dataset_ids and underlyings must not be empty")
        if len(self.target_tenors) != len(self.tenor_tolerances):
            raise ValueError("each target tenor requires one tolerance")
        if any(value <= 0 for value in self.target_tenors):
            raise ValueError("target_tenors must be positive")
        if any(value < 0 for value in self.tenor_tolerances):
            raise ValueError("tenor_tolerances must be non-negative")
        if self.minimum_smile_points < 2:
            raise ValueError("minimum_smile_points must be at least 2")
        if any(not isinstance(value, SmileIvSource) for value in self.iv_sources):
            raise ValueError("iv_sources must contain SmileIvSource values")
        object.__setattr__(self, "dataset_ids", tuple(dict.fromkeys(self.dataset_ids)))
        object.__setattr__(self, "underlyings", tuple(dict.fromkeys(self.underlyings)))
        object.__setattr__(self, "iv_sources", tuple(dict.fromkeys(self.iv_sources)))

    def to_record(self) -> dict[str, object]:
        record = asdict(self)
        record["start_date"] = self.start_date.isoformat()
        record["end_date"] = self.end_date.isoformat()
        record["iv_sources"] = [source.value for source in self.iv_sources]
        for field in ("dataset_ids", "underlyings", "target_tenors", "tenor_tolerances"):
            record[field] = list(record[field])
        return record

    @property
    def sha256(self) -> str:
        return sha256(_canonical_json(self.to_record())).hexdigest()

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(self.to_record())
        payload["config_sha256"] = self.sha256
        output.write_bytes(_canonical_json(payload) + b"\n")
        return output


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

