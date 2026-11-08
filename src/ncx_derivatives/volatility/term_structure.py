from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from math import isfinite
from typing import Any

from ncx_derivatives.utils._optional import require_pandas

from .delta_metrics import ButterflyResult, RiskReversalResult, SmileDeltaMetrics
from .smile_metrics import SmileMetricResult
from .smiles import VolatilitySmile


class DuplicateTermStructurePolicy(str, Enum):
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class VolatilityTermStructurePoint:
    expiration: date
    time_to_maturity: float
    smile: VolatilitySmile
    local_metrics: SmileMetricResult
    delta_metrics: SmileDeltaMetrics

    def __post_init__(self) -> None:
        if not isinstance(self.expiration, date):
            raise ValueError("expiration must be a date")
        if not isinstance(self.smile, VolatilitySmile):
            raise ValueError("smile must be a VolatilitySmile")
        if not isinstance(self.local_metrics, SmileMetricResult):
            raise ValueError("local_metrics must be a SmileMetricResult")
        if not isinstance(self.delta_metrics, SmileDeltaMetrics):
            raise ValueError("delta_metrics must be a SmileDeltaMetrics")
        if self.local_metrics.smile is not self.smile:
            raise ValueError("local_metrics must reference smile")
        if self.delta_metrics.smile is not self.smile:
            raise ValueError("delta_metrics must reference smile")
        if self.expiration != self.smile.expiration:
            raise ValueError("expiration must match smile")
        if (
            isinstance(self.time_to_maturity, bool)
            or not isinstance(self.time_to_maturity, (int, float))
            or not isfinite(float(self.time_to_maturity))
            or self.time_to_maturity < 0.0
        ):
            raise ValueError("time_to_maturity must be non-negative and finite")
        if self.time_to_maturity != self.smile.time_to_maturity:
            raise ValueError("time_to_maturity must match smile")

    @property
    def atm_volatility(self) -> float | None:
        return self.local_metrics.atm.atm_volatility

    @property
    def atm_total_variance(self) -> float | None:
        return self.local_metrics.atm.atm_total_variance

    @property
    def total_variance_skew_slope(self) -> float | None:
        return self.local_metrics.skew.total_variance_skew_slope

    @property
    def total_variance_curvature(self) -> float | None:
        return self.local_metrics.curvature.total_variance_curvature

    def risk_reversal(self, delta_magnitude: float) -> RiskReversalResult | None:
        return self.delta_metrics.risk_reversal(delta_magnitude)

    def butterfly(self, delta_magnitude: float) -> ButterflyResult | None:
        return self.delta_metrics.butterfly(delta_magnitude)

    @property
    def rr_25(self) -> RiskReversalResult | None:
        return self.risk_reversal(0.25)

    @property
    def bf_25(self) -> ButterflyResult | None:
        return self.butterfly(0.25)

    @property
    def sort_key(self) -> tuple:
        return (self.time_to_maturity, self.expiration)


@dataclass(frozen=True, slots=True)
class VolatilityTermStructure:
    underlying_symbol: str
    valuation_timestamp: datetime
    points: tuple[VolatilityTermStructurePoint, ...]
    duplicate_policy: DuplicateTermStructurePolicy = (
        DuplicateTermStructurePolicy.ERROR
    )

    def __post_init__(self) -> None:
        if not isinstance(self.underlying_symbol, str) or not self.underlying_symbol:
            raise ValueError("underlying_symbol must be a non-empty string")
        if not isinstance(self.valuation_timestamp, datetime):
            raise ValueError("valuation_timestamp must be a datetime")
        if not isinstance(self.duplicate_policy, DuplicateTermStructurePolicy):
            raise ValueError("duplicate_policy must be a DuplicateTermStructurePolicy")
        original_points = tuple(self.points)
        if not original_points:
            raise ValueError("points must not be empty")
        if any(
            not isinstance(point, VolatilityTermStructurePoint)
            for point in original_points
        ):
            raise ValueError("points must contain VolatilityTermStructurePoint")
        points = tuple(sorted(original_points, key=lambda point: point.sort_key))
        for point in points:
            smile = point.smile
            if smile.underlying_symbol != self.underlying_symbol:
                raise ValueError("all points must share underlying_symbol")
            if smile.valuation_timestamp != self.valuation_timestamp:
                raise ValueError("all points must share valuation_timestamp")
        expirations = [point.expiration for point in points]
        maturities = [point.time_to_maturity for point in points]
        if len(set(expirations)) != len(expirations):
            raise ValueError("duplicate expiration is not allowed by ERROR policy")
        if len(set(maturities)) != len(maturities):
            raise ValueError("duplicate maturity is not allowed by ERROR policy")
        object.__setattr__(self, "points", points)

    @property
    def sort_key(self) -> tuple:
        return (self.underlying_symbol, self.valuation_timestamp)


TERM_STRUCTURE_COLUMNS = (
    "underlying_symbol",
    "valuation_timestamp",
    "expiration",
    "time_to_maturity",
    "point_count",
    "atm_iv",
    "atm_total_variance",
    "atm_status",
    "atm_failure_reason",
    "total_variance_skew_slope",
    "skew_status",
    "skew_failure_reason",
    "total_variance_curvature",
    "curvature_status",
    "curvature_failure_reason",
    "rr_25",
    "rr_25_status",
    "rr_25_failure_reason",
    "bf_25",
    "bf_25_status",
    "bf_25_failure_reason",
)


def build_volatility_term_structures(
    local_metric_results: Iterable[SmileMetricResult],
    delta_metric_results: Iterable[SmileDeltaMetrics],
    *,
    duplicate_policy: DuplicateTermStructurePolicy = (
        DuplicateTermStructurePolicy.ERROR
    ),
) -> tuple[VolatilityTermStructure, ...]:
    if not isinstance(duplicate_policy, DuplicateTermStructurePolicy):
        raise ValueError("duplicate_policy must be a DuplicateTermStructurePolicy")
    local = tuple(local_metric_results)
    delta = tuple(delta_metric_results)
    if any(not isinstance(item, SmileMetricResult) for item in local):
        raise ValueError("local_metric_results must contain SmileMetricResult")
    if any(not isinstance(item, SmileDeltaMetrics) for item in delta):
        raise ValueError("delta_metric_results must contain SmileDeltaMetrics")
    local_by_smile = _unique_by_smile(local, "local_metric_results")
    delta_by_smile = _unique_by_smile(delta, "delta_metric_results")
    if set(local_by_smile) != set(delta_by_smile):
        raise ValueError("local and delta results must reference identical smiles")
    grouped: dict[tuple, list[VolatilityTermStructurePoint]] = {}
    for identity, local_result in local_by_smile.items():
        smile = local_result.smile
        grouped.setdefault(
            (smile.underlying_symbol, smile.valuation_timestamp),
            [],
        ).append(
            VolatilityTermStructurePoint(
                expiration=smile.expiration,
                time_to_maturity=smile.time_to_maturity,
                smile=smile,
                local_metrics=local_result,
                delta_metrics=delta_by_smile[identity],
            ),
        )
    return tuple(
        VolatilityTermStructure(
            underlying_symbol=key[0],
            valuation_timestamp=key[1],
            points=tuple(points),
            duplicate_policy=duplicate_policy,
        )
        for key, points in sorted(grouped.items())
    )


def volatility_term_structures_to_records(
    structures: Iterable[VolatilityTermStructure],
) -> tuple[dict[str, Any], ...]:
    values = tuple(structures)
    if any(not isinstance(value, VolatilityTermStructure) for value in values):
        raise ValueError("structures must contain VolatilityTermStructure objects")
    return tuple(
        _term_record(structure, point)
        for structure in sorted(values, key=lambda value: value.sort_key)
        for point in structure.points
    )


def volatility_term_structures_to_dataframe(
    structures: Iterable[VolatilityTermStructure],
):
    pandas = require_pandas()
    return pandas.DataFrame.from_records(
        volatility_term_structures_to_records(structures),
        columns=TERM_STRUCTURE_COLUMNS,
    )


def _unique_by_smile(values: tuple, field_name: str) -> dict[int, Any]:
    result = {id(value.smile): value for value in values}
    if len(result) != len(values):
        raise ValueError(f"{field_name} contains duplicate smiles")
    return result


def _term_record(
    structure: VolatilityTermStructure,
    point: VolatilityTermStructurePoint,
) -> dict[str, Any]:
    local = point.local_metrics
    rr = point.rr_25
    bf = point.bf_25
    return {
        "underlying_symbol": structure.underlying_symbol,
        "valuation_timestamp": structure.valuation_timestamp,
        "expiration": point.expiration,
        "time_to_maturity": point.time_to_maturity,
        "point_count": len(point.smile.points),
        "atm_iv": local.atm.atm_volatility,
        "atm_total_variance": local.atm.atm_total_variance,
        "atm_status": local.atm.status.value,
        "atm_failure_reason": _enum_value(local.atm.failure_reason),
        "total_variance_skew_slope": local.skew.total_variance_skew_slope,
        "skew_status": local.skew.status.value,
        "skew_failure_reason": _enum_value(local.skew.failure_reason),
        "total_variance_curvature": local.curvature.total_variance_curvature,
        "curvature_status": local.curvature.status.value,
        "curvature_failure_reason": _enum_value(local.curvature.failure_reason),
        "rr_25": None if rr is None else rr.value,
        "rr_25_status": None if rr is None else rr.status.value,
        "rr_25_failure_reason": None if rr is None else _enum_value(rr.failure_reason),
        "bf_25": None if bf is None else bf.value,
        "bf_25_status": None if bf is None else bf.status.value,
        "bf_25_failure_reason": None if bf is None else _enum_value(bf.failure_reason),
    }


def _enum_value(value: Enum | None) -> str | None:
    return None if value is None else value.value
