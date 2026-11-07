"""Piecewise-linear interpolation in log-forward moneyness and total variance."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite, sqrt
from numbers import Real

from .smiles import VolatilitySmile


class SurfaceFailureReason(str, Enum):
    NO_MATURITY_NODES = "NO_MATURITY_NODES"
    NO_LOWER_MATURITY = "NO_LOWER_MATURITY"
    NO_UPPER_MATURITY = "NO_UPPER_MATURITY"
    OUTSIDE_STRIKE_SUPPORT = "OUTSIDE_STRIKE_SUPPORT"


@dataclass(frozen=True, slots=True)
class VarianceInterpolation:
    total_variance: float | None = None
    lower_maturity: float | None = None
    upper_maturity: float | None = None
    upper_weight: float | None = None
    failure_reason: SurfaceFailureReason | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class SurfaceQueryResult(VarianceInterpolation):
    log_forward_moneyness: float
    maturity: float

    @property
    def implied_volatility(self) -> float | None:
        if self.total_variance is None:
            return None
        return sqrt(self.total_variance) / sqrt(self.maturity)


@dataclass(frozen=True, slots=True)
class SurfaceResidual:
    maturity: float
    log_forward_moneyness: float
    observed_total_variance: float
    fitted_total_variance: float | None

    @property
    def total_variance_residual(self) -> float | None:
        """Fitted minus observed; missing when a held-out point cannot be bracketed."""
        if self.fitted_total_variance is None:
            return None
        return self.fitted_total_variance - self.observed_total_variance


@dataclass(frozen=True, slots=True)
class _VarianceCurve:
    coordinates: tuple[float, ...]
    variances: tuple[float, ...]

    @classmethod
    def from_nodes(cls, nodes: Iterable[tuple[float, float]]) -> _VarianceCurve:
        ordered = sorted(nodes)
        coordinates = tuple(x for x, _ in ordered)
        variances = tuple(w for _, w in ordered)
        if any(not _finite_number(x) for x in coordinates):
            raise ValueError("coordinates must be finite numbers")
        if any(not _finite_number(w) or w < 0 for w in variances):
            raise ValueError("total variances must be finite and non-negative")
        if len(set(coordinates)) != len(coordinates):
            raise ValueError("duplicate interpolation coordinate")
        return cls(coordinates, variances)

    def value(self, coordinate: float) -> float | None:
        lower, upper = _bracket(self.coordinates, coordinate)
        if lower is None or upper is None:
            return None
        if lower == upper:
            return self.variances[lower]
        weight = _weight(self.coordinates[lower], self.coordinates[upper], coordinate)
        return _blend(self.variances[lower], self.variances[upper], weight)


def interpolate_total_variance(
    nodes: Iterable[tuple[float, float]],
    maturity: float,
) -> VarianceInterpolation:
    """Interpolate (year fraction, total variance) pairs without extrapolation."""
    _validate_maturity(maturity)
    curve = _VarianceCurve.from_nodes(nodes)
    if curve.coordinates and curve.coordinates[0] <= 0:
        raise ValueError("node maturities must be positive")
    return _interpolate_maturities(
        curve.coordinates, maturity, curve.variances.__getitem__
    )


@dataclass(frozen=True, slots=True)
class VolatilitySurface:
    """A single underlying, valuation snapshot, and quote side of Stage 3.2 smiles."""

    smiles: tuple[VolatilitySmile, ...]
    maturities: tuple[float, ...] = field(init=False)
    _curves: tuple[_VarianceCurve, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        smiles = tuple(self.smiles)
        if not smiles or any(
            not isinstance(smile, VolatilitySmile) for smile in smiles
        ):
            raise ValueError("smiles must contain at least one VolatilitySmile")
        if len({smile.sort_key[:2] for smile in smiles}) != 1:
            raise ValueError("smiles must share an underlying and valuation timestamp")
        if len({point.iv_source for smile in smiles for point in smile.points}) > 1:
            raise ValueError("smiles must share an IV source")
        smiles = tuple(sorted(smiles, key=lambda smile: smile.time_to_maturity))
        maturities = tuple(smile.time_to_maturity for smile in smiles)
        if len(set(maturities)) != len(smiles) or len(
            {s.expiration for s in smiles}
        ) != len(smiles):
            raise ValueError("duplicate surface expiration or maturity")
        curves = []
        for smile in smiles:
            _validate_maturity(smile.time_to_maturity)
            curves.append(
                _VarianceCurve.from_nodes(
                    (
                        point.log_forward_moneyness,
                        point.implied_volatility
                        * point.implied_volatility
                        * smile.time_to_maturity,
                    )
                    for point in smile.points
                )
            )
        object.__setattr__(self, "smiles", smiles)
        object.__setattr__(self, "maturities", maturities)
        object.__setattr__(self, "_curves", tuple(curves))

    def query(
        self, log_forward_moneyness: float, maturity: float
    ) -> SurfaceQueryResult:
        """Use the immediate maturity neighbors, each with its own strike bracket."""
        if not _finite_number(log_forward_moneyness):
            raise ValueError("log_forward_moneyness must be finite")
        _validate_maturity(maturity)
        result = _interpolate_maturities(
            self.maturities,
            maturity,
            lambda index: self._curves[index].value(log_forward_moneyness),
        )
        return SurfaceQueryResult(
            total_variance=result.total_variance,
            lower_maturity=result.lower_maturity,
            upper_maturity=result.upper_maturity,
            upper_weight=result.upper_weight,
            failure_reason=result.failure_reason,
            log_forward_moneyness=log_forward_moneyness,
            maturity=maturity,
        )

    def residuals(self, *, leave_one_out: bool = False) -> tuple[SurfaceResidual, ...]:
        """Compare fitted and observed nodes, optionally withholding each in turn."""
        records = []
        for maturity, curve in zip(self.maturities, self._curves):
            for index, (k, observed) in enumerate(
                zip(curve.coordinates, curve.variances)
            ):
                fitted = curve.value(k)
                if leave_one_out:
                    fitted = None
                    if 0 < index < len(curve.coordinates) - 1:
                        weight = _weight(
                            curve.coordinates[index - 1],
                            curve.coordinates[index + 1],
                            k,
                        )
                        fitted = _blend(
                            curve.variances[index - 1],
                            curve.variances[index + 1],
                            weight,
                        )
                records.append(SurfaceResidual(maturity, k, observed, fitted))
        return tuple(records)


def _interpolate_maturities(
    maturities: tuple[float, ...],
    maturity: float,
    value_at: Callable[[int], float | None],
) -> VarianceInterpolation:
    if not maturities:
        return VarianceInterpolation(
            failure_reason=SurfaceFailureReason.NO_MATURITY_NODES
        )
    lower, upper = _bracket(maturities, maturity)
    if lower is None:
        return VarianceInterpolation(
            upper_maturity=maturities[0],
            failure_reason=SurfaceFailureReason.NO_LOWER_MATURITY,
        )
    if upper is None:
        return VarianceInterpolation(
            lower_maturity=maturities[-1],
            failure_reason=SurfaceFailureReason.NO_UPPER_MATURITY,
        )
    low, high = maturities[lower], maturities[upper]
    left = value_at(lower)
    right = left if lower == upper else value_at(upper)
    if left is None or right is None:
        return VarianceInterpolation(
            lower_maturity=low,
            upper_maturity=high,
            failure_reason=SurfaceFailureReason.OUTSIDE_STRIKE_SUPPORT,
        )
    weight = 0.0 if lower == upper else _weight(low, high, maturity)
    return VarianceInterpolation(_blend(left, right, weight), low, high, weight)


def _bracket(
    coordinates: tuple[float, ...], target: float
) -> tuple[int | None, int | None]:
    upper = bisect_left(coordinates, target)
    if upper < len(coordinates) and coordinates[upper] == target:
        return upper, upper
    return (upper - 1 if upper else None, upper if upper < len(coordinates) else None)


def _weight(lower: float, upper: float, target: float) -> float:
    return (target - lower) / (upper - lower)


def _blend(left: float, right: float, weight: float) -> float:
    return (1.0 - weight) * left + weight * right


def _finite_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)


def _validate_maturity(maturity: float) -> None:
    if not _finite_number(maturity) or maturity <= 0:
        raise ValueError("maturity must be finite and positive")
