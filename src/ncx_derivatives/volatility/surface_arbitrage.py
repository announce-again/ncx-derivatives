"""Static-arbitrage diagnostics for Stage 3.3 surfaces on their supported domain."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from math import erfc, exp, isfinite, sqrt
from sys import float_info

from .surface import SurfaceResidual, VolatilitySurface


class SurfaceViolationKind(str, Enum):
    CALENDAR_TOTAL_VARIANCE = "CALENDAR_TOTAL_VARIANCE"
    CALL_SPREAD = "CALL_SPREAD"
    BUTTERFLY = "BUTTERFLY"


@dataclass(frozen=True, slots=True)
class SurfaceViolation:
    kind: SurfaceViolationKind
    maturity: float
    log_forward_moneyness: tuple[float, ...]
    magnitude: float
    uses_interpolation: bool
    later_maturity: float | None = None
    normalized_density_estimate: float | None = None


@dataclass(frozen=True, slots=True)
class StrikeDiagnosticCoverage:
    maturity: float
    minimum_log_forward_moneyness: float | None
    maximum_log_forward_moneyness: float | None
    sample_count: int
    left_wing_call_slope: float | None
    right_wing_call_slope: float | None
    minimum_normalized_density_estimate: float | None


@dataclass(frozen=True, slots=True)
class SurfaceArbitrageDiagnostics:
    violations: tuple[SurfaceViolation, ...]
    strike_coverage: tuple[StrikeDiagnosticCoverage, ...]
    calendar_comparison_count: int
    calendar_pairs_without_overlap: tuple[tuple[float, float], ...]
    observed_residuals: tuple[SurfaceResidual, ...]


def diagnose_surface_arbitrage(
    surface: VolatilitySurface,
    *,
    tolerance: float = 1e-10,
    strike_subdivisions: int = 2,
    maturity_subdivisions: int = 2,
) -> SurfaceArbitrageDiagnostics:
    """Report calendar decreases and sampled normalized-call spread/butterfly violations."""
    if not isinstance(surface, VolatilitySurface):
        raise ValueError("surface must be a VolatilitySurface")
    if isinstance(tolerance, bool) or not isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    for value in (strike_subdivisions, maturity_subdivisions):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("subdivisions must be positive integers")

    residuals = surface.residuals()
    observed = {maturity: set() for maturity in surface.maturities}
    for record in residuals:
        observed[record.maturity].add(record.log_forward_moneyness)
    violations = []
    coverage = []
    for maturity in surface.maturities:
        knots = sorted(observed[maturity])
        # Evaluate the observed grid independently so dense sampling cannot hide a coarse butterfly.
        if strike_subdivisions > 1:
            observed_violations, _ = _diagnose_strikes(
                surface, maturity, knots, observed[maturity], tolerance
            )
            violations.extend(observed_violations)
        sampled, extent = _diagnose_strikes(
            surface,
            maturity,
            _subdivide(knots, strike_subdivisions),
            observed[maturity],
            tolerance,
        )
        violations.extend(sampled)
        coverage.append(extent)

    calendar_count = 0
    no_overlap = []
    for lower, upper in combinations(surface.maturities, 2):
        knots = _overlap_knots(observed[lower], observed[upper])
        if not knots:
            no_overlap.append((lower, upper))
        for k in knots:
            calendar_count += 1
            earlier = surface.query(k, lower).total_variance
            later = surface.query(k, upper).total_variance
            decrease = earlier - later
            if decrease > tolerance:
                violations.append(
                    SurfaceViolation(
                        SurfaceViolationKind.CALENDAR_TOTAL_VARIANCE,
                        lower,
                        (k,),
                        decrease,
                        k not in observed[lower] or k not in observed[upper],
                        later_maturity=upper,
                    )
                )
    for lower, upper in zip(surface.maturities, surface.maturities[1:]):
        knots = _overlap_knots(observed[lower], observed[upper])
        for maturity in _subdivide([lower, upper], maturity_subdivisions)[1:-1]:
            sampled, extent = _diagnose_strikes(
                surface,
                maturity,
                _subdivide(knots, strike_subdivisions),
                set(),
                tolerance,
            )
            violations.extend(sampled)
            coverage.append(extent)
    return SurfaceArbitrageDiagnostics(
        tuple(dict.fromkeys(violations)),
        tuple(sorted(coverage, key=lambda item: item.maturity)),
        calendar_count,
        tuple(no_overlap),
        residuals,
    )


def _diagnose_strikes(surface, maturity, coordinates, observed, tolerance):
    distinct_strikes = {}
    for k in coordinates:
        ratio = exp(k)
        if ratio not in distinct_strikes or k in observed:
            distinct_strikes[ratio] = k
    ratios = list(distinct_strikes)
    coordinates = list(distinct_strikes.values())
    calls = [
        _normalized_call(k, ratio, surface.query(k, maturity).total_variance)
        for k, ratio in zip(coordinates, ratios)
    ]
    widths = [right - left for left, right in zip(ratios, ratios[1:])]
    slopes = [
        (right - left) / width for left, right, width in zip(calls, calls[1:], widths)
    ]
    # The subtraction of two normalized call prices loses precision on very narrow spreads.
    slope_roundoff = [8 * float_info.epsilon / width for width in widths]
    violations = []
    for index, slope in enumerate(slopes):
        magnitude = max(slope, -1.0 - slope)
        if magnitude > tolerance + slope_roundoff[index]:
            ks = tuple(coordinates[index : index + 2])
            violations.append(
                SurfaceViolation(
                    SurfaceViolationKind.CALL_SPREAD,
                    maturity,
                    ks,
                    magnitude,
                    any(k not in observed for k in ks),
                )
            )
    densities = []
    for index, (left, right) in enumerate(zip(slopes, slopes[1:])):
        change = right - left
        density = 2.0 * change / (ratios[index + 2] - ratios[index])
        densities.append(density)
        if -change > tolerance + slope_roundoff[index] + slope_roundoff[index + 1]:
            ks = tuple(coordinates[index : index + 3])
            violations.append(
                SurfaceViolation(
                    SurfaceViolationKind.BUTTERFLY,
                    maturity,
                    ks,
                    -change,
                    any(k not in observed for k in ks),
                    normalized_density_estimate=density,
                )
            )
    return violations, StrikeDiagnosticCoverage(
        maturity,
        coordinates[0] if coordinates else None,
        coordinates[-1] if coordinates else None,
        len(coordinates),
        slopes[0] if slopes else None,
        slopes[-1] if slopes else None,
        min(densities, default=None),
    )


def _normalized_call(k, strike_ratio, variance):
    if variance == 0.0:
        return max(1.0 - strike_ratio, 0.0)
    stddev = sqrt(variance)
    d1 = -k / stddev + 0.5 * stddev
    return 0.5 * erfc(-d1 / sqrt(2.0)) - strike_ratio * (
        0.5 * erfc((stddev - d1) / sqrt(2.0))
    )


def _overlap_knots(left, right):
    if not left or not right:
        return []
    low = max(min(left), min(right))
    high = min(max(left), max(right))
    return sorted(k for k in left | right if low <= k <= high)


def _subdivide(nodes, subdivisions):
    if len(nodes) < 2 or subdivisions == 1:
        return nodes
    grid = set(nodes)
    for left, right in zip(nodes, nodes[1:]):
        grid.update(
            left + (right - left) * step / subdivisions
            for step in range(1, subdivisions)
        )
    return sorted(grid)
