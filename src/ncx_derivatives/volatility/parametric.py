"""Raw SVI smile and constrained SSVI surface calibration in total variance."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import Enum
from math import hypot, sqrt
from typing import Generic, TypeVar

from .smiles import VolatilitySmile, VolatilitySmilePoint
from .surface import VolatilitySurface, _finite_number, interpolate_total_variance


class CalibrationWeighting(str, Enum):
    UNIFORM = "uniform"
    INVERSE_SPREAD = "inverse_spread"
    VEGA = "vega"


_SPREAD_FLOOR = 1e-4


@dataclass(frozen=True, slots=True)
class SVIParameters:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def __post_init__(self) -> None:
        if not all(
            _finite_number(x) for x in (self.a, self.b, self.rho, self.m, self.sigma)
        ):
            raise ValueError("SVI parameters must be finite")
        if self.b < 0 or abs(self.rho) >= 1 or self.sigma <= 0:
            raise ValueError("SVI requires b >= 0, |rho| < 1, and sigma > 0")
        if self.a + self.b * self.sigma * sqrt(1 - self.rho**2) <= 0:
            raise ValueError("minimum SVI total variance must be positive")

    @property
    def wing_slopes(self) -> tuple[float, float]:
        return self.b * (1 - self.rho), self.b * (1 + self.rho)

    def total_variance(self, k: float) -> float:
        if not _finite_number(k):
            raise ValueError("log-forward moneyness must be finite")
        x = k - self.m
        return self.a + self.b * (self.rho * x + hypot(x, self.sigma))

    def butterfly_factor(self, k: float) -> float:
        """Density sign factor g(k) in the Gatheral–Jacquier criterion."""
        w = self.total_variance(k)
        x = k - self.m
        radius = hypot(x, self.sigma)
        first = self.b * (self.rho + x / radius)
        second = self.b * (self.sigma / radius) ** 2 / radius
        return (
            (1 - k * first / (2 * w)) ** 2 - first**2 / 4 * (1 / w + 0.25) + second / 2
        )


@dataclass(frozen=True, slots=True)
class SSVISurface:
    """SSVI with phi(theta) = eta / sqrt(theta * (1 + theta))."""

    theta_nodes: tuple[tuple[float, float], ...]
    rho: float
    eta: float

    def __post_init__(self) -> None:
        nodes = tuple(tuple(node) for node in self.theta_nodes)
        if len(nodes) < 2 or any(
            len(node) != 2 or any(not _finite_number(x) or x <= 0 for x in node)
            for node in nodes
        ):
            raise ValueError(
                "SSVI requires at least two positive finite maturity/variance nodes"
            )
        if any(t1 >= t2 or w1 > w2 for (t1, w1), (t2, w2) in zip(nodes, nodes[1:])):
            raise ValueError(
                "SSVI maturities must increase and ATM variance must not decrease"
            )
        if not _finite_number(self.rho) or abs(self.rho) >= 1:
            raise ValueError("SSVI requires |rho| < 1")
        if not _finite_number(self.eta) or not 0 <= self.eta <= 2 / (1 + abs(self.rho)):
            raise ValueError("SSVI requires 0 <= eta <= 2 / (1 + |rho|)")
        object.__setattr__(self, "theta_nodes", nodes)

    def total_variance(self, k: float, maturity: float) -> float | None:
        if not _finite_number(k):
            raise ValueError("log-forward moneyness must be finite")
        theta = interpolate_total_variance(self.theta_nodes, maturity).total_variance
        if theta is None:
            return None
        x = self.eta * (k / sqrt(theta)) / sqrt(1 + theta)
        return (
            theta / 2 * (1 + self.rho * x + hypot(x + self.rho, sqrt(1 - self.rho**2)))
        )

    def implied_volatility(self, k: float, maturity: float) -> float | None:
        variance = self.total_variance(k, maturity)
        return None if variance is None else sqrt(variance / maturity)

    def slice_parameters(self, maturity: float) -> SVIParameters | None:
        """Return the equivalent raw SVI slice for wing and density diagnostics."""
        theta = interpolate_total_variance(self.theta_nodes, maturity).total_variance
        if theta is None:
            return None
        if self.eta == 0:
            return SVIParameters(theta, 0, 0, 0, 1)
        phi = self.eta / sqrt(theta) / sqrt(1 + theta)
        return SVIParameters(
            theta * (1 - self.rho**2) / 2,
            theta * phi / 2,
            self.rho,
            -self.rho / phi,
            sqrt(1 - self.rho**2) / phi,
        )


@dataclass(frozen=True, slots=True)
class CalibrationResidual:
    maturity: float
    log_forward_moneyness: float
    observed_total_variance: float
    fitted_total_variance: float
    weight: float

    @property
    def error(self) -> float:
        return self.fitted_total_variance - self.observed_total_variance


_Model = TypeVar("_Model", SVIParameters, SSVISurface)


@dataclass(frozen=True, slots=True)
class CalibrationResult(Generic[_Model]):
    model: _Model
    residuals: tuple[CalibrationResidual, ...]
    weighted_rmse: float
    omitted_points: int
    converged: bool
    attempts: int
    evaluations: int
    optimizer_message: str
    minimum_sampled_butterfly_factor: float


@dataclass(frozen=True, slots=True)
class ParametricHoldout:
    maturity: float
    log_forward_moneyness: float
    observed_total_variance: float
    nonparametric_total_variance: float
    svi_total_variance: float
    ssvi_total_variance: float


def compare_parametric_holdouts(
    smiles: Iterable[VolatilitySmile],
    *,
    weighting: CalibrationWeighting = CalibrationWeighting.UNIFORM,
) -> tuple[ParametricHoldout, ...]:
    """Withhold one interior quote per expiry and compare models on the same data."""
    surface = VolatilitySurface(tuple(smiles))
    if len(surface.smiles) < 2:
        raise ValueError("holdout comparison requires at least two expiries")
    training, withheld = [], []
    for smile in surface.smiles:
        eligible = {row[0] for row in _observations(smile, weighting, _SPREAD_FLOOR)}
        points = tuple(
            point for point in smile.points if point.log_forward_moneyness in eligible
        )
        if len(points) < 6:
            raise ValueError(
                "holdout comparison requires at least six eligible strikes per expiry"
            )
        index = len(points) // 2
        withheld.append(points[index])
        points = points[:index] + points[index + 1 :]
        nearest = min(
            range(len(points)),
            key=lambda i: (
                abs(points[i].log_forward_moneyness),
                points[i].strike,
                points[i].option_type.value,
            ),
        )
        training.append(replace(smile, points=points, nearest_atm_index=nearest))
    nonparametric = VolatilitySurface(tuple(training))
    ssvi = calibrate_ssvi(training, weighting=weighting)
    records = []
    for smile, point in zip(training, withheld):
        svi = calibrate_svi(smile, weighting=weighting)
        if not (svi.converged and ssvi.converged):
            raise RuntimeError("parametric holdout calibration did not converge")
        k, maturity = point.log_forward_moneyness, smile.time_to_maturity
        records.append(
            ParametricHoldout(
                maturity,
                k,
                point.implied_volatility**2 * maturity,
                nonparametric.query(k, maturity).total_variance,
                svi.model.total_variance(k),
                ssvi.model.total_variance(k, maturity),
            )
        )
    return tuple(records)


def calibrate_svi(
    smile: VolatilitySmile,
    *,
    weighting: CalibrationWeighting = CalibrationWeighting.UNIFORM,
    spread_floor: float = _SPREAD_FLOOR,
    max_evaluations: int = 1500,
) -> CalibrationResult[SVIParameters]:
    """Fit a single expiry; constrain positivity and both wing slopes below two."""
    if not isinstance(smile, VolatilitySmile) or smile.time_to_maturity <= 0:
        raise ValueError("calibration requires a positive-maturity VolatilitySmile")
    np, least_squares = _optimizer()
    observations = _observations(smile, weighting, spread_floor)
    if len(observations) < 5:
        raise ValueError("raw SVI requires at least five distinct eligible strikes")
    k, observed, weights = np.asarray(observations).T
    weights = weights / weights.mean()
    width = k[-1] - k[0]
    level = observed.max()

    def model(p):
        minimum, wing_fraction, rho, center, sigma = p
        b = 1.999 * wing_fraction / (1 + abs(rho))
        return SVIParameters(
            minimum - b * sigma * sqrt(1 - rho**2), b, rho, center, sigma
        )

    def errors(p):
        minimum, wing_fraction, rho, center, sigma = p
        b = 1.999 * wing_fraction / (1 + abs(rho))
        x = k - center
        fitted = minimum + b * (rho * x + np.hypot(x, sigma) - sigma * sqrt(1 - rho**2))
        return np.sqrt(weights) * (fitted - observed) / level

    wing = min(0.75, max(0.005, float(np.ptp(observed) / width)))
    starts = [
        (max(1e-12, observed.min() * 0.7), wing, rho, k[index], width * scale)
        for rho, scale, index in (
            (-0.6, 0.25, len(k) // 2),
            (0.0, 0.5, int(observed.argmin())),
            (0.6, 0.25, len(k) // 2),
            (-0.3, 1.0, int(observed.argmin())),
        )
    ]
    lower = (1e-13, 0, -0.999, k[0] - 2 * width, max(width / 10000, 1e-10))
    upper = (max(level * 5, 1e-8), 1, 0.999, k[-1] + 2 * width, max(width * 10, 1e-8))
    best, evaluations = _fit(
        least_squares, errors, starts, (lower, upper), max_evaluations
    )
    fitted = model(best.x)
    return _result(
        fitted,
        best,
        starts,
        evaluations,
        (smile,),
        (observations,),
        weights,
        _minimum_density(fitted, k, np),
    )


def calibrate_ssvi(
    smiles: Iterable[VolatilitySmile],
    *,
    weighting: CalibrationWeighting = CalibrationWeighting.UNIFORM,
    spread_floor: float = _SPREAD_FLOOR,
    max_evaluations: int = 1500,
) -> CalibrationResult[SSVISurface]:
    """Fit shared skew/curvature and nondecreasing ATM variance to all expiries."""
    surface = VolatilitySurface(tuple(smiles))
    np, least_squares = _optimizer()
    slices = tuple(
        _observations(smile, weighting, spread_floor) for smile in surface.smiles
    )
    if len(slices) < 2 or any(len(rows) < 3 for rows in slices):
        raise ValueError(
            "SSVI requires at least two expiries and three eligible strikes per expiry"
        )
    k, observed, weights = np.asarray([row for rows in slices for row in rows]).T
    weights = weights / weights.mean()
    expiry_index = np.repeat(np.arange(len(slices)), [len(rows) for rows in slices])
    level = observed.max()

    def errors(p):
        rho, eta_fraction = p[:2]
        theta = np.cumsum(p[2:])[expiry_index]
        x = 2 * eta_fraction / (1 + abs(rho)) * k / np.sqrt(theta * (1 + theta))
        fitted = theta / 2 * (1 + rho * x + np.hypot(x + rho, sqrt(1 - rho**2)))
        return np.sqrt(weights) * (fitted - observed) / level

    initial = np.maximum.accumulate(
        [np.interp(0, [r[0] for r in rows], [r[1] for r in rows]) for rows in slices]
    )
    increments = np.maximum(1e-12, np.diff(initial, prepend=0))
    starts = [
        (rho, eta, *increments) for rho, eta in ((-0.6, 0.4), (0.0, 0.2), (0.6, 0.4))
    ]
    bounds = (
        (-0.999, 0, 1e-13, *([0] * (len(slices) - 1))),
        (0.999, 1, *([max(5 * level, 1e-8)] * len(slices))),
    )
    best, evaluations = _fit(least_squares, errors, starts, bounds, max_evaluations)
    rho, eta_fraction = best.x[:2]
    model = SSVISurface(
        tuple(zip(surface.maturities, np.cumsum(best.x[2:]))),
        rho,
        2 * eta_fraction / (1 + abs(rho)),
    )
    minimum_density = min(
        _minimum_density(
            model.slice_parameters(maturity), np.asarray([row[0] for row in rows]), np
        )
        for maturity, rows in zip(surface.maturities, slices)
    )
    return _result(
        model,
        best,
        starts,
        evaluations,
        surface.smiles,
        slices,
        weights,
        minimum_density,
    )


def _minimum_density(model, k, np):
    grid = np.unique(np.r_[k, np.linspace(min(-2, k[0]), max(2, k[-1]), 401)])
    return min(model.butterfly_factor(x) for x in grid)


def _observations(smile, weighting, spread_floor):
    if not isinstance(weighting, CalibrationWeighting):
        raise ValueError("weighting must be a CalibrationWeighting")
    if not _finite_number(spread_floor) or not 0 < spread_floor <= 1:
        raise ValueError("spread_floor must lie in (0, 1]")
    rows = []
    for point in smile.points:
        weight = _weight(point, weighting, spread_floor)
        w = point.implied_volatility**2 * smile.time_to_maturity
        if weight is not None and w > 0:
            rows.append((point.log_forward_moneyness, w, weight))
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError("calibration strikes must be distinct")
    return tuple(sorted(rows))


def _weight(
    point: VolatilitySmilePoint, weighting: CalibrationWeighting, spread_floor: float
):
    if weighting is CalibrationWeighting.UNIFORM:
        return 1.0
    if weighting is CalibrationWeighting.VEGA:
        return point.vega if point.vega and point.vega > 0 else None
    if point.iv_bid_ask_spread is None:
        return None
    return (spread_floor / max(spread_floor, point.iv_bid_ask_spread)) ** 2


def _optimizer():
    try:
        import numpy
        from scipy.optimize import least_squares
    except ImportError as error:
        raise ImportError(
            "SVI/SSVI calibration requires scipy. Install ncx-derivatives[calibration]."
        ) from error
    return numpy, least_squares


def _fit(least_squares, errors, starts, bounds, max_evaluations):
    if (
        isinstance(max_evaluations, bool)
        or not isinstance(max_evaluations, int)
        or max_evaluations < 1
    ):
        raise ValueError("max_evaluations must be a positive integer")
    fits = [
        least_squares(
            errors,
            start,
            bounds=bounds,
            x_scale="jac",
            max_nfev=max_evaluations,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        for start in starts
    ]
    best = min(fits, key=lambda fit: (not fit.success, fit.cost))
    return best, sum(fit.nfev for fit in fits)


def _result(model, best, starts, evaluations, smiles, slices, weights, minimum_density):
    residuals = []
    all_weights = iter(weights)
    for smile, rows in zip(smiles, slices):
        maturity = smile.time_to_maturity
        for k, observed, _ in rows:
            fitted = (
                model.total_variance(k)
                if isinstance(model, SVIParameters)
                else model.total_variance(k, maturity)
            )
            residuals.append(
                CalibrationResidual(
                    maturity, k, observed, fitted, float(next(all_weights))
                )
            )
    return CalibrationResult(
        model,
        tuple(residuals),
        sqrt(sum(row.weight * row.error**2 for row in residuals) / sum(weights)),
        sum(len(smile.points) for smile in smiles) - len(residuals),
        bool(best.success),
        len(starts),
        evaluations,
        best.message,
        minimum_density,
    )
