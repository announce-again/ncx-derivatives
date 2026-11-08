from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite, sqrt
from typing import Any

from ncx_derivatives.utils._optional import require_pandas

from .smiles import VolatilitySmile, VolatilitySmilePoint


class SmileMetricStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class SmileMetricFailureReason(str, Enum):
    EMPTY_SMILE = "EMPTY_SMILE"
    NON_POSITIVE_MATURITY = "NON_POSITIVE_MATURITY"
    INSUFFICIENT_POINTS = "INSUFFICIENT_POINTS"
    ATM_NOT_BRACKETED = "ATM_NOT_BRACKETED"
    DEGENERATE_COORDINATES = "DEGENERATE_COORDINATES"
    NEGATIVE_TOTAL_VARIANCE = "NEGATIVE_TOTAL_VARIANCE"
    NON_FINITE_RESULT = "NON_FINITE_RESULT"
    LOCAL_FIT_FAILED = "LOCAL_FIT_FAILED"


class AtmInterpolationMethod(str, Enum):
    OBSERVED = "OBSERVED"
    LINEAR_TOTAL_VARIANCE = "LINEAR_TOTAL_VARIANCE"


class LocalSkewMethod(str, Enum):
    BRACKET_SECANT = "BRACKET_SECANT"
    QUADRATIC_LOCAL_FIT = "QUADRATIC_LOCAL_FIT"


class LocalCurvatureMethod(str, Enum):
    QUADRATIC_LOCAL_FIT = "QUADRATIC_LOCAL_FIT"


class LocalFitWeighting(str, Enum):
    UNWEIGHTED = "UNWEIGHTED"


@dataclass(frozen=True, slots=True)
class LocalSmileFitConfig:
    max_points_each_side: int = 2
    minimum_point_count: int = 3
    maximum_abs_log_moneyness: float | None = None
    weighting: LocalFitWeighting = LocalFitWeighting.UNWEIGHTED
    coordinate_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        _validate_positive_int(
            self.max_points_each_side,
            "max_points_each_side",
        )
        _validate_positive_int(self.minimum_point_count, "minimum_point_count")
        if self.minimum_point_count < 3:
            raise ValueError("minimum_point_count must be at least 3")
        _validate_optional_positive_finite(
            self.maximum_abs_log_moneyness,
            "maximum_abs_log_moneyness",
        )
        if not isinstance(self.weighting, LocalFitWeighting):
            raise ValueError("weighting must be a LocalFitWeighting")
        _validate_non_negative_finite(
            self.coordinate_tolerance,
            "coordinate_tolerance",
        )


@dataclass(frozen=True, slots=True)
class SmileMetricConfig:
    coordinate_tolerance: float = 1e-12
    allow_observed_atm: bool = True
    interpolation_method: AtmInterpolationMethod = (
        AtmInterpolationMethod.LINEAR_TOTAL_VARIANCE
    )
    local_fit: LocalSmileFitConfig = field(default_factory=LocalSmileFitConfig)

    def __post_init__(self) -> None:
        _validate_non_negative_finite(
            self.coordinate_tolerance,
            "coordinate_tolerance",
        )
        if not isinstance(self.allow_observed_atm, bool):
            raise ValueError("allow_observed_atm must be a bool")
        if not isinstance(self.interpolation_method, AtmInterpolationMethod):
            raise ValueError(
                "interpolation_method must be an AtmInterpolationMethod",
            )
        if self.interpolation_method is not (
            AtmInterpolationMethod.LINEAR_TOTAL_VARIANCE
        ):
            raise ValueError(
                "interpolation_method must be LINEAR_TOTAL_VARIANCE",
            )
        if not isinstance(self.local_fit, LocalSmileFitConfig):
            raise ValueError("local_fit must be a LocalSmileFitConfig")


@dataclass(frozen=True, slots=True)
class AtmVolatilityResult:
    atm_volatility: float | None
    atm_total_variance: float | None
    method: AtmInterpolationMethod | None
    left_point: VolatilitySmilePoint | None
    center_point: VolatilitySmilePoint | None
    right_point: VolatilitySmilePoint | None
    interpolation_weight: float | None
    status: SmileMetricStatus
    failure_reason: SmileMetricFailureReason | None

    @property
    def is_success(self) -> bool:
        return self.status is SmileMetricStatus.SUCCESS


@dataclass(frozen=True, slots=True)
class LocalSkewResult:
    total_variance_skew_slope: float | None
    method: LocalSkewMethod | None
    source_points: tuple[VolatilitySmilePoint, ...]
    fit_intercept: float | None
    fit_linear_coefficient: float | None
    fit_quadratic_coefficient: float | None
    status: SmileMetricStatus
    failure_reason: SmileMetricFailureReason | None

    @property
    def is_success(self) -> bool:
        return self.status is SmileMetricStatus.SUCCESS


@dataclass(frozen=True, slots=True)
class LocalCurvatureResult:
    total_variance_curvature: float | None
    method: LocalCurvatureMethod | None
    source_points: tuple[VolatilitySmilePoint, ...]
    fit_intercept: float | None
    fit_linear_coefficient: float | None
    fit_quadratic_coefficient: float | None
    status: SmileMetricStatus
    failure_reason: SmileMetricFailureReason | None

    @property
    def is_success(self) -> bool:
        return self.status is SmileMetricStatus.SUCCESS


@dataclass(frozen=True, slots=True)
class SmileMetricResult:
    smile: VolatilitySmile
    atm: AtmVolatilityResult
    skew: LocalSkewResult
    curvature: LocalCurvatureResult
    config: SmileMetricConfig

    def __post_init__(self) -> None:
        if not isinstance(self.smile, VolatilitySmile):
            raise ValueError("smile must be a VolatilitySmile")
        if not isinstance(self.atm, AtmVolatilityResult):
            raise ValueError("atm must be an AtmVolatilityResult")
        if not isinstance(self.skew, LocalSkewResult):
            raise ValueError("skew must be a LocalSkewResult")
        if not isinstance(self.curvature, LocalCurvatureResult):
            raise ValueError("curvature must be a LocalCurvatureResult")
        if not isinstance(self.config, SmileMetricConfig):
            raise ValueError("config must be a SmileMetricConfig")

    @property
    def sort_key(self) -> tuple:
        return self.smile.sort_key


@dataclass(frozen=True, slots=True)
class _QuadraticFit:
    intercept: float
    linear_coefficient: float
    quadratic_coefficient: float
    source_points: tuple[VolatilitySmilePoint, ...]


SMILE_METRIC_COLUMNS = (
    "underlying_symbol",
    "valuation_timestamp",
    "expiration",
    "time_to_maturity",
    "point_count",
    "atm_volatility",
    "atm_total_variance",
    "atm_method",
    "atm_status",
    "atm_failure_reason",
    "atm_left_strike",
    "atm_left_log_forward_moneyness",
    "atm_center_strike",
    "atm_center_log_forward_moneyness",
    "atm_right_strike",
    "atm_right_log_forward_moneyness",
    "atm_interpolation_weight",
    "total_variance_skew_slope",
    "skew_method",
    "skew_status",
    "skew_failure_reason",
    "skew_source_point_count",
    "skew_source_strikes",
    "skew_source_option_types",
    "skew_source_log_forward_moneyness",
    "skew_source_min_log_forward_moneyness",
    "skew_source_max_log_forward_moneyness",
    "total_variance_curvature",
    "curvature_method",
    "curvature_status",
    "curvature_failure_reason",
    "curvature_source_point_count",
    "curvature_source_strikes",
    "curvature_source_option_types",
    "curvature_source_log_forward_moneyness",
    "curvature_source_min_log_forward_moneyness",
    "curvature_source_max_log_forward_moneyness",
    "local_fit_intercept",
    "local_fit_linear_coefficient",
    "local_fit_quadratic_coefficient",
    "coordinate_tolerance",
    "local_fit_coordinate_tolerance",
    "local_fit_max_points_each_side",
    "local_fit_minimum_point_count",
    "local_fit_maximum_abs_log_moneyness",
    "local_fit_weighting",
)


def calculate_smile_metrics(
    smile: VolatilitySmile,
    config: SmileMetricConfig | None = None,
) -> SmileMetricResult:
    if not isinstance(smile, VolatilitySmile):
        raise ValueError("smile must be a VolatilitySmile")
    metric_config = config or SmileMetricConfig()
    if not isinstance(metric_config, SmileMetricConfig):
        raise ValueError("config must be a SmileMetricConfig or None")
    return SmileMetricResult(
        smile=smile,
        atm=_calculate_atm(smile, metric_config),
        skew=_calculate_skew(smile, metric_config),
        curvature=_calculate_curvature(smile, metric_config),
        config=metric_config,
    )


def calculate_smile_metrics_for_smiles(
    smiles: Iterable[VolatilitySmile],
    config: SmileMetricConfig | None = None,
) -> tuple[SmileMetricResult, ...]:
    smile_tuple = tuple(smiles)
    if any(not isinstance(smile, VolatilitySmile) for smile in smile_tuple):
        raise ValueError("smiles must contain VolatilitySmile objects")
    return tuple(
        calculate_smile_metrics(smile, config)
        for smile in sorted(smile_tuple, key=lambda item: item.sort_key)
    )


def smile_metrics_to_records(
    results: Iterable[SmileMetricResult],
) -> tuple[dict[str, Any], ...]:
    result_tuple = tuple(results)
    if any(not isinstance(result, SmileMetricResult) for result in result_tuple):
        raise ValueError("results must contain SmileMetricResult objects")
    return tuple(
        _metric_record(result)
        for result in sorted(result_tuple, key=lambda item: item.sort_key)
    )


def smile_metrics_to_dataframe(results: Iterable[SmileMetricResult]):
    pandas = require_pandas()
    return pandas.DataFrame.from_records(
        smile_metrics_to_records(results),
        columns=SMILE_METRIC_COLUMNS,
    )


def _calculate_atm(
    smile: VolatilitySmile,
    config: SmileMetricConfig,
) -> AtmVolatilityResult:
    context_failure = _context_failure(smile)
    if context_failure is not None:
        return _atm_failure(context_failure)

    if config.allow_observed_atm and smile.observed_atm_point is not None:
        point = smile.observed_atm_point
        total_variance = _total_variance(point, smile.time_to_maturity)
        if total_variance is None:
            return _atm_failure(
                SmileMetricFailureReason.NON_FINITE_RESULT,
                left=point,
                center=point,
                right=point,
            )
        return AtmVolatilityResult(
            atm_volatility=point.implied_volatility,
            atm_total_variance=total_variance,
            method=AtmInterpolationMethod.OBSERVED,
            left_point=point,
            center_point=point,
            right_point=point,
            interpolation_weight=0.0,
            status=SmileMetricStatus.SUCCESS,
            failure_reason=None,
        )

    left, right = _bracket_points(smile.points, config.coordinate_tolerance)
    if left is None or right is None:
        return _atm_failure(
            SmileMetricFailureReason.ATM_NOT_BRACKETED,
            left=left,
            right=right,
        )
    denominator = right.log_forward_moneyness - left.log_forward_moneyness
    if denominator <= config.coordinate_tolerance:
        return _atm_failure(
            SmileMetricFailureReason.DEGENERATE_COORDINATES,
            left=left,
            right=right,
        )
    left_variance = _total_variance(left, smile.time_to_maturity)
    right_variance = _total_variance(right, smile.time_to_maturity)
    if left_variance is None or right_variance is None:
        return _atm_failure(
            SmileMetricFailureReason.NON_FINITE_RESULT,
            left=left,
            right=right,
        )
    weight = -left.log_forward_moneyness / denominator
    total_variance = left_variance + weight * (right_variance - left_variance)
    if not isfinite(total_variance):
        return _atm_failure(
            SmileMetricFailureReason.NON_FINITE_RESULT,
            left=left,
            right=right,
        )
    if total_variance < 0.0:
        return _atm_failure(
            SmileMetricFailureReason.NEGATIVE_TOTAL_VARIANCE,
            left=left,
            right=right,
        )
    volatility = sqrt(total_variance / smile.time_to_maturity)
    if not isfinite(volatility):
        return _atm_failure(
            SmileMetricFailureReason.NON_FINITE_RESULT,
            left=left,
            right=right,
        )
    return AtmVolatilityResult(
        atm_volatility=volatility,
        atm_total_variance=total_variance,
        method=config.interpolation_method,
        left_point=left,
        center_point=None,
        right_point=right,
        interpolation_weight=weight,
        status=SmileMetricStatus.SUCCESS,
        failure_reason=None,
    )


def _calculate_skew(
    smile: VolatilitySmile,
    config: SmileMetricConfig,
) -> LocalSkewResult:
    context_failure = _context_failure(smile)
    if context_failure is not None:
        return _skew_failure(context_failure)
    points = _select_local_points(smile, config.local_fit)
    coordinate_failure = _coordinate_failure(points, config.local_fit)
    if coordinate_failure is not None:
        return _skew_failure(coordinate_failure, points)
    left, right = _bracket_points(
        points,
        config.local_fit.coordinate_tolerance,
    )
    if left is None or right is None:
        return _skew_failure(
            SmileMetricFailureReason.ATM_NOT_BRACKETED,
            points,
        )
    if len(points) < config.local_fit.minimum_point_count:
        denominator = right.log_forward_moneyness - left.log_forward_moneyness
        if denominator <= config.local_fit.coordinate_tolerance:
            return _skew_failure(
                SmileMetricFailureReason.DEGENERATE_COORDINATES,
                points,
            )
        left_variance = _total_variance(left, smile.time_to_maturity)
        right_variance = _total_variance(right, smile.time_to_maturity)
        if left_variance is None or right_variance is None:
            return _skew_failure(
                SmileMetricFailureReason.NON_FINITE_RESULT,
                points,
            )
        slope = (right_variance - left_variance) / denominator
        if not isfinite(slope):
            return _skew_failure(
                SmileMetricFailureReason.NON_FINITE_RESULT,
                points,
            )
        return LocalSkewResult(
            total_variance_skew_slope=slope,
            method=LocalSkewMethod.BRACKET_SECANT,
            source_points=(left, right),
            fit_intercept=None,
            fit_linear_coefficient=None,
            fit_quadratic_coefficient=None,
            status=SmileMetricStatus.SUCCESS,
            failure_reason=None,
        )

    fit, failure = _quadratic_fit(smile, points, config.local_fit)
    if fit is None:
        return _skew_failure(
            failure or SmileMetricFailureReason.LOCAL_FIT_FAILED,
            points,
        )
    return LocalSkewResult(
        total_variance_skew_slope=fit.linear_coefficient,
        method=LocalSkewMethod.QUADRATIC_LOCAL_FIT,
        source_points=fit.source_points,
        fit_intercept=fit.intercept,
        fit_linear_coefficient=fit.linear_coefficient,
        fit_quadratic_coefficient=fit.quadratic_coefficient,
        status=SmileMetricStatus.SUCCESS,
        failure_reason=None,
    )


def _calculate_curvature(
    smile: VolatilitySmile,
    config: SmileMetricConfig,
) -> LocalCurvatureResult:
    context_failure = _context_failure(smile)
    if context_failure is not None:
        return _curvature_failure(context_failure)
    points = _select_local_points(smile, config.local_fit)
    coordinate_failure = _coordinate_failure(points, config.local_fit)
    if coordinate_failure is not None:
        return _curvature_failure(coordinate_failure, points)
    left, right = _bracket_points(
        points,
        config.local_fit.coordinate_tolerance,
    )
    if left is None or right is None:
        return _curvature_failure(
            SmileMetricFailureReason.ATM_NOT_BRACKETED,
            points,
        )
    if len(points) < config.local_fit.minimum_point_count:
        return _curvature_failure(
            SmileMetricFailureReason.INSUFFICIENT_POINTS,
            points,
        )
    fit, failure = _quadratic_fit(smile, points, config.local_fit)
    if fit is None:
        return _curvature_failure(
            failure or SmileMetricFailureReason.LOCAL_FIT_FAILED,
            points,
        )
    curvature = 2.0 * fit.quadratic_coefficient
    if not isfinite(curvature):
        return _curvature_failure(
            SmileMetricFailureReason.NON_FINITE_RESULT,
            points,
        )
    return LocalCurvatureResult(
        total_variance_curvature=curvature,
        method=LocalCurvatureMethod.QUADRATIC_LOCAL_FIT,
        source_points=fit.source_points,
        fit_intercept=fit.intercept,
        fit_linear_coefficient=fit.linear_coefficient,
        fit_quadratic_coefficient=fit.quadratic_coefficient,
        status=SmileMetricStatus.SUCCESS,
        failure_reason=None,
    )


def _context_failure(
    smile: VolatilitySmile,
) -> SmileMetricFailureReason | None:
    if smile.time_to_maturity <= 0.0:
        return SmileMetricFailureReason.NON_POSITIVE_MATURITY
    if not smile.points:
        return SmileMetricFailureReason.EMPTY_SMILE
    return None


def _select_local_points(
    smile: VolatilitySmile,
    config: LocalSmileFitConfig,
) -> tuple[VolatilitySmilePoint, ...]:
    eligible = tuple(
        point
        for point in smile.points
        if config.maximum_abs_log_moneyness is None
        or abs(point.log_forward_moneyness)
        <= config.maximum_abs_log_moneyness
    )
    tolerance = config.coordinate_tolerance
    left = sorted(
        (
            point
            for point in eligible
            if point.log_forward_moneyness < -tolerance
        ),
        key=lambda point: point.log_forward_moneyness,
        reverse=True,
    )[: config.max_points_each_side]
    right = sorted(
        (
            point
            for point in eligible
            if point.log_forward_moneyness > tolerance
        ),
        key=lambda point: point.log_forward_moneyness,
    )[: config.max_points_each_side]
    center = sorted(
        (
            point
            for point in eligible
            if abs(point.log_forward_moneyness) <= tolerance
        ),
        key=lambda point: (
            abs(point.log_forward_moneyness),
            point.strike,
            point.option_type.value,
        ),
    )[:1]
    return tuple(
        sorted(
            (*left, *center, *right),
            key=lambda point: point.log_forward_moneyness,
        )
    )


def _coordinate_failure(
    points: tuple[VolatilitySmilePoint, ...],
    config: LocalSmileFitConfig,
) -> SmileMetricFailureReason | None:
    if len(points) < 2:
        return SmileMetricFailureReason.INSUFFICIENT_POINTS
    coordinates = tuple(point.log_forward_moneyness for point in points)
    if any(
        right - left <= config.coordinate_tolerance
        for left, right in zip(coordinates, coordinates[1:])
    ):
        return SmileMetricFailureReason.DEGENERATE_COORDINATES
    return None


def _quadratic_fit(
    smile: VolatilitySmile,
    points: tuple[VolatilitySmilePoint, ...],
    config: LocalSmileFitConfig,
) -> tuple[_QuadraticFit | None, SmileMetricFailureReason | None]:
    scale = max(abs(point.log_forward_moneyness) for point in points)
    if scale <= config.coordinate_tolerance:
        return None, SmileMetricFailureReason.DEGENERATE_COORDINATES
    coordinates = tuple(point.log_forward_moneyness / scale for point in points)
    variances = tuple(
        _total_variance(point, smile.time_to_maturity) for point in points
    )
    if any(value is None for value in variances):
        return None, SmileMetricFailureReason.NON_FINITE_RESULT
    values = tuple(float(value) for value in variances if value is not None)
    s0 = float(len(points))
    s1 = sum(coordinates)
    s2 = sum(value * value for value in coordinates)
    s3 = sum(value**3 for value in coordinates)
    s4 = sum(value**4 for value in coordinates)
    t0 = sum(values)
    t1 = sum(x * value for x, value in zip(coordinates, values))
    t2 = sum(x * x * value for x, value in zip(coordinates, values))
    coefficients = _solve_3x3(
        (
            (s0, s1, s2),
            (s1, s2, s3),
            (s2, s3, s4),
        ),
        (t0, t1, t2),
    )
    if coefficients is None:
        return None, SmileMetricFailureReason.LOCAL_FIT_FAILED
    intercept, scaled_linear, scaled_quadratic = coefficients
    linear = scaled_linear / scale
    quadratic = scaled_quadratic / (scale * scale)
    if not all(isfinite(value) for value in (intercept, linear, quadratic)):
        return None, SmileMetricFailureReason.NON_FINITE_RESULT
    if intercept < 0.0:
        return None, SmileMetricFailureReason.LOCAL_FIT_FAILED
    return (
        _QuadraticFit(
            intercept=intercept,
            linear_coefficient=linear,
            quadratic_coefficient=quadratic,
            source_points=points,
        ),
        None,
    )


def _solve_3x3(
    matrix: tuple[tuple[float, float, float], ...],
    vector: tuple[float, float, float],
) -> tuple[float, float, float] | None:
    augmented = [list(row) + [value] for row, value in zip(matrix, vector)]
    for column in range(3):
        pivot_row = max(
            range(column, 3),
            key=lambda row: abs(augmented[row][column]),
        )
        if abs(augmented[pivot_row][column]) <= 1e-14:
            return None
        augmented[column], augmented[pivot_row] = (
            augmented[pivot_row],
            augmented[column],
        )
        pivot = augmented[column][column]
        for index in range(column, 4):
            augmented[column][index] /= pivot
        for row in range(3):
            if row == column:
                continue
            factor = augmented[row][column]
            for index in range(column, 4):
                augmented[row][index] -= factor * augmented[column][index]
    result = tuple(augmented[row][3] for row in range(3))
    if not all(isfinite(value) for value in result):
        return None
    return result[0], result[1], result[2]


def _bracket_points(
    points: Iterable[VolatilitySmilePoint],
    tolerance: float,
) -> tuple[VolatilitySmilePoint | None, VolatilitySmilePoint | None]:
    point_tuple = tuple(points)
    left_candidates = tuple(
        point
        for point in point_tuple
        if point.log_forward_moneyness < -tolerance
    )
    right_candidates = tuple(
        point
        for point in point_tuple
        if point.log_forward_moneyness > tolerance
    )
    left = (
        max(left_candidates, key=lambda point: point.log_forward_moneyness)
        if left_candidates
        else None
    )
    right = (
        min(right_candidates, key=lambda point: point.log_forward_moneyness)
        if right_candidates
        else None
    )
    return left, right


def _total_variance(
    point: VolatilitySmilePoint,
    maturity: float,
) -> float | None:
    result = point.implied_volatility * point.implied_volatility * maturity
    if not isfinite(result) or result < 0.0:
        return None
    return result


def _atm_failure(
    reason: SmileMetricFailureReason,
    *,
    left: VolatilitySmilePoint | None = None,
    center: VolatilitySmilePoint | None = None,
    right: VolatilitySmilePoint | None = None,
) -> AtmVolatilityResult:
    return AtmVolatilityResult(
        atm_volatility=None,
        atm_total_variance=None,
        method=None,
        left_point=left,
        center_point=center,
        right_point=right,
        interpolation_weight=None,
        status=SmileMetricStatus.FAILED,
        failure_reason=reason,
    )


def _skew_failure(
    reason: SmileMetricFailureReason,
    points: tuple[VolatilitySmilePoint, ...] = (),
) -> LocalSkewResult:
    return LocalSkewResult(
        total_variance_skew_slope=None,
        method=None,
        source_points=points,
        fit_intercept=None,
        fit_linear_coefficient=None,
        fit_quadratic_coefficient=None,
        status=SmileMetricStatus.FAILED,
        failure_reason=reason,
    )


def _curvature_failure(
    reason: SmileMetricFailureReason,
    points: tuple[VolatilitySmilePoint, ...] = (),
) -> LocalCurvatureResult:
    return LocalCurvatureResult(
        total_variance_curvature=None,
        method=None,
        source_points=points,
        fit_intercept=None,
        fit_linear_coefficient=None,
        fit_quadratic_coefficient=None,
        status=SmileMetricStatus.FAILED,
        failure_reason=reason,
    )


def _metric_record(result: SmileMetricResult) -> dict[str, Any]:
    smile = result.smile
    atm = result.atm
    skew = result.skew
    curvature = result.curvature
    fit = result.config.local_fit
    return {
        "underlying_symbol": smile.underlying_symbol,
        "valuation_timestamp": smile.valuation_timestamp,
        "expiration": smile.expiration,
        "time_to_maturity": smile.time_to_maturity,
        "point_count": len(smile.points),
        "atm_volatility": atm.atm_volatility,
        "atm_total_variance": atm.atm_total_variance,
        "atm_method": _enum_value(atm.method),
        "atm_status": atm.status.value,
        "atm_failure_reason": _enum_value(atm.failure_reason),
        **_point_record_fields("atm_left", atm.left_point),
        **_point_record_fields("atm_center", atm.center_point),
        **_point_record_fields("atm_right", atm.right_point),
        "atm_interpolation_weight": atm.interpolation_weight,
        "total_variance_skew_slope": skew.total_variance_skew_slope,
        "skew_method": _enum_value(skew.method),
        "skew_status": skew.status.value,
        "skew_failure_reason": _enum_value(skew.failure_reason),
        "skew_source_point_count": len(skew.source_points),
        "skew_source_strikes": _source_values(skew.source_points, "strike"),
        "skew_source_option_types": _source_values(
            skew.source_points,
            "option_type",
        ),
        "skew_source_log_forward_moneyness": _source_values(
            skew.source_points,
            "log_forward_moneyness",
        ),
        "skew_source_min_log_forward_moneyness": _source_min_k(
            skew.source_points,
        ),
        "skew_source_max_log_forward_moneyness": _source_max_k(
            skew.source_points,
        ),
        "total_variance_curvature": curvature.total_variance_curvature,
        "curvature_method": _enum_value(curvature.method),
        "curvature_status": curvature.status.value,
        "curvature_failure_reason": _enum_value(curvature.failure_reason),
        "curvature_source_point_count": len(curvature.source_points),
        "curvature_source_strikes": _source_values(
            curvature.source_points,
            "strike",
        ),
        "curvature_source_option_types": _source_values(
            curvature.source_points,
            "option_type",
        ),
        "curvature_source_log_forward_moneyness": _source_values(
            curvature.source_points,
            "log_forward_moneyness",
        ),
        "curvature_source_min_log_forward_moneyness": _source_min_k(
            curvature.source_points,
        ),
        "curvature_source_max_log_forward_moneyness": _source_max_k(
            curvature.source_points,
        ),
        "local_fit_intercept": curvature.fit_intercept,
        "local_fit_linear_coefficient": curvature.fit_linear_coefficient,
        "local_fit_quadratic_coefficient": curvature.fit_quadratic_coefficient,
        "coordinate_tolerance": result.config.coordinate_tolerance,
        "local_fit_coordinate_tolerance": fit.coordinate_tolerance,
        "local_fit_max_points_each_side": fit.max_points_each_side,
        "local_fit_minimum_point_count": fit.minimum_point_count,
        "local_fit_maximum_abs_log_moneyness": (
            fit.maximum_abs_log_moneyness
        ),
        "local_fit_weighting": fit.weighting.value,
    }


def _point_record_fields(
    prefix: str,
    point: VolatilitySmilePoint | None,
) -> dict[str, Any]:
    return {
        f"{prefix}_strike": None if point is None else point.strike,
        f"{prefix}_log_forward_moneyness": (
            None if point is None else point.log_forward_moneyness
        ),
    }


def _source_min_k(points: tuple[VolatilitySmilePoint, ...]) -> float | None:
    if not points:
        return None
    return min(point.log_forward_moneyness for point in points)


def _source_values(
    points: tuple[VolatilitySmilePoint, ...],
    field_name: str,
) -> str:
    values = []
    for point in points:
        value = getattr(point, field_name)
        enum_value = getattr(value, "value", None)
        values.append(str(enum_value if enum_value is not None else value))
    return "|".join(values)


def _source_max_k(points: tuple[VolatilitySmilePoint, ...]) -> float | None:
    if not points:
        return None
    return max(point.log_forward_moneyness for point in points)


def _enum_value(value: Enum | None) -> str | None:
    return None if value is None else value.value


def _validate_positive_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _validate_non_negative_finite(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    if not isfinite(float(value)) or value < 0.0:
        raise ValueError(f"{field_name} must be non-negative and finite")


def _validate_optional_positive_finite(
    value: float | None,
    field_name: str,
) -> None:
    if value is None:
        return
    _validate_non_negative_finite(value, field_name)
    if value <= 0.0:
        raise ValueError(f"{field_name} must be positive or None")
