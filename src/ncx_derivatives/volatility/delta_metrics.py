from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any

from ncx_derivatives.market_data import OptionType
from ncx_derivatives.utils._optional import require_pandas

from .smile_metrics import (
    AtmVolatilityResult,
    SmileMetricResult,
    SmileMetricStatus,
    calculate_smile_metrics,
)
from .smiles import VolatilitySmile, VolatilitySmilePoint


class DeltaInterpolationMethod(str, Enum):
    OBSERVED = "OBSERVED"
    LINEAR_IV = "LINEAR_IV"


class DeltaMetricFailureReason(str, Enum):
    EMPTY_SMILE = "EMPTY_SMILE"
    INVALID_TARGET_DELTA = "INVALID_TARGET_DELTA"
    NO_POINTS_FOR_OPTION_TYPE = "NO_POINTS_FOR_OPTION_TYPE"
    DELTA_UNAVAILABLE = "DELTA_UNAVAILABLE"
    TARGET_DELTA_NOT_BRACKETED = "TARGET_DELTA_NOT_BRACKETED"
    AMBIGUOUS_DELTA_BRACKET = "AMBIGUOUS_DELTA_BRACKET"
    DEGENERATE_DELTA_COORDINATES = "DEGENERATE_DELTA_COORDINATES"
    NON_FINITE_RESULT = "NON_FINITE_RESULT"


class DeltaStructureFailureReason(str, Enum):
    INVALID_DELTA_MAGNITUDE = "INVALID_DELTA_MAGNITUDE"
    CALL_INTERPOLATION_FAILED = "CALL_INTERPOLATION_FAILED"
    PUT_INTERPOLATION_FAILED = "PUT_INTERPOLATION_FAILED"
    ATM_FAILED = "ATM_FAILED"
    NON_FINITE_RESULT = "NON_FINITE_RESULT"


@dataclass(frozen=True, slots=True)
class DeltaMetricConfig:
    delta_coordinate_tolerance: float = 1e-12
    standard_delta_magnitudes: tuple[float, ...] = (0.25,)

    def __post_init__(self) -> None:
        _validate_non_negative_finite(
            self.delta_coordinate_tolerance,
            "delta_coordinate_tolerance",
        )
        magnitudes = tuple(self.standard_delta_magnitudes)
        if not magnitudes:
            raise ValueError("standard_delta_magnitudes must not be empty")
        for magnitude in magnitudes:
            if not _is_valid_magnitude(magnitude):
                raise ValueError(
                    "standard_delta_magnitudes must be finite numbers "
                    "strictly between 0 and 1",
                )
        normalised = tuple(sorted({float(value) for value in magnitudes}))
        object.__setattr__(self, "standard_delta_magnitudes", normalised)


@dataclass(frozen=True, slots=True)
class DeltaVolatilityResult:
    smile: VolatilitySmile
    option_type: OptionType
    target_delta: float
    status: SmileMetricStatus
    implied_volatility: float | None
    method: DeltaInterpolationMethod | None
    left_point: VolatilitySmilePoint | None
    right_point: VolatilitySmilePoint | None
    interpolation_weight: float | None
    usable_point_count: int
    excluded_point_count: int
    failure_reason: DeltaMetricFailureReason | None

    def __post_init__(self) -> None:
        if not isinstance(self.smile, VolatilitySmile):
            raise ValueError("smile must be a VolatilitySmile")
        if not isinstance(self.option_type, OptionType):
            raise ValueError("option_type must be an OptionType")
        if not isinstance(self.status, SmileMetricStatus):
            raise ValueError("status must be a SmileMetricStatus")
        _validate_non_negative_int(self.usable_point_count, "usable_point_count")
        _validate_non_negative_int(
            self.excluded_point_count,
            "excluded_point_count",
        )
        for point in (self.left_point, self.right_point):
            if point is not None and point.option_type is not self.option_type:
                raise ValueError("source points must match option_type")
            if point is not None and not any(
                point is candidate for candidate in self.smile.points
            ):
                raise ValueError("source points must belong to smile")
        if self.status is SmileMetricStatus.SUCCESS:
            if not _is_valid_target(self.option_type, self.target_delta):
                raise ValueError("successful result requires a valid target_delta")
            _validate_non_negative_finite(
                self.implied_volatility,
                "implied_volatility",
            )
            if not isinstance(self.method, DeltaInterpolationMethod):
                raise ValueError("successful result requires a method")
            if self.failure_reason is not None:
                raise ValueError("successful result cannot have failure_reason")
            if self.left_point is None or self.right_point is None:
                raise ValueError("successful result requires source points")
            _validate_non_negative_finite(
                self.interpolation_weight,
                "interpolation_weight",
            )
            if self.interpolation_weight > 1.0:
                raise ValueError("interpolation_weight must not exceed 1")
        else:
            if not isinstance(self.failure_reason, DeltaMetricFailureReason):
                raise ValueError("failed result requires failure_reason")
            if any(
                value is not None
                for value in (
                    self.implied_volatility,
                    self.method,
                    self.interpolation_weight,
                )
            ):
                raise ValueError("failed result cannot contain a computed value")

    @property
    def is_success(self) -> bool:
        return self.status is SmileMetricStatus.SUCCESS

    @property
    def sort_key(self) -> tuple:
        target_magnitude = (
            abs(float(self.target_delta))
            if isinstance(self.target_delta, (int, float))
            and not isinstance(self.target_delta, bool)
            else float("inf")
        )
        return (*self.smile.sort_key, target_magnitude, self.option_type.value)


@dataclass(frozen=True, slots=True)
class RiskReversalResult:
    smile: VolatilitySmile
    delta_magnitude: float
    status: SmileMetricStatus
    value: float | None
    call_result: DeltaVolatilityResult
    put_result: DeltaVolatilityResult
    failure_reason: DeltaStructureFailureReason | None

    def __post_init__(self) -> None:
        _validate_structure_result(self)

    @property
    def is_success(self) -> bool:
        return self.status is SmileMetricStatus.SUCCESS

    @property
    def sort_key(self) -> tuple:
        return (*self.smile.sort_key, self.delta_magnitude)


@dataclass(frozen=True, slots=True)
class ButterflyResult:
    smile: VolatilitySmile
    delta_magnitude: float
    status: SmileMetricStatus
    value: float | None
    call_result: DeltaVolatilityResult
    put_result: DeltaVolatilityResult
    atm_result: AtmVolatilityResult
    failure_reason: DeltaStructureFailureReason | None

    def __post_init__(self) -> None:
        _validate_structure_result(self)
        if not isinstance(self.atm_result, AtmVolatilityResult):
            raise ValueError("atm_result must be an AtmVolatilityResult")

    @property
    def is_success(self) -> bool:
        return self.status is SmileMetricStatus.SUCCESS

    @property
    def sort_key(self) -> tuple:
        return (*self.smile.sort_key, self.delta_magnitude)


@dataclass(frozen=True, slots=True)
class SmileDeltaMetrics:
    smile: VolatilitySmile
    delta_results: tuple[DeltaVolatilityResult, ...]
    risk_reversals: tuple[RiskReversalResult, ...]
    butterflies: tuple[ButterflyResult, ...]
    config: DeltaMetricConfig

    def __post_init__(self) -> None:
        if not isinstance(self.smile, VolatilitySmile):
            raise ValueError("smile must be a VolatilitySmile")
        if not isinstance(self.config, DeltaMetricConfig):
            raise ValueError("config must be a DeltaMetricConfig")
        deltas = tuple(sorted(self.delta_results, key=lambda item: item.sort_key))
        reversals = tuple(
            sorted(self.risk_reversals, key=lambda item: item.sort_key)
        )
        butterflies = tuple(
            sorted(self.butterflies, key=lambda item: item.sort_key)
        )
        expected_count = 2 * len(self.config.standard_delta_magnitudes)
        if len(deltas) != expected_count:
            raise ValueError("delta_results must contain call and put per magnitude")
        if len(reversals) != len(self.config.standard_delta_magnitudes):
            raise ValueError("risk_reversals must contain one result per magnitude")
        if len(butterflies) != len(self.config.standard_delta_magnitudes):
            raise ValueError("butterflies must contain one result per magnitude")
        if any(item.smile is not self.smile for item in deltas):
            raise ValueError("delta_results must reference smile")
        if any(item.smile is not self.smile for item in reversals):
            raise ValueError("risk_reversals must reference smile")
        if any(item.smile is not self.smile for item in butterflies):
            raise ValueError("butterflies must reference smile")
        object.__setattr__(self, "delta_results", deltas)
        object.__setattr__(self, "risk_reversals", reversals)
        object.__setattr__(self, "butterflies", butterflies)

    @property
    def sort_key(self) -> tuple:
        return self.smile.sort_key

    def delta_result(
        self,
        option_type: OptionType,
        delta_magnitude: float,
    ) -> DeltaVolatilityResult | None:
        target = delta_magnitude if option_type is OptionType.CALL else -delta_magnitude
        return next(
            (
                result
                for result in self.delta_results
                if result.option_type is option_type
                and result.target_delta == target
            ),
            None,
        )

    def risk_reversal(self, delta_magnitude: float) -> RiskReversalResult | None:
        return next(
            (
                result
                for result in self.risk_reversals
                if result.delta_magnitude == delta_magnitude
            ),
            None,
        )

    def butterfly(self, delta_magnitude: float) -> ButterflyResult | None:
        return next(
            (
                result
                for result in self.butterflies
                if result.delta_magnitude == delta_magnitude
            ),
            None,
        )


DELTA_VOLATILITY_COLUMNS = (
    "underlying_symbol",
    "valuation_timestamp",
    "expiration",
    "time_to_maturity",
    "option_type",
    "target_delta",
    "status",
    "failure_reason",
    "method",
    "implied_volatility",
    "left_strike",
    "left_delta",
    "left_iv",
    "right_strike",
    "right_delta",
    "right_iv",
    "interpolation_weight",
    "usable_point_count",
    "excluded_point_count",
    "coordinate_tolerance",
)


DELTA_STRUCTURE_COLUMNS = (
    "underlying_symbol",
    "valuation_timestamp",
    "expiration",
    "time_to_maturity",
    "delta_magnitude",
    "rr_status",
    "rr_failure_reason",
    "risk_reversal",
    "bf_status",
    "bf_failure_reason",
    "butterfly",
    "call_target_delta",
    "call_iv",
    "call_status",
    "call_failure_reason",
    "put_target_delta",
    "put_iv",
    "put_status",
    "put_failure_reason",
    "atm_iv",
    "atm_status",
    "atm_failure_reason",
    "atm_method",
)


RISK_REVERSAL_COLUMNS = (
    "underlying_symbol",
    "valuation_timestamp",
    "expiration",
    "time_to_maturity",
    "delta_magnitude",
    "status",
    "failure_reason",
    "risk_reversal",
    "call_target_delta",
    "call_iv",
    "call_status",
    "call_failure_reason",
    "put_target_delta",
    "put_iv",
    "put_status",
    "put_failure_reason",
)


BUTTERFLY_COLUMNS = (
    "underlying_symbol",
    "valuation_timestamp",
    "expiration",
    "time_to_maturity",
    "delta_magnitude",
    "status",
    "failure_reason",
    "butterfly",
    "call_target_delta",
    "call_iv",
    "call_status",
    "call_failure_reason",
    "put_target_delta",
    "put_iv",
    "put_status",
    "put_failure_reason",
    "atm_iv",
    "atm_status",
    "atm_failure_reason",
    "atm_method",
)


def interpolate_smile_at_delta(
    smile: VolatilitySmile,
    option_type: OptionType,
    target_delta: float,
    config: DeltaMetricConfig | None = None,
) -> DeltaVolatilityResult:
    metric_config = _metric_config(config)
    if not isinstance(smile, VolatilitySmile):
        raise ValueError("smile must be a VolatilitySmile")
    if not isinstance(option_type, OptionType):
        raise ValueError("option_type must be an OptionType")
    typed_points = tuple(
        point for point in smile.points if point.option_type is option_type
    )
    usable = tuple(point for point in typed_points if point.delta is not None)
    counts = (len(usable), len(typed_points) - len(usable))
    if not smile.points:
        return _delta_failure(
            smile,
            option_type,
            target_delta,
            DeltaMetricFailureReason.EMPTY_SMILE,
            *counts,
        )
    if not _is_valid_target(option_type, target_delta):
        return _delta_failure(
            smile,
            option_type,
            target_delta,
            DeltaMetricFailureReason.INVALID_TARGET_DELTA,
            *counts,
        )
    if not typed_points:
        return _delta_failure(
            smile,
            option_type,
            target_delta,
            DeltaMetricFailureReason.NO_POINTS_FOR_OPTION_TYPE,
            *counts,
        )
    if not usable:
        return _delta_failure(
            smile,
            option_type,
            target_delta,
            DeltaMetricFailureReason.DELTA_UNAVAILABLE,
            *counts,
        )

    ordered = tuple(sorted(usable, key=lambda point: point.sort_key))
    tolerance = metric_config.delta_coordinate_tolerance
    exact = tuple(
        point
        for point in ordered
        if abs(float(point.delta) - float(target_delta)) <= tolerance
    )
    if len(exact) == 1:
        point = exact[0]
        return DeltaVolatilityResult(
            smile=smile,
            option_type=option_type,
            target_delta=float(target_delta),
            status=SmileMetricStatus.SUCCESS,
            implied_volatility=point.implied_volatility,
            method=DeltaInterpolationMethod.OBSERVED,
            left_point=point,
            right_point=point,
            interpolation_weight=0.0,
            usable_point_count=counts[0],
            excluded_point_count=counts[1],
            failure_reason=None,
        )
    if len(exact) > 1:
        return _delta_failure(
            smile,
            option_type,
            target_delta,
            DeltaMetricFailureReason.AMBIGUOUS_DELTA_BRACKET,
            *counts,
            left=exact[0],
            right=exact[1],
        )

    degenerate = next(
        (
            (left, right)
            for left, right in zip(ordered, ordered[1:])
            if abs(float(right.delta) - float(left.delta)) <= tolerance
        ),
        None,
    )
    if degenerate is not None:
        return _delta_failure(
            smile,
            option_type,
            target_delta,
            DeltaMetricFailureReason.DEGENERATE_DELTA_COORDINATES,
            *counts,
            left=degenerate[0],
            right=degenerate[1],
        )

    brackets = tuple(
        (left, right)
        for left, right in zip(ordered, ordered[1:])
        if (float(left.delta) - float(target_delta))
        * (float(right.delta) - float(target_delta))
        < 0.0
    )
    if not brackets:
        return _delta_failure(
            smile,
            option_type,
            target_delta,
            DeltaMetricFailureReason.TARGET_DELTA_NOT_BRACKETED,
            *counts,
        )
    if len(brackets) > 1:
        return _delta_failure(
            smile,
            option_type,
            target_delta,
            DeltaMetricFailureReason.AMBIGUOUS_DELTA_BRACKET,
            *counts,
            left=brackets[0][0],
            right=brackets[0][1],
        )

    first, second = brackets[0]
    left, right = sorted((first, second), key=lambda point: float(point.delta))
    denominator = float(right.delta) - float(left.delta)
    if denominator <= tolerance:
        return _delta_failure(
            smile,
            option_type,
            target_delta,
            DeltaMetricFailureReason.DEGENERATE_DELTA_COORDINATES,
            *counts,
            left=left,
            right=right,
        )
    weight = (float(target_delta) - float(left.delta)) / denominator
    volatility = _linear_interpolate_iv(left, right, weight)
    if (
        not isfinite(weight)
        or weight < 0.0
        or weight > 1.0
        or not isfinite(volatility)
        or volatility < 0.0
    ):
        return _delta_failure(
            smile,
            option_type,
            target_delta,
            DeltaMetricFailureReason.NON_FINITE_RESULT,
            *counts,
            left=left,
            right=right,
        )
    return DeltaVolatilityResult(
        smile=smile,
        option_type=option_type,
        target_delta=float(target_delta),
        status=SmileMetricStatus.SUCCESS,
        implied_volatility=volatility,
        method=DeltaInterpolationMethod.LINEAR_IV,
        left_point=left,
        right_point=right,
        interpolation_weight=weight,
        usable_point_count=counts[0],
        excluded_point_count=counts[1],
        failure_reason=None,
    )


def calculate_risk_reversal(
    smile: VolatilitySmile,
    delta_magnitude: float = 0.25,
    config: DeltaMetricConfig | None = None,
    *,
    call_result: DeltaVolatilityResult | None = None,
    put_result: DeltaVolatilityResult | None = None,
) -> RiskReversalResult:
    metric_config = _metric_config(config)
    call, put = _structure_legs(
        smile,
        delta_magnitude,
        metric_config,
        call_result,
        put_result,
    )
    reason = _leg_failure_reason(delta_magnitude, call, put)
    value = None
    if reason is None:
        value = _risk_reversal_value(call, put)
        if not isfinite(value):
            reason = DeltaStructureFailureReason.NON_FINITE_RESULT
            value = None
    return RiskReversalResult(
        smile=smile,
        delta_magnitude=delta_magnitude,
        status=(
            SmileMetricStatus.SUCCESS
            if reason is None
            else SmileMetricStatus.FAILED
        ),
        value=value,
        call_result=call,
        put_result=put,
        failure_reason=reason,
    )


def calculate_symmetric_delta_butterfly(
    smile: VolatilitySmile,
    delta_magnitude: float = 0.25,
    config: DeltaMetricConfig | None = None,
    *,
    atm_result: AtmVolatilityResult | None = None,
    call_result: DeltaVolatilityResult | None = None,
    put_result: DeltaVolatilityResult | None = None,
) -> ButterflyResult:
    metric_config = _metric_config(config)
    call, put = _structure_legs(
        smile,
        delta_magnitude,
        metric_config,
        call_result,
        put_result,
    )
    atm = atm_result or calculate_smile_metrics(smile).atm
    if not isinstance(atm, AtmVolatilityResult):
        raise ValueError("atm_result must be an AtmVolatilityResult or None")
    reason = _leg_failure_reason(delta_magnitude, call, put)
    if reason is None and not atm.is_success:
        reason = DeltaStructureFailureReason.ATM_FAILED
    value = None
    if reason is None:
        value = _butterfly_value(call, put, atm)
        if not isfinite(value):
            reason = DeltaStructureFailureReason.NON_FINITE_RESULT
            value = None
    return ButterflyResult(
        smile=smile,
        delta_magnitude=delta_magnitude,
        status=(
            SmileMetricStatus.SUCCESS
            if reason is None
            else SmileMetricStatus.FAILED
        ),
        value=value,
        call_result=call,
        put_result=put,
        atm_result=atm,
        failure_reason=reason,
    )


def calculate_smile_delta_metrics(
    smile: VolatilitySmile,
    config: DeltaMetricConfig | None = None,
    *,
    local_metric_result: SmileMetricResult | None = None,
) -> SmileDeltaMetrics:
    metric_config = _metric_config(config)
    if local_metric_result is None:
        local_metric_result = calculate_smile_metrics(smile)
    elif (
        not isinstance(local_metric_result, SmileMetricResult)
        or local_metric_result.smile is not smile
    ):
        raise ValueError("local_metric_result must reference smile")
    deltas = []
    reversals = []
    butterflies = []
    for magnitude in metric_config.standard_delta_magnitudes:
        call = interpolate_smile_at_delta(
            smile,
            OptionType.CALL,
            magnitude,
            metric_config,
        )
        put = interpolate_smile_at_delta(
            smile,
            OptionType.PUT,
            -magnitude,
            metric_config,
        )
        deltas.extend((call, put))
        reversals.append(
            calculate_risk_reversal(
                smile,
                magnitude,
                metric_config,
                call_result=call,
                put_result=put,
            ),
        )
        butterflies.append(
            calculate_symmetric_delta_butterfly(
                smile,
                magnitude,
                metric_config,
                atm_result=local_metric_result.atm,
                call_result=call,
                put_result=put,
            ),
        )
    return SmileDeltaMetrics(
        smile=smile,
        delta_results=tuple(deltas),
        risk_reversals=tuple(reversals),
        butterflies=tuple(butterflies),
        config=metric_config,
    )


def calculate_smile_delta_metrics_for_smiles(
    smiles: Iterable[VolatilitySmile],
    config: DeltaMetricConfig | None = None,
    *,
    local_metric_results: Iterable[SmileMetricResult] | None = None,
) -> tuple[SmileDeltaMetrics, ...]:
    ordered = tuple(sorted(tuple(smiles), key=lambda smile: smile.sort_key))
    if any(not isinstance(smile, VolatilitySmile) for smile in ordered):
        raise ValueError("smiles must contain VolatilitySmile objects")
    if local_metric_results is None:
        local_by_identity = {}
    else:
        local_tuple = tuple(local_metric_results)
        if any(not isinstance(item, SmileMetricResult) for item in local_tuple):
            raise ValueError("local_metric_results must contain SmileMetricResult")
        local_by_identity = {id(item.smile): item for item in local_tuple}
        if len(local_by_identity) != len(local_tuple):
            raise ValueError("local_metric_results contains duplicate smiles")
        if set(local_by_identity) != {id(smile) for smile in ordered}:
            raise ValueError("local_metric_results must match smiles exactly")
    return tuple(
        calculate_smile_delta_metrics(
            smile,
            config,
            local_metric_result=local_by_identity.get(id(smile)),
        )
        for smile in ordered
    )


def delta_volatility_results_to_records(
    results: Iterable[SmileDeltaMetrics],
) -> tuple[dict[str, Any], ...]:
    aggregates = _sorted_aggregates(results)
    return tuple(
        _delta_record(delta, aggregate.config)
        for aggregate in aggregates
        for delta in aggregate.delta_results
    )


def delta_volatility_results_to_dataframe(results: Iterable[SmileDeltaMetrics]):
    pandas = require_pandas()
    return pandas.DataFrame.from_records(
        delta_volatility_results_to_records(results),
        columns=DELTA_VOLATILITY_COLUMNS,
    )


def delta_structure_results_to_records(
    results: Iterable[SmileDeltaMetrics],
) -> tuple[dict[str, Any], ...]:
    aggregates = _sorted_aggregates(results)
    return tuple(
        _structure_record(reversal, butterfly)
        for aggregate in aggregates
        for reversal, butterfly in zip(
            aggregate.risk_reversals,
            aggregate.butterflies,
        )
    )


def delta_structure_results_to_dataframe(results: Iterable[SmileDeltaMetrics]):
    pandas = require_pandas()
    return pandas.DataFrame.from_records(
        delta_structure_results_to_records(results),
        columns=DELTA_STRUCTURE_COLUMNS,
    )


def risk_reversal_results_to_records(
    results: Iterable[SmileDeltaMetrics],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        _risk_reversal_record(reversal)
        for aggregate in _sorted_aggregates(results)
        for reversal in aggregate.risk_reversals
    )


def risk_reversal_results_to_dataframe(results: Iterable[SmileDeltaMetrics]):
    pandas = require_pandas()
    return pandas.DataFrame.from_records(
        risk_reversal_results_to_records(results),
        columns=RISK_REVERSAL_COLUMNS,
    )


def butterfly_results_to_records(
    results: Iterable[SmileDeltaMetrics],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        _butterfly_record(butterfly)
        for aggregate in _sorted_aggregates(results)
        for butterfly in aggregate.butterflies
    )


def butterfly_results_to_dataframe(results: Iterable[SmileDeltaMetrics]):
    pandas = require_pandas()
    return pandas.DataFrame.from_records(
        butterfly_results_to_records(results),
        columns=BUTTERFLY_COLUMNS,
    )


def _structure_legs(
    smile: VolatilitySmile,
    magnitude: float,
    config: DeltaMetricConfig,
    call: DeltaVolatilityResult | None,
    put: DeltaVolatilityResult | None,
) -> tuple[DeltaVolatilityResult, DeltaVolatilityResult]:
    call_result = call or interpolate_smile_at_delta(
        smile,
        OptionType.CALL,
        magnitude,
        config,
    )
    put_result = put or interpolate_smile_at_delta(
        smile,
        OptionType.PUT,
        -magnitude,
        config,
    )
    if (
        call_result.smile is not smile
        or call_result.option_type is not OptionType.CALL
        or put_result.smile is not smile
        or put_result.option_type is not OptionType.PUT
    ):
        raise ValueError("provided delta results must match smile and leg type")
    if _is_valid_magnitude(magnitude):
        tolerance = config.delta_coordinate_tolerance
        if (
            abs(float(call_result.target_delta) - float(magnitude)) > tolerance
            or abs(float(put_result.target_delta) + float(magnitude)) > tolerance
        ):
            raise ValueError("provided delta results must match delta_magnitude")
    return call_result, put_result


def _leg_failure_reason(
    magnitude: float,
    call: DeltaVolatilityResult,
    put: DeltaVolatilityResult,
) -> DeltaStructureFailureReason | None:
    if not _is_valid_magnitude(magnitude):
        return DeltaStructureFailureReason.INVALID_DELTA_MAGNITUDE
    if not call.is_success:
        return DeltaStructureFailureReason.CALL_INTERPOLATION_FAILED
    if not put.is_success:
        return DeltaStructureFailureReason.PUT_INTERPOLATION_FAILED
    return None


def _validate_structure_result(result: Any) -> None:
    if not isinstance(result.smile, VolatilitySmile):
        raise ValueError("smile must be a VolatilitySmile")
    if not isinstance(result.status, SmileMetricStatus):
        raise ValueError("status must be a SmileMetricStatus")
    if not isinstance(result.call_result, DeltaVolatilityResult):
        raise ValueError("call_result must be a DeltaVolatilityResult")
    if not isinstance(result.put_result, DeltaVolatilityResult):
        raise ValueError("put_result must be a DeltaVolatilityResult")
    if (
        result.call_result.smile is not result.smile
        or result.put_result.smile is not result.smile
    ):
        raise ValueError("component results must reference smile")
    if result.status is SmileMetricStatus.SUCCESS:
        if not _is_valid_magnitude(result.delta_magnitude):
            raise ValueError("successful result requires a valid magnitude")
        _validate_finite(result.value, "value")
        if result.failure_reason is not None:
            raise ValueError("successful result cannot have failure_reason")
    else:
        if result.value is not None:
            raise ValueError("failed result cannot have value")
        if not isinstance(result.failure_reason, DeltaStructureFailureReason):
            raise ValueError("failed result requires failure_reason")


def _delta_failure(
    smile: VolatilitySmile,
    option_type: OptionType,
    target_delta: float,
    reason: DeltaMetricFailureReason,
    usable_count: int,
    excluded_count: int,
    *,
    left: VolatilitySmilePoint | None = None,
    right: VolatilitySmilePoint | None = None,
) -> DeltaVolatilityResult:
    return DeltaVolatilityResult(
        smile=smile,
        option_type=option_type,
        target_delta=target_delta,
        status=SmileMetricStatus.FAILED,
        implied_volatility=None,
        method=None,
        left_point=left,
        right_point=right,
        interpolation_weight=None,
        usable_point_count=usable_count,
        excluded_point_count=excluded_count,
        failure_reason=reason,
    )


def _delta_record(
    result: DeltaVolatilityResult,
    config: DeltaMetricConfig,
) -> dict[str, Any]:
    smile = result.smile
    return {
        "underlying_symbol": smile.underlying_symbol,
        "valuation_timestamp": smile.valuation_timestamp,
        "expiration": smile.expiration,
        "time_to_maturity": smile.time_to_maturity,
        "option_type": result.option_type.value,
        "target_delta": result.target_delta,
        "status": result.status.value,
        "failure_reason": _enum_value(result.failure_reason),
        "method": _enum_value(result.method),
        "implied_volatility": result.implied_volatility,
        **_source_fields("left", result.left_point),
        **_source_fields("right", result.right_point),
        "interpolation_weight": result.interpolation_weight,
        "usable_point_count": result.usable_point_count,
        "excluded_point_count": result.excluded_point_count,
        "coordinate_tolerance": config.delta_coordinate_tolerance,
    }


def _structure_record(
    reversal: RiskReversalResult,
    butterfly: ButterflyResult,
) -> dict[str, Any]:
    smile = reversal.smile
    call = reversal.call_result
    put = reversal.put_result
    atm = butterfly.atm_result
    return {
        "underlying_symbol": smile.underlying_symbol,
        "valuation_timestamp": smile.valuation_timestamp,
        "expiration": smile.expiration,
        "time_to_maturity": smile.time_to_maturity,
        "delta_magnitude": reversal.delta_magnitude,
        "rr_status": reversal.status.value,
        "rr_failure_reason": _enum_value(reversal.failure_reason),
        "risk_reversal": reversal.value,
        "bf_status": butterfly.status.value,
        "bf_failure_reason": _enum_value(butterfly.failure_reason),
        "butterfly": butterfly.value,
        "call_target_delta": call.target_delta,
        "call_iv": call.implied_volatility,
        "call_status": call.status.value,
        "call_failure_reason": _enum_value(call.failure_reason),
        "put_target_delta": put.target_delta,
        "put_iv": put.implied_volatility,
        "put_status": put.status.value,
        "put_failure_reason": _enum_value(put.failure_reason),
        "atm_iv": atm.atm_volatility,
        "atm_status": atm.status.value,
        "atm_failure_reason": _enum_value(atm.failure_reason),
        "atm_method": _enum_value(atm.method),
    }


def _risk_reversal_record(result: RiskReversalResult) -> dict[str, Any]:
    smile = result.smile
    call = result.call_result
    put = result.put_result
    return {
        "underlying_symbol": smile.underlying_symbol,
        "valuation_timestamp": smile.valuation_timestamp,
        "expiration": smile.expiration,
        "time_to_maturity": smile.time_to_maturity,
        "delta_magnitude": result.delta_magnitude,
        "status": result.status.value,
        "failure_reason": _enum_value(result.failure_reason),
        "risk_reversal": result.value,
        "call_target_delta": call.target_delta,
        "call_iv": call.implied_volatility,
        "call_status": call.status.value,
        "call_failure_reason": _enum_value(call.failure_reason),
        "put_target_delta": put.target_delta,
        "put_iv": put.implied_volatility,
        "put_status": put.status.value,
        "put_failure_reason": _enum_value(put.failure_reason),
    }


def _butterfly_record(result: ButterflyResult) -> dict[str, Any]:
    smile = result.smile
    call = result.call_result
    put = result.put_result
    atm = result.atm_result
    return {
        "underlying_symbol": smile.underlying_symbol,
        "valuation_timestamp": smile.valuation_timestamp,
        "expiration": smile.expiration,
        "time_to_maturity": smile.time_to_maturity,
        "delta_magnitude": result.delta_magnitude,
        "status": result.status.value,
        "failure_reason": _enum_value(result.failure_reason),
        "butterfly": result.value,
        "call_target_delta": call.target_delta,
        "call_iv": call.implied_volatility,
        "call_status": call.status.value,
        "call_failure_reason": _enum_value(call.failure_reason),
        "put_target_delta": put.target_delta,
        "put_iv": put.implied_volatility,
        "put_status": put.status.value,
        "put_failure_reason": _enum_value(put.failure_reason),
        "atm_iv": atm.atm_volatility,
        "atm_status": atm.status.value,
        "atm_failure_reason": _enum_value(atm.failure_reason),
        "atm_method": _enum_value(atm.method),
    }


def _source_fields(
    prefix: str,
    point: VolatilitySmilePoint | None,
) -> dict[str, Any]:
    return {
        f"{prefix}_strike": None if point is None else point.strike,
        f"{prefix}_delta": None if point is None else point.delta,
        f"{prefix}_iv": None if point is None else point.implied_volatility,
    }


def _linear_interpolate_iv(
    left: VolatilitySmilePoint,
    right: VolatilitySmilePoint,
    weight: float,
) -> float:
    return left.implied_volatility + weight * (
        right.implied_volatility - left.implied_volatility
    )


def _risk_reversal_value(
    call: DeltaVolatilityResult,
    put: DeltaVolatilityResult,
) -> float:
    return float(call.implied_volatility) - float(put.implied_volatility)


def _butterfly_value(
    call: DeltaVolatilityResult,
    put: DeltaVolatilityResult,
    atm: AtmVolatilityResult,
) -> float:
    return (
        (float(call.implied_volatility) + float(put.implied_volatility)) / 2.0
        - float(atm.atm_volatility)
    )


def _sorted_aggregates(
    results: Iterable[SmileDeltaMetrics],
) -> tuple[SmileDeltaMetrics, ...]:
    values = tuple(results)
    if any(not isinstance(value, SmileDeltaMetrics) for value in values):
        raise ValueError("results must contain SmileDeltaMetrics objects")
    return tuple(sorted(values, key=lambda value: value.sort_key))


def _metric_config(config: DeltaMetricConfig | None) -> DeltaMetricConfig:
    result = config or DeltaMetricConfig()
    if not isinstance(result, DeltaMetricConfig):
        raise ValueError("config must be a DeltaMetricConfig or None")
    return result


def _is_valid_target(option_type: OptionType, target: float) -> bool:
    if isinstance(target, bool) or not isinstance(target, (int, float)):
        return False
    if not isfinite(float(target)):
        return False
    return (0.0 < target < 1.0) if option_type is OptionType.CALL else (-1.0 < target < 0.0)


def _is_valid_magnitude(value: float) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and isfinite(float(value))
        and 0.0 < value < 1.0
    )


def _validate_non_negative_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _validate_non_negative_finite(value: Any, field_name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or value < 0.0
    ):
        raise ValueError(f"{field_name} must be non-negative and finite")


def _validate_finite(value: Any, field_name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
    ):
        raise ValueError(f"{field_name} must be finite")


def _enum_value(value: Enum | None) -> str | None:
    return None if value is None else value.value
