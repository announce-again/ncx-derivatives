from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from math import inf, isclose, isfinite, log
from numbers import Real
from typing import Any

from ncx_derivatives.greeks import call_delta, put_delta
from ncx_derivatives.market_data import OptionType
from ncx_derivatives.utils._optional import require_pandas

from .chains import (
    ImpliedVolatilityChain,
    ImpliedVolatilityDiagnosticFlag,
    ImpliedVolatilityQuote,
    ImpliedVolatilityResult,
    ImpliedVolatilityStatus,
)


class SmileIvSource(str, Enum):
    BID = "bid"
    MIDPOINT = "midpoint"
    ASK = "ask"


class SmileSelectionReason(str, Enum):
    FAILED_IV = "FAILED_IV"
    MISSING_BID = "MISSING_BID"
    MISSING_ASK = "MISSING_ASK"
    LOW_VEGA = "LOW_VEGA"
    VEGA_UNAVAILABLE = "VEGA_UNAVAILABLE"
    UPPER_BOUND_IV = "UPPER_BOUND_IV"
    NOT_OTM = "NOT_OTM"
    LIQUIDITY_FILTER = "LIQUIDITY_FILTER"
    STALE_QUOTE = "STALE_QUOTE"
    DUPLICATE_STRIKE = "DUPLICATE_STRIKE"
    NON_FINITE_IV = "NON_FINITE_IV"


class DuplicateStrikePolicy(str, Enum):
    PREFER_OTM = "PREFER_OTM"
    MOST_LIQUID = "MOST_LIQUID"
    PREFER_CALL = "PREFER_CALL"
    PREFER_PUT = "PREFER_PUT"


class SmilePointDiagnosticFlag(str, Enum):
    DELTA_BOUNDARY_UNAVAILABLE = "DELTA_BOUNDARY_UNAVAILABLE"
    DELTA_NUMERICAL_FAILURE = "DELTA_NUMERICAL_FAILURE"
    IV_SPREAD_UNAVAILABLE = "IV_SPREAD_UNAVAILABLE"


class SmileGroupDiagnosticReason(str, Enum):
    INCONSISTENT_MARKET_STATE = "INCONSISTENT_MARKET_STATE"


_FLAG_REASONS = {
    ImpliedVolatilityDiagnosticFlag.LOW_VEGA: SmileSelectionReason.LOW_VEGA,
    ImpliedVolatilityDiagnosticFlag.VEGA_UNAVAILABLE: (
        SmileSelectionReason.VEGA_UNAVAILABLE
    ),
    ImpliedVolatilityDiagnosticFlag.UPPER_BOUND_IV: (
        SmileSelectionReason.UPPER_BOUND_IV
    ),
}


@dataclass(frozen=True, slots=True)
class SmileSelectionConfig:
    iv_source: SmileIvSource = SmileIvSource.MIDPOINT
    otm_only: bool = True
    require_two_sided_quote: bool = True
    duplicate_strike_policy: DuplicateStrikePolicy = (
        DuplicateStrikePolicy.PREFER_OTM
    )
    excluded_diagnostic_flags: tuple[ImpliedVolatilityDiagnosticFlag, ...] = (
        ImpliedVolatilityDiagnosticFlag.LOW_VEGA,
        ImpliedVolatilityDiagnosticFlag.VEGA_UNAVAILABLE,
        ImpliedVolatilityDiagnosticFlag.UPPER_BOUND_IV,
    )
    atm_log_moneyness_tolerance: float = 1e-12
    max_relative_spread: float | None = None
    min_bid_size: int | None = None
    min_ask_size: int | None = None
    min_session_volume: int | None = None
    min_open_interest: int | None = None
    market_state_relative_tolerance: float = 1e-12
    market_state_absolute_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if not isinstance(self.iv_source, SmileIvSource):
            raise ValueError("iv_source must be a SmileIvSource")
        if not isinstance(self.otm_only, bool):
            raise ValueError("otm_only must be a bool")
        if not isinstance(self.require_two_sided_quote, bool):
            raise ValueError("require_two_sided_quote must be a bool")
        if not isinstance(self.duplicate_strike_policy, DuplicateStrikePolicy):
            raise ValueError(
                "duplicate_strike_policy must be a DuplicateStrikePolicy",
            )
        flags = tuple(self.excluded_diagnostic_flags)
        if any(
            not isinstance(flag, ImpliedVolatilityDiagnosticFlag)
            for flag in flags
        ):
            raise ValueError(
                "excluded_diagnostic_flags must contain diagnostic flags",
            )
        if any(flag not in _FLAG_REASONS for flag in flags):
            raise ValueError("unsupported excluded diagnostic flag")
        object.__setattr__(self, "excluded_diagnostic_flags", tuple(dict.fromkeys(flags)))
        _validate_non_negative_finite(
            self.atm_log_moneyness_tolerance,
            "atm_log_moneyness_tolerance",
        )
        _validate_optional_non_negative_finite(
            self.max_relative_spread,
            "max_relative_spread",
        )
        _validate_non_negative_finite(
            self.market_state_relative_tolerance,
            "market_state_relative_tolerance",
        )
        _validate_non_negative_finite(
            self.market_state_absolute_tolerance,
            "market_state_absolute_tolerance",
        )
        for field_name in (
            "min_bid_size",
            "min_ask_size",
            "min_session_volume",
            "min_open_interest",
        ):
            _validate_optional_non_negative_int(getattr(self, field_name), field_name)


@dataclass(frozen=True, slots=True)
class VolatilitySmilePoint:
    iv_quote: ImpliedVolatilityQuote
    iv_source: SmileIvSource
    implied_volatility: float
    vega: float | None
    delta: float | None
    iv_bid_ask_spread: float | None
    diagnostic_flags: tuple[SmilePointDiagnosticFlag, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.iv_quote, ImpliedVolatilityQuote):
            raise ValueError("iv_quote must be an ImpliedVolatilityQuote")
        if not isinstance(self.iv_source, SmileIvSource):
            raise ValueError("iv_source must be a SmileIvSource")
        if (
            isinstance(self.implied_volatility, bool)
            or not isinstance(self.implied_volatility, Real)
            or not isfinite(float(self.implied_volatility))
            or self.implied_volatility < 0.0
        ):
            raise ValueError(
                "implied_volatility must be non-negative and finite",
            )
        _validate_optional_non_negative_finite(self.vega, "vega")
        _validate_optional_finite(self.delta, "delta")
        _validate_optional_non_negative_finite(
            self.iv_bid_ask_spread,
            "iv_bid_ask_spread",
        )
        flags = tuple(dict.fromkeys(self.diagnostic_flags))
        if any(not isinstance(flag, SmilePointDiagnosticFlag) for flag in flags):
            raise ValueError(
                "diagnostic_flags must contain SmilePointDiagnosticFlag values",
            )
        delta_flags = {
            SmilePointDiagnosticFlag.DELTA_BOUNDARY_UNAVAILABLE,
            SmilePointDiagnosticFlag.DELTA_NUMERICAL_FAILURE,
        }
        delta_flag_count = sum(flag in delta_flags for flag in flags)
        if (
            (self.delta is None and delta_flag_count != 1)
            or (self.delta is not None and delta_flag_count != 0)
        ):
            raise ValueError(
                "delta availability must match a delta diagnostic flag",
            )
        has_spread_flag = (
            SmilePointDiagnosticFlag.IV_SPREAD_UNAVAILABLE in flags
        )
        if (self.iv_bid_ask_spread is None) != has_spread_flag:
            raise ValueError(
                "IV spread availability must match its diagnostic flag",
            )
        object.__setattr__(self, "diagnostic_flags", flags)

    @property
    def strike(self) -> float:
        return self.iv_quote.enriched_quote.quote.contract.strike

    @property
    def option_type(self) -> OptionType:
        return self.iv_quote.enriched_quote.quote.contract.option_type

    @property
    def log_forward_moneyness(self) -> float:
        return self.iv_quote.enriched_quote.log_moneyness

    @property
    def quote_timestamp(self) -> datetime:
        return self.iv_quote.enriched_quote.quote.quote_timestamp

    @property
    def sort_key(self) -> tuple:
        return (self.strike, self.option_type.value, self.iv_quote.sort_key)


@dataclass(frozen=True, slots=True)
class VolatilitySmile:
    underlying_symbol: str
    valuation_timestamp: datetime
    expiration: date
    time_to_maturity: float
    spot_price: float
    forward_price: float
    points: tuple[VolatilitySmilePoint, ...]
    nearest_atm_index: int | None
    atm_log_moneyness_tolerance: float
    market_state_relative_tolerance: float
    market_state_absolute_tolerance: float

    def __post_init__(self) -> None:
        original_points = tuple(self.points)
        if any(
            not isinstance(point, VolatilitySmilePoint)
            for point in original_points
        ):
            raise ValueError("points must contain VolatilitySmilePoint objects")
        original_atm = (
            None
            if self.nearest_atm_index is None
            else original_points[self.nearest_atm_index]
            if not isinstance(self.nearest_atm_index, bool)
            and isinstance(self.nearest_atm_index, int)
            and 0 <= self.nearest_atm_index < len(original_points)
            else None
        )
        points = tuple(sorted(original_points, key=lambda point: point.sort_key))
        object.__setattr__(self, "points", points)
        _validate_non_negative_finite(
            self.atm_log_moneyness_tolerance,
            "atm_log_moneyness_tolerance",
        )
        _validate_non_negative_finite(
            self.market_state_relative_tolerance,
            "market_state_relative_tolerance",
        )
        _validate_non_negative_finite(
            self.market_state_absolute_tolerance,
            "market_state_absolute_tolerance",
        )
        _validate_non_negative_finite(
            self.time_to_maturity,
            "time_to_maturity",
        )
        _validate_positive_finite(self.spot_price, "spot_price")
        _validate_positive_finite(self.forward_price, "forward_price")
        _validate_smile_point_group_identity(self, points)
        _validate_smile_point_market_state(self, points)
        if self.nearest_atm_index is None:
            if points:
                raise ValueError(
                    "nearest_atm_index is required when points are present",
                )
        elif original_atm is None:
            raise ValueError(
                "nearest_atm_index must identify an existing point",
            )
        else:
            sorted_atm_index = next(
                index
                for index, point in enumerate(points)
                if point is original_atm
            )
            expected_atm_index = min(
                range(len(points)),
                key=lambda index: (
                    abs(points[index].log_forward_moneyness),
                    points[index].strike,
                    points[index].option_type.value,
                ),
            )
            if sorted_atm_index != expected_atm_index:
                raise ValueError(
                    "nearest_atm_index must identify the nearest-forward point",
                )
            object.__setattr__(
                self,
                "nearest_atm_index",
                sorted_atm_index,
            )

    @property
    def nearest_atm_point(self) -> VolatilitySmilePoint | None:
        if self.nearest_atm_index is None:
            return None
        return self.points[self.nearest_atm_index]

    @property
    def observed_atm_point(self) -> VolatilitySmilePoint | None:
        point = self.nearest_atm_point
        if point is None:
            return None
        if abs(point.log_forward_moneyness) > self.atm_log_moneyness_tolerance:
            return None
        return point

    @property
    def has_observed_atm_point(self) -> bool:
        return self.observed_atm_point is not None

    @property
    def sort_key(self) -> tuple[str, datetime, date]:
        return (self.underlying_symbol, self.valuation_timestamp, self.expiration)


@dataclass(frozen=True, slots=True)
class SmileSelectionDiagnostic:
    iv_quote: ImpliedVolatilityQuote
    iv_source: SmileIvSource
    reasons: tuple[SmileSelectionReason, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.iv_quote, ImpliedVolatilityQuote):
            raise ValueError("iv_quote must be an ImpliedVolatilityQuote")
        if not isinstance(self.iv_source, SmileIvSource):
            raise ValueError("iv_source must be a SmileIvSource")
        reasons = tuple(dict.fromkeys(self.reasons))
        if not reasons or any(
            not isinstance(reason, SmileSelectionReason) for reason in reasons
        ):
            raise ValueError("reasons must contain SmileSelectionReason values")
        object.__setattr__(self, "reasons", reasons)

    @property
    def source_result(self) -> ImpliedVolatilityResult:
        return _source_result(self.iv_quote, self.iv_source)

    @property
    def sort_key(self) -> tuple:
        return (
            self.iv_quote.sort_key,
            tuple(reason.value for reason in self.reasons),
        )


@dataclass(frozen=True, slots=True)
class SmileGroupDiagnostic:
    underlying_symbol: str
    valuation_timestamp: datetime
    expiration: date
    reason: SmileGroupDiagnosticReason
    inconsistent_fields: tuple[str, ...]
    iv_quotes: tuple[ImpliedVolatilityQuote, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.reason, SmileGroupDiagnosticReason):
            raise ValueError("reason must be a SmileGroupDiagnosticReason")
        fields = tuple(dict.fromkeys(self.inconsistent_fields))
        if not fields:
            raise ValueError("inconsistent_fields must not be empty")
        quotes = tuple(sorted(self.iv_quotes, key=lambda quote: quote.sort_key))
        if not quotes or any(
            not isinstance(quote, ImpliedVolatilityQuote) for quote in quotes
        ):
            raise ValueError(
                "iv_quotes must contain ImpliedVolatilityQuote objects",
            )
        for iv_quote in quotes:
            enriched = iv_quote.enriched_quote
            contract = enriched.quote.contract
            if (
                contract.underlying_symbol != self.underlying_symbol
                or enriched.valuation_timestamp != self.valuation_timestamp
                or contract.expiration != self.expiration
            ):
                raise ValueError(
                    "iv_quotes must belong to the diagnostic group",
                )
        object.__setattr__(self, "inconsistent_fields", fields)
        object.__setattr__(self, "iv_quotes", quotes)

    @property
    def quote_count(self) -> int:
        return len(self.iv_quotes)

    @property
    def sort_key(self) -> tuple[str, datetime, date]:
        return (self.underlying_symbol, self.valuation_timestamp, self.expiration)


@dataclass(frozen=True, slots=True)
class SmileSelectionSummary:
    input_quote_count: int
    smile_count: int
    empty_smile_count: int
    selected_point_count: int
    excluded_quote_count: int
    quote_diagnostic_count: int
    group_diagnostic_count: int
    group_rejected_quote_count: int
    reason_counts: tuple[tuple[SmileSelectionReason, int], ...]
    group_reason_counts: tuple[
        tuple[SmileGroupDiagnosticReason, int], ...
    ]


@dataclass(frozen=True, slots=True)
class SmileSelectionResult:
    smiles: tuple[VolatilitySmile, ...]
    diagnostics: tuple[SmileSelectionDiagnostic, ...]
    group_diagnostics: tuple[SmileGroupDiagnostic, ...]
    input_quote_count: int
    config: SmileSelectionConfig

    def __post_init__(self) -> None:
        if not isinstance(self.config, SmileSelectionConfig):
            raise ValueError("config must be a SmileSelectionConfig")
        smiles = tuple(sorted(self.smiles, key=lambda smile: smile.sort_key))
        diagnostics = tuple(
            sorted(self.diagnostics, key=lambda diagnostic: diagnostic.sort_key)
        )
        group_diagnostics = tuple(
            sorted(
                self.group_diagnostics,
                key=lambda diagnostic: diagnostic.sort_key,
            )
        )
        object.__setattr__(self, "smiles", smiles)
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "group_diagnostics", group_diagnostics)

    @property
    def summary(self) -> SmileSelectionSummary:
        counts = Counter(
            reason
            for diagnostic in self.diagnostics
            for reason in diagnostic.reasons
        )
        group_rejected_quote_count = sum(
            diagnostic.quote_count for diagnostic in self.group_diagnostics
        )
        group_counts = Counter(
            diagnostic.reason for diagnostic in self.group_diagnostics
        )
        return SmileSelectionSummary(
            input_quote_count=self.input_quote_count,
            smile_count=len(self.smiles),
            empty_smile_count=sum(not smile.points for smile in self.smiles),
            selected_point_count=sum(len(smile.points) for smile in self.smiles),
            excluded_quote_count=(
                len(self.diagnostics) + group_rejected_quote_count
            ),
            quote_diagnostic_count=len(self.diagnostics),
            group_diagnostic_count=len(self.group_diagnostics),
            group_rejected_quote_count=group_rejected_quote_count,
            reason_counts=tuple(
                (reason, counts[reason])
                for reason in SmileSelectionReason
                if counts[reason]
            ),
            group_reason_counts=tuple(
                (reason, group_counts[reason])
                for reason in SmileGroupDiagnosticReason
                if group_counts[reason]
            ),
        )


SMILE_POINT_COLUMNS = (
    "underlying_symbol",
    "valuation_timestamp",
    "expiration",
    "time_to_maturity",
    "spot_price",
    "forward_price",
    "atm_log_moneyness_tolerance",
    "market_state_relative_tolerance",
    "market_state_absolute_tolerance",
    "quote_timestamp",
    "strike",
    "option_type",
    "iv_source",
    "implied_volatility",
    "vega",
    "delta",
    "bid_iv",
    "midpoint_iv",
    "ask_iv",
    "iv_bid_ask_spread",
    "spot_moneyness",
    "forward_moneyness",
    "log_forward_moneyness",
    "relative_spread",
    "bid_size",
    "ask_size",
    "session_volume",
    "open_interest",
    "source_status",
    "source_failure_reason",
    "source_diagnostic_flags",
    "point_diagnostic_flags",
    "is_nearest_atm",
    "is_observed_atm",
)


SMILE_SELECTION_DIAGNOSTIC_COLUMNS = (
    "underlying_symbol",
    "valuation_timestamp",
    "expiration",
    "quote_timestamp",
    "strike",
    "option_type",
    "iv_source",
    "reasons",
    "source_status",
    "source_failure_reason",
    "source_diagnostic_flags",
)


SMILE_GROUP_DIAGNOSTIC_COLUMNS = (
    "underlying_symbol",
    "valuation_timestamp",
    "expiration",
    "reason",
    "inconsistent_fields",
    "quote_count",
)


def build_volatility_smiles(
    chain: ImpliedVolatilityChain,
    config: SmileSelectionConfig | None = None,
) -> SmileSelectionResult:
    """Select research smile points from Stage 3.1 IV observations."""

    if not isinstance(chain, ImpliedVolatilityChain):
        raise ValueError("chain must be an ImpliedVolatilityChain")
    selection = config or SmileSelectionConfig()
    if not isinstance(selection, SmileSelectionConfig):
        raise ValueError("config must be a SmileSelectionConfig or None")

    grouped: dict[tuple[str, datetime, date], list[ImpliedVolatilityQuote]] = (
        defaultdict(list)
    )
    for iv_quote in chain.quotes:
        enriched = iv_quote.enriched_quote
        contract = enriched.quote.contract
        grouped[
            (
                contract.underlying_symbol,
                enriched.valuation_timestamp,
                contract.expiration,
            )
        ].append(iv_quote)

    smiles: list[VolatilitySmile] = []
    diagnostics: list[SmileSelectionDiagnostic] = []
    group_diagnostics: list[SmileGroupDiagnostic] = []
    for group_key in sorted(grouped):
        smile, quote_diagnostics, group_diagnostic = _build_group_smile(
            group_key,
            grouped[group_key],
            selection,
        )
        if smile is not None:
            smiles.append(smile)
        diagnostics.extend(quote_diagnostics)
        if group_diagnostic is not None:
            group_diagnostics.append(group_diagnostic)

    return SmileSelectionResult(
        smiles=tuple(smiles),
        diagnostics=tuple(diagnostics),
        group_diagnostics=tuple(group_diagnostics),
        input_quote_count=len(chain.quotes),
        config=selection,
    )


def volatility_smiles_to_records(
    smiles: Iterable[VolatilitySmile],
) -> tuple[dict[str, Any], ...]:
    smile_tuple = tuple(smiles)
    if any(not isinstance(smile, VolatilitySmile) for smile in smile_tuple):
        raise ValueError("smiles must contain VolatilitySmile objects")
    return tuple(
        _point_record(
            smile,
            point,
            point_index == smile.nearest_atm_index,
            point is smile.observed_atm_point,
        )
        for smile in sorted(smile_tuple, key=lambda item: item.sort_key)
        for point_index, point in enumerate(smile.points)
    )


def volatility_smiles_to_dataframe(smiles: Iterable[VolatilitySmile]):
    pandas = require_pandas()
    return pandas.DataFrame.from_records(
        volatility_smiles_to_records(smiles),
        columns=SMILE_POINT_COLUMNS,
    )


def smile_selection_diagnostics_to_records(
    diagnostics: Iterable[SmileSelectionDiagnostic],
) -> tuple[dict[str, Any], ...]:
    diagnostic_tuple = tuple(diagnostics)
    if any(
        not isinstance(diagnostic, SmileSelectionDiagnostic)
        for diagnostic in diagnostic_tuple
    ):
        raise ValueError(
            "diagnostics must contain SmileSelectionDiagnostic objects",
        )
    return tuple(
        _diagnostic_record(diagnostic)
        for diagnostic in sorted(
            diagnostic_tuple,
            key=lambda item: item.sort_key,
        )
    )


def smile_selection_diagnostics_to_dataframe(
    diagnostics: Iterable[SmileSelectionDiagnostic],
):
    pandas = require_pandas()
    return pandas.DataFrame.from_records(
        smile_selection_diagnostics_to_records(diagnostics),
        columns=SMILE_SELECTION_DIAGNOSTIC_COLUMNS,
    )


def smile_group_diagnostics_to_records(
    diagnostics: Iterable[SmileGroupDiagnostic],
) -> tuple[dict[str, Any], ...]:
    diagnostic_tuple = tuple(diagnostics)
    if any(
        not isinstance(diagnostic, SmileGroupDiagnostic)
        for diagnostic in diagnostic_tuple
    ):
        raise ValueError(
            "diagnostics must contain SmileGroupDiagnostic objects",
        )
    return tuple(
        _group_diagnostic_record(diagnostic)
        for diagnostic in sorted(
            diagnostic_tuple,
            key=lambda item: item.sort_key,
        )
    )


def smile_group_diagnostics_to_dataframe(
    diagnostics: Iterable[SmileGroupDiagnostic],
):
    pandas = require_pandas()
    return pandas.DataFrame.from_records(
        smile_group_diagnostics_to_records(diagnostics),
        columns=SMILE_GROUP_DIAGNOSTIC_COLUMNS,
    )


def _build_group_smile(
    group_key: tuple[str, datetime, date],
    iv_quotes: list[ImpliedVolatilityQuote],
    config: SmileSelectionConfig,
) -> tuple[
    VolatilitySmile | None,
    tuple[SmileSelectionDiagnostic, ...],
    SmileGroupDiagnostic | None,
]:
    ordered_quotes = sorted(iv_quotes, key=lambda quote: quote.sort_key)
    diagnostics: list[SmileSelectionDiagnostic] = []
    eligible: list[ImpliedVolatilityQuote] = []

    for iv_quote in ordered_quotes:
        reasons = _initial_selection_reasons(iv_quote, config)
        if reasons:
            diagnostics.append(
                SmileSelectionDiagnostic(iv_quote, config.iv_source, reasons),
            )
        else:
            eligible.append(iv_quote)

    latest_candidates: list[ImpliedVolatilityQuote] = []
    by_contract: dict[tuple[float, OptionType], list[ImpliedVolatilityQuote]] = (
        defaultdict(list)
    )
    for iv_quote in eligible:
        contract = iv_quote.enriched_quote.quote.contract
        by_contract[(contract.strike, contract.option_type)].append(iv_quote)
    for candidates in by_contract.values():
        ordered = sorted(candidates, key=_latest_quote_rank)
        latest_candidates.append(ordered[0])
        diagnostics.extend(
            SmileSelectionDiagnostic(
                candidate,
                config.iv_source,
                (SmileSelectionReason.STALE_QUOTE,),
            )
            for candidate in ordered[1:]
        )

    inconsistent_fields = _inconsistent_market_state_fields(
        latest_candidates,
        config,
    )
    if inconsistent_fields:
        return (
            None,
            tuple(diagnostics),
            SmileGroupDiagnostic(
                underlying_symbol=group_key[0],
                valuation_timestamp=group_key[1],
                expiration=group_key[2],
                reason=(
                    SmileGroupDiagnosticReason.INCONSISTENT_MARKET_STATE
                ),
                inconsistent_fields=inconsistent_fields,
                iv_quotes=tuple(latest_candidates),
            ),
        )

    smile_candidates: list[ImpliedVolatilityQuote] = []
    for iv_quote in latest_candidates:
        if config.otm_only and not _is_otm(
            iv_quote,
            config.atm_log_moneyness_tolerance,
        ):
            diagnostics.append(
                SmileSelectionDiagnostic(
                    iv_quote,
                    config.iv_source,
                    (SmileSelectionReason.NOT_OTM,),
                ),
            )
        else:
            smile_candidates.append(iv_quote)

    selected: list[ImpliedVolatilityQuote] = []
    by_strike: dict[float, list[ImpliedVolatilityQuote]] = defaultdict(list)
    for iv_quote in smile_candidates:
        by_strike[iv_quote.enriched_quote.quote.contract.strike].append(iv_quote)
    for candidates in by_strike.values():
        ordered = sorted(
            candidates,
            key=lambda candidate: _duplicate_strike_rank(candidate, config),
        )
        selected.append(ordered[0])
        diagnostics.extend(
            SmileSelectionDiagnostic(
                candidate,
                config.iv_source,
                (SmileSelectionReason.DUPLICATE_STRIKE,),
            )
            for candidate in ordered[1:]
        )

    points = tuple(
        sorted(
            (_build_point(iv_quote, config.iv_source) for iv_quote in selected),
            key=lambda point: point.sort_key,
        )
    )
    nearest_atm_index = (
        None
        if not points
        else min(
            range(len(points)),
            key=lambda index: (
                abs(points[index].log_forward_moneyness),
                points[index].strike,
                points[index].option_type.value,
            ),
        )
    )
    metadata_source = (
        latest_candidates[0].enriched_quote
        if latest_candidates
        else ordered_quotes[0].enriched_quote
    )
    return (
        VolatilitySmile(
            underlying_symbol=group_key[0],
            valuation_timestamp=group_key[1],
            expiration=group_key[2],
            time_to_maturity=metadata_source.time_to_maturity,
            spot_price=metadata_source.spot_price,
            forward_price=metadata_source.forward_price,
            points=points,
            nearest_atm_index=nearest_atm_index,
            atm_log_moneyness_tolerance=(
                config.atm_log_moneyness_tolerance
            ),
            market_state_relative_tolerance=(
                config.market_state_relative_tolerance
            ),
            market_state_absolute_tolerance=(
                config.market_state_absolute_tolerance
            ),
        ),
        tuple(diagnostics),
        None,
    )


def _initial_selection_reasons(
    iv_quote: ImpliedVolatilityQuote,
    config: SmileSelectionConfig,
) -> tuple[SmileSelectionReason, ...]:
    enriched = iv_quote.enriched_quote
    quote = enriched.quote
    result = _source_result(iv_quote, config.iv_source)
    reasons: list[SmileSelectionReason] = []

    if config.require_two_sided_quote:
        if quote.bid is None:
            reasons.append(SmileSelectionReason.MISSING_BID)
        if quote.ask is None:
            reasons.append(SmileSelectionReason.MISSING_ASK)
    if result.status is ImpliedVolatilityStatus.FAILED:
        reasons.append(SmileSelectionReason.FAILED_IV)
    if (
        result.implied_volatility is not None
        and not isfinite(result.implied_volatility)
        and not (
            ImpliedVolatilityDiagnosticFlag.UPPER_BOUND_IV
            in result.diagnostic_flags
            and ImpliedVolatilityDiagnosticFlag.UPPER_BOUND_IV
            in config.excluded_diagnostic_flags
        )
    ):
        reasons.append(SmileSelectionReason.NON_FINITE_IV)
    for flag in config.excluded_diagnostic_flags:
        if flag in result.diagnostic_flags:
            reasons.append(_FLAG_REASONS[flag])
    if not _passes_liquidity_filters(iv_quote, config):
        reasons.append(SmileSelectionReason.LIQUIDITY_FILTER)
    return tuple(dict.fromkeys(reasons))


_MARKET_STATE_FIELDS = (
    "time_to_maturity",
    "spot_price",
    "forward_price",
    "risk_free_discount_factor",
    "dividend_discount_factor",
)


def _validate_smile_point_group_identity(
    smile: VolatilitySmile,
    points: tuple[VolatilitySmilePoint, ...],
) -> None:
    for point in points:
        enriched = point.iv_quote.enriched_quote
        contract = enriched.quote.contract
        if (
            contract.underlying_symbol != smile.underlying_symbol
            or enriched.valuation_timestamp != smile.valuation_timestamp
            or contract.expiration != smile.expiration
        ):
            raise ValueError("points must belong to the smile group")


def _validate_smile_point_market_state(
    smile: VolatilitySmile,
    points: tuple[VolatilitySmilePoint, ...],
) -> None:
    if not points:
        return
    first = points[0].iv_quote.enriched_quote
    for point in points:
        enriched = point.iv_quote.enriched_quote
        values = (
            (enriched.time_to_maturity, smile.time_to_maturity),
            (enriched.spot_price, smile.spot_price),
            (enriched.forward_price, smile.forward_price),
            (
                enriched.risk_free_discount_factor,
                first.risk_free_discount_factor,
            ),
            (
                enriched.dividend_discount_factor,
                first.dividend_discount_factor,
            ),
        )
        if any(
            not isclose(
                value,
                reference,
                rel_tol=smile.market_state_relative_tolerance,
                abs_tol=smile.market_state_absolute_tolerance,
            )
            for value, reference in values
        ):
            raise ValueError("points must share the smile market state")


def _inconsistent_market_state_fields(
    iv_quotes: list[ImpliedVolatilityQuote],
    config: SmileSelectionConfig,
) -> tuple[str, ...]:
    if not iv_quotes:
        return ()
    reference = iv_quotes[0].enriched_quote
    inconsistent: list[str] = []
    for field_name in _MARKET_STATE_FIELDS:
        reference_value = getattr(reference, field_name)
        if any(
            not isclose(
                getattr(iv_quote.enriched_quote, field_name),
                reference_value,
                rel_tol=config.market_state_relative_tolerance,
                abs_tol=config.market_state_absolute_tolerance,
            )
            for iv_quote in iv_quotes[1:]
        ):
            inconsistent.append(field_name)
    return tuple(inconsistent)


def _is_otm(iv_quote: ImpliedVolatilityQuote, tolerance: float) -> bool:
    enriched = iv_quote.enriched_quote
    option_type = enriched.quote.contract.option_type
    if abs(enriched.log_moneyness) <= tolerance:
        return True
    if option_type is OptionType.CALL:
        return enriched.log_moneyness > 0.0
    return enriched.log_moneyness < 0.0


def _passes_liquidity_filters(
    iv_quote: ImpliedVolatilityQuote,
    config: SmileSelectionConfig,
) -> bool:
    enriched = iv_quote.enriched_quote
    quote = enriched.quote
    if config.max_relative_spread is not None and (
        enriched.relative_spread is None
        or enriched.relative_spread > config.max_relative_spread
    ):
        return False
    for value, minimum in (
        (quote.bid_size, config.min_bid_size),
        (quote.ask_size, config.min_ask_size),
        (quote.session_volume, config.min_session_volume),
        (quote.open_interest, config.min_open_interest),
    ):
        if minimum is not None and (value is None or value < minimum):
            return False
    return True


def _latest_quote_rank(iv_quote: ImpliedVolatilityQuote) -> tuple:
    quote = iv_quote.enriched_quote.quote
    return (-quote.quote_timestamp.timestamp(),) + _liquidity_rank(iv_quote)


def _duplicate_strike_rank(
    iv_quote: ImpliedVolatilityQuote,
    config: SmileSelectionConfig,
) -> tuple:
    option_type = iv_quote.enriched_quote.quote.contract.option_type
    policy = config.duplicate_strike_policy
    if policy is DuplicateStrikePolicy.PREFER_CALL:
        preference = 0 if option_type is OptionType.CALL else 1
    elif policy is DuplicateStrikePolicy.PREFER_PUT:
        preference = 0 if option_type is OptionType.PUT else 1
    elif policy is DuplicateStrikePolicy.PREFER_OTM:
        preference = 0 if _is_otm(
            iv_quote,
            config.atm_log_moneyness_tolerance,
        ) else 1
    else:
        preference = 0
    return (preference,) + _liquidity_rank(iv_quote)


def _liquidity_rank(iv_quote: ImpliedVolatilityQuote) -> tuple:
    enriched = iv_quote.enriched_quote
    quote = enriched.quote
    depth = min(quote.bid_size or 0, quote.ask_size or 0)
    return (
        enriched.relative_spread if enriched.relative_spread is not None else inf,
        -depth,
        -(quote.session_volume or 0),
        -(quote.open_interest or 0),
        -quote.quote_timestamp.timestamp(),
        quote.contract.option_type.value,
        quote.sort_key,
    )


def _build_point(
    iv_quote: ImpliedVolatilityQuote,
    iv_source: SmileIvSource,
) -> VolatilitySmilePoint:
    result = _source_result(iv_quote, iv_source)
    if result.implied_volatility is None:
        raise ValueError("selected IV result must contain implied volatility")
    delta, delta_flag = _calculate_delta(
        iv_quote,
        result.implied_volatility,
    )
    iv_spread = _iv_bid_ask_spread(iv_quote)
    flags: list[SmilePointDiagnosticFlag] = []
    if delta_flag is not None:
        flags.append(delta_flag)
    if iv_spread is None:
        flags.append(SmilePointDiagnosticFlag.IV_SPREAD_UNAVAILABLE)
    return VolatilitySmilePoint(
        iv_quote=iv_quote,
        iv_source=iv_source,
        implied_volatility=result.implied_volatility,
        vega=result.vega,
        delta=delta,
        iv_bid_ask_spread=iv_spread,
        diagnostic_flags=tuple(flags),
    )


def _calculate_delta(
    iv_quote: ImpliedVolatilityQuote,
    implied_volatility: float,
) -> tuple[float | None, SmilePointDiagnosticFlag | None]:
    enriched = iv_quote.enriched_quote
    if (
        enriched.time_to_maturity <= 0.0
        or implied_volatility <= 0.0
        or not isfinite(implied_volatility)
    ):
        return None, SmilePointDiagnosticFlag.DELTA_BOUNDARY_UNAVAILABLE
    rate = -log(enriched.risk_free_discount_factor) / enriched.time_to_maturity
    dividend_yield = (
        -log(enriched.dividend_discount_factor) / enriched.time_to_maturity
    )
    function = (
        call_delta
        if enriched.quote.contract.option_type is OptionType.CALL
        else put_delta
    )
    try:
        delta = function(
            enriched.spot_price,
            enriched.quote.contract.strike,
            enriched.time_to_maturity,
            rate,
            implied_volatility,
            dividend_yield,
        )
    except (ArithmeticError, ValueError):
        return None, SmilePointDiagnosticFlag.DELTA_NUMERICAL_FAILURE
    if not isfinite(delta):
        return None, SmilePointDiagnosticFlag.DELTA_NUMERICAL_FAILURE
    return delta, None


def _iv_bid_ask_spread(iv_quote: ImpliedVolatilityQuote) -> float | None:
    bid_iv = iv_quote.bid.implied_volatility
    ask_iv = iv_quote.ask.implied_volatility
    if (
        bid_iv is None
        or ask_iv is None
        or not isfinite(bid_iv)
        or not isfinite(ask_iv)
    ):
        return None
    spread = ask_iv - bid_iv
    return spread if spread >= 0.0 else None


def _source_result(
    iv_quote: ImpliedVolatilityQuote,
    iv_source: SmileIvSource,
) -> ImpliedVolatilityResult:
    if iv_source is SmileIvSource.BID:
        return iv_quote.bid
    if iv_source is SmileIvSource.ASK:
        return iv_quote.ask
    return iv_quote.midpoint


def _point_record(
    smile: VolatilitySmile,
    point: VolatilitySmilePoint,
    is_nearest_atm: bool,
    is_observed_atm: bool,
) -> dict[str, Any]:
    iv_quote = point.iv_quote
    enriched = iv_quote.enriched_quote
    quote = enriched.quote
    source = _source_result(iv_quote, point.iv_source)
    return {
        "underlying_symbol": smile.underlying_symbol,
        "valuation_timestamp": smile.valuation_timestamp,
        "expiration": smile.expiration,
        "time_to_maturity": smile.time_to_maturity,
        "spot_price": smile.spot_price,
        "forward_price": smile.forward_price,
        "atm_log_moneyness_tolerance": smile.atm_log_moneyness_tolerance,
        "market_state_relative_tolerance": (
            smile.market_state_relative_tolerance
        ),
        "market_state_absolute_tolerance": (
            smile.market_state_absolute_tolerance
        ),
        "quote_timestamp": quote.quote_timestamp,
        "strike": point.strike,
        "option_type": point.option_type.value,
        "iv_source": point.iv_source.value,
        "implied_volatility": point.implied_volatility,
        "vega": point.vega,
        "delta": point.delta,
        "bid_iv": iv_quote.bid.implied_volatility,
        "midpoint_iv": iv_quote.midpoint.implied_volatility,
        "ask_iv": iv_quote.ask.implied_volatility,
        "iv_bid_ask_spread": point.iv_bid_ask_spread,
        "spot_moneyness": enriched.spot_moneyness,
        "forward_moneyness": enriched.forward_moneyness,
        "log_forward_moneyness": enriched.log_moneyness,
        "relative_spread": enriched.relative_spread,
        "bid_size": quote.bid_size,
        "ask_size": quote.ask_size,
        "session_volume": quote.session_volume,
        "open_interest": quote.open_interest,
        "source_status": source.status.value,
        "source_failure_reason": _enum_value(source.failure_reason),
        "source_diagnostic_flags": _flags_value(source.diagnostic_flags),
        "point_diagnostic_flags": _point_flags_value(point.diagnostic_flags),
        "is_nearest_atm": is_nearest_atm,
        "is_observed_atm": is_observed_atm,
    }


def _diagnostic_record(
    diagnostic: SmileSelectionDiagnostic,
) -> dict[str, Any]:
    iv_quote = diagnostic.iv_quote
    enriched = iv_quote.enriched_quote
    quote = enriched.quote
    contract = quote.contract
    source = diagnostic.source_result
    return {
        "underlying_symbol": contract.underlying_symbol,
        "valuation_timestamp": enriched.valuation_timestamp,
        "expiration": contract.expiration,
        "quote_timestamp": quote.quote_timestamp,
        "strike": contract.strike,
        "option_type": contract.option_type.value,
        "iv_source": diagnostic.iv_source.value,
        "reasons": "|".join(reason.value for reason in diagnostic.reasons),
        "source_status": source.status.value,
        "source_failure_reason": _enum_value(source.failure_reason),
        "source_diagnostic_flags": _flags_value(source.diagnostic_flags),
    }


def _group_diagnostic_record(
    diagnostic: SmileGroupDiagnostic,
) -> dict[str, Any]:
    return {
        "underlying_symbol": diagnostic.underlying_symbol,
        "valuation_timestamp": diagnostic.valuation_timestamp,
        "expiration": diagnostic.expiration,
        "reason": diagnostic.reason.value,
        "inconsistent_fields": "|".join(diagnostic.inconsistent_fields),
        "quote_count": diagnostic.quote_count,
    }


def _flags_value(flags: tuple[ImpliedVolatilityDiagnosticFlag, ...]) -> str:
    return "|".join(flag.value for flag in flags)


def _point_flags_value(flags: tuple[SmilePointDiagnosticFlag, ...]) -> str:
    return "|".join(flag.value for flag in flags)


def _enum_value(value: Enum | None) -> str | None:
    return None if value is None else value.value


def _validate_non_negative_finite(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    if not isfinite(float(value)) or value < 0.0:
        raise ValueError(f"{field_name} must be non-negative and finite")


def _validate_positive_finite(value: float, field_name: str) -> None:
    _validate_non_negative_finite(value, field_name)
    if value <= 0.0:
        raise ValueError(f"{field_name} must be positive")


def _validate_optional_non_negative_finite(
    value: float | None,
    field_name: str,
) -> None:
    if value is not None:
        _validate_non_negative_finite(value, field_name)


def _validate_optional_finite(
    value: float | None,
    field_name: str,
) -> None:
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(float(value))
    ):
        raise ValueError(f"{field_name} must be finite or None")


def _validate_optional_non_negative_int(
    value: int | None,
    field_name: str,
) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value < 0
    ):
        raise ValueError(f"{field_name} must be a non-negative integer or None")
