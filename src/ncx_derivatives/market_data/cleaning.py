from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from math import isfinite

from .derived import EnrichedOptionQuote
from .models import OptionChainSnapshot, OptionQuote
from .validation import ValidationSeverity


class RejectionReason(str, Enum):
    MISSING_BID = "MISSING_BID"
    MISSING_ASK = "MISSING_ASK"
    EMPTY_MARKET = "EMPTY_MARKET"
    CROSSED_MARKET = "CROSSED_MARKET"
    LOCKED_MARKET = "LOCKED_MARKET"
    ZERO_MIDPOINT = "ZERO_MIDPOINT"
    EXCESSIVE_SPREAD = "EXCESSIVE_SPREAD"
    INSUFFICIENT_VOLUME = "INSUFFICIENT_VOLUME"
    INSUFFICIENT_OPEN_INTEREST = "INSUFFICIENT_OPEN_INTEREST"
    STALE_QUOTE = "STALE_QUOTE"
    MATURITY_OUT_OF_RANGE = "MATURITY_OUT_OF_RANGE"
    STRIKE_OUT_OF_RANGE = "STRIKE_OUT_OF_RANGE"
    SPOT_MONEYNESS_OUT_OF_RANGE = "SPOT_MONEYNESS_OUT_OF_RANGE"
    FORWARD_MONEYNESS_OUT_OF_RANGE = "FORWARD_MONEYNESS_OUT_OF_RANGE"
    PRICE_BELOW_MINIMUM = "PRICE_BELOW_MINIMUM"


@dataclass(frozen=True, slots=True)
class CleaningConfig:
    reject_missing_bid: bool = True
    reject_missing_ask: bool = True
    reject_crossed_market: bool = True
    reject_locked_market: bool = False
    reject_zero_midpoint: bool = True
    max_relative_spread: float | None = None
    min_volume: int | None = None
    min_open_interest: int | None = None
    max_quote_age: timedelta | None = None
    min_maturity: float | None = None
    max_maturity: float | None = None
    min_strike: float | None = None
    max_strike: float | None = None
    min_spot_moneyness: float | None = None
    max_spot_moneyness: float | None = None
    min_forward_moneyness: float | None = None
    max_forward_moneyness: float | None = None
    min_option_price: float | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "reject_missing_bid",
            "reject_missing_ask",
            "reject_crossed_market",
            "reject_locked_market",
            "reject_zero_midpoint",
        ):
            _validate_bool(getattr(self, field_name), field_name)

        _validate_optional_non_negative_finite(
            self.max_relative_spread,
            "max_relative_spread",
        )
        _validate_optional_non_negative_int(self.min_volume, "min_volume")
        _validate_optional_non_negative_int(
            self.min_open_interest,
            "min_open_interest",
        )
        _validate_optional_non_negative_timedelta(
            self.max_quote_age,
            "max_quote_age",
        )
        _validate_optional_non_negative_finite(self.min_maturity, "min_maturity")
        _validate_optional_non_negative_finite(self.max_maturity, "max_maturity")
        _validate_optional_positive_finite(self.min_strike, "min_strike")
        _validate_optional_positive_finite(self.max_strike, "max_strike")
        _validate_optional_positive_finite(
            self.min_spot_moneyness,
            "min_spot_moneyness",
        )
        _validate_optional_positive_finite(
            self.max_spot_moneyness,
            "max_spot_moneyness",
        )
        _validate_optional_positive_finite(
            self.min_forward_moneyness,
            "min_forward_moneyness",
        )
        _validate_optional_positive_finite(
            self.max_forward_moneyness,
            "max_forward_moneyness",
        )
        _validate_optional_non_negative_finite(
            self.min_option_price,
            "min_option_price",
        )
        _validate_range(self.min_maturity, self.max_maturity, "maturity")
        _validate_range(self.min_strike, self.max_strike, "strike")
        _validate_range(
            self.min_spot_moneyness,
            self.max_spot_moneyness,
            "spot_moneyness",
        )
        _validate_range(
            self.min_forward_moneyness,
            self.max_forward_moneyness,
            "forward_moneyness",
        )


@dataclass(frozen=True, slots=True)
class CleaningDiagnostic:
    severity: ValidationSeverity
    reason: RejectionReason
    message: str
    location: tuple[str, ...] = ()
    context: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.severity, ValidationSeverity):
            raise ValueError("severity must be a ValidationSeverity")
        if not isinstance(self.reason, RejectionReason):
            raise ValueError("reason must be a RejectionReason")
        _validate_non_empty_text(self.message, "message")

        location = tuple(self.location)
        for item in location:
            _validate_non_empty_text(item, "location item")

        context = tuple(self.context)
        for item in context:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("context must contain key-value string pairs")
            key, value = item
            _validate_non_empty_text(key, "context key")
            if not isinstance(value, str):
                raise ValueError("context values must be strings")

        object.__setattr__(self, "location", location)
        object.__setattr__(self, "context", context)


@dataclass(frozen=True, slots=True)
class RejectedQuote:
    quote: OptionQuote
    diagnostics: tuple[CleaningDiagnostic, ...]
    enriched_quote: EnrichedOptionQuote | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.quote, OptionQuote):
            raise ValueError("quote must be an OptionQuote")
        diagnostics = tuple(self.diagnostics)
        if not diagnostics:
            raise ValueError("diagnostics must not be empty")
        for diagnostic in diagnostics:
            if not isinstance(diagnostic, CleaningDiagnostic):
                raise ValueError(
                    "diagnostics must contain CleaningDiagnostic objects",
                )
        if self.enriched_quote is not None:
            if not isinstance(self.enriched_quote, EnrichedOptionQuote):
                raise ValueError(
                    "enriched_quote must be EnrichedOptionQuote or None",
                )
            if self.enriched_quote.quote != self.quote:
                raise ValueError("enriched_quote quote must match quote")
        object.__setattr__(self, "diagnostics", diagnostics)


@dataclass(frozen=True, slots=True)
class CleaningResult:
    accepted: tuple[OptionQuote, ...] = field(default_factory=tuple)
    rejected: tuple[RejectedQuote, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        accepted = tuple(self.accepted)
        rejected = tuple(self.rejected)
        for quote in accepted:
            if not isinstance(quote, OptionQuote):
                raise ValueError("accepted must contain OptionQuote objects")
        for rejected_quote in rejected:
            if not isinstance(rejected_quote, RejectedQuote):
                raise ValueError("rejected must contain RejectedQuote objects")
        object.__setattr__(
            self,
            "accepted",
            tuple(sorted(accepted, key=lambda quote: quote.sort_key)),
        )
        object.__setattr__(
            self,
            "rejected",
            tuple(sorted(rejected, key=lambda item: item.quote.sort_key)),
        )

    @property
    def diagnostics(self) -> tuple[CleaningDiagnostic, ...]:
        return tuple(
            diagnostic
            for rejected_quote in self.rejected
            for diagnostic in rejected_quote.diagnostics
        )

    @property
    def rejected_quotes(self) -> tuple[OptionQuote, ...]:
        return tuple(item.quote for item in self.rejected)

    @property
    def accepted_count(self) -> int:
        return len(self.accepted)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)


@dataclass(frozen=True, slots=True)
class EnrichedCleaningResult:
    accepted: tuple[EnrichedOptionQuote, ...] = field(default_factory=tuple)
    rejected: tuple[RejectedQuote, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        accepted = tuple(self.accepted)
        rejected = tuple(self.rejected)
        for quote in accepted:
            if not isinstance(quote, EnrichedOptionQuote):
                raise ValueError(
                    "accepted must contain EnrichedOptionQuote objects",
                )
        for rejected_quote in rejected:
            if not isinstance(rejected_quote, RejectedQuote):
                raise ValueError("rejected must contain RejectedQuote objects")
            if rejected_quote.enriched_quote is None:
                raise ValueError(
                    "rejected enriched results must preserve enriched_quote",
                )
        object.__setattr__(
            self,
            "accepted",
            tuple(sorted(accepted, key=lambda item: item.quote.sort_key)),
        )
        object.__setattr__(
            self,
            "rejected",
            tuple(sorted(rejected, key=lambda item: item.quote.sort_key)),
        )

    @property
    def diagnostics(self) -> tuple[CleaningDiagnostic, ...]:
        return tuple(
            diagnostic
            for rejected_quote in self.rejected
            for diagnostic in rejected_quote.diagnostics
        )

    @property
    def rejected_quotes(self) -> tuple[OptionQuote, ...]:
        return tuple(item.quote for item in self.rejected)

    @property
    def accepted_count(self) -> int:
        return len(self.accepted)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)


def clean_option_chain(
    snapshot: OptionChainSnapshot,
    config: CleaningConfig | None = None,
) -> CleaningResult:
    if not isinstance(snapshot, OptionChainSnapshot):
        raise ValueError("snapshot must be an OptionChainSnapshot")
    if config is None:
        config = CleaningConfig()
    if not isinstance(config, CleaningConfig):
        raise ValueError("config must be a CleaningConfig")
    _reject_enriched_only_config(config)

    accepted: list[OptionQuote] = []
    rejected: list[RejectedQuote] = []

    for index, quote in enumerate(snapshot.quotes):
        diagnostics = _cleaning_diagnostics_for_quote(
            quote=quote,
            config=config,
            location=("quotes", str(index)),
            snapshot=snapshot,
            enriched_quote=None,
        )
        if diagnostics:
            rejected.append(RejectedQuote(quote=quote, diagnostics=diagnostics))
        else:
            accepted.append(quote)

    return CleaningResult(
        accepted=tuple(accepted),
        rejected=tuple(rejected),
    )


def clean_enriched_option_quotes(
    enriched_quotes: tuple[EnrichedOptionQuote, ...],
    config: CleaningConfig | None = None,
) -> EnrichedCleaningResult:
    if config is None:
        config = CleaningConfig()
    if not isinstance(config, CleaningConfig):
        raise ValueError("config must be a CleaningConfig")
    enriched_items = tuple(enriched_quotes)
    for item in enriched_items:
        if not isinstance(item, EnrichedOptionQuote):
            raise ValueError(
                "enriched_quotes must contain EnrichedOptionQuote objects",
            )

    accepted: list[EnrichedOptionQuote] = []
    rejected: list[RejectedQuote] = []

    for index, item in enumerate(enriched_items):
        diagnostics = _cleaning_diagnostics_for_quote(
            quote=item.quote,
            config=config,
            location=("enriched_quotes", str(index)),
            snapshot=None,
            enriched_quote=item,
        )
        if diagnostics:
            rejected.append(
                RejectedQuote(
                    quote=item.quote,
                    diagnostics=diagnostics,
                    enriched_quote=item,
                ),
            )
        else:
            accepted.append(item)

    return EnrichedCleaningResult(
        accepted=tuple(accepted),
        rejected=tuple(rejected),
    )


def _cleaning_diagnostics_for_quote(
    quote: OptionQuote,
    config: CleaningConfig,
    location: tuple[str, ...],
    snapshot: OptionChainSnapshot | None,
    enriched_quote: EnrichedOptionQuote | None,
) -> tuple[CleaningDiagnostic, ...]:
    midpoint = _quote_midpoint(quote)
    diagnostics = _market_diagnostics(quote, config, location, midpoint, enriched_quote)

    if (
        config.min_volume is not None
        and (quote.session_volume is None or quote.session_volume < config.min_volume)
    ):
        diagnostics.append(
            _diagnostic(
                RejectionReason.INSUFFICIENT_VOLUME,
                "quote session volume is below minimum",
                location,
                context=(
                    ("session_volume", _optional_str(quote.session_volume)),
                    ("min_volume", str(config.min_volume)),
                ),
            ),
        )

    if (
        config.min_open_interest is not None
        and (
            quote.open_interest is None
            or quote.open_interest < config.min_open_interest
        )
    ):
        diagnostics.append(
            _diagnostic(
                RejectionReason.INSUFFICIENT_OPEN_INTEREST,
                "quote open interest is below minimum",
                location,
                context=(
                    ("open_interest", _optional_str(quote.open_interest)),
                    ("min_open_interest", str(config.min_open_interest)),
                ),
            ),
        )

    if config.max_quote_age is not None:
        reference_timestamp = (
            enriched_quote.valuation_timestamp if enriched_quote is not None else None
        )
        if reference_timestamp is None and snapshot is not None:
            reference_timestamp = snapshot.as_of
        if reference_timestamp is not None:
            age = reference_timestamp - quote.quote_timestamp
            if age > config.max_quote_age:
                diagnostics.append(
                    _diagnostic(
                        RejectionReason.STALE_QUOTE,
                        "quote timestamp is older than maximum age",
                        location,
                        context=(
                            ("quote_timestamp", quote.quote_timestamp.isoformat()),
                            ("reference_timestamp", reference_timestamp.isoformat()),
                            ("max_quote_age_seconds", str(config.max_quote_age.total_seconds())),
                        ),
                    ),
                )

    strike = quote.contract.strike
    if _outside_range(strike, config.min_strike, config.max_strike):
        diagnostics.append(
            _diagnostic(
                RejectionReason.STRIKE_OUT_OF_RANGE,
                "quote strike is outside configured range",
                location,
                context=(
                    ("strike", str(strike)),
                    ("min_strike", _optional_str(config.min_strike)),
                    ("max_strike", _optional_str(config.max_strike)),
                ),
            ),
        )

    option_price = enriched_quote.midpoint if enriched_quote is not None else midpoint
    if (
        config.min_option_price is not None
        and (option_price is None or option_price < config.min_option_price)
    ):
        diagnostics.append(
            _diagnostic(
                RejectionReason.PRICE_BELOW_MINIMUM,
                "quote midpoint is below minimum option price",
                location,
                context=(
                    ("midpoint", _optional_str(option_price)),
                    ("min_option_price", str(config.min_option_price)),
                ),
            ),
        )

    if enriched_quote is not None:
        diagnostics.extend(_enriched_diagnostics(enriched_quote, config, location))
    return tuple(diagnostics)


def _market_diagnostics(
    quote: OptionQuote,
    config: CleaningConfig,
    location: tuple[str, ...],
    midpoint: float | None,
    enriched_quote: EnrichedOptionQuote | None,
) -> list[CleaningDiagnostic]:
    diagnostics: list[CleaningDiagnostic] = []

    if quote.bid is None and quote.ask is None:
        diagnostics.append(
            _diagnostic(
                RejectionReason.EMPTY_MARKET,
                "quote has neither bid nor ask",
                location,
            ),
        )
    else:
        if config.reject_missing_bid and quote.bid is None:
            diagnostics.append(
                _diagnostic(RejectionReason.MISSING_BID, "quote is missing bid", location),
            )
        if config.reject_missing_ask and quote.ask is None:
            diagnostics.append(
                _diagnostic(RejectionReason.MISSING_ASK, "quote is missing ask", location),
            )

    if quote.bid is not None and quote.ask is not None:
        if config.reject_crossed_market and quote.bid > quote.ask:
            diagnostics.append(
                _diagnostic(
                    RejectionReason.CROSSED_MARKET,
                    "quote bid is greater than ask",
                    location,
                    context=(("bid", str(quote.bid)), ("ask", str(quote.ask))),
                ),
            )
        elif config.reject_locked_market and quote.bid == quote.ask:
            diagnostics.append(
                _diagnostic(
                    RejectionReason.LOCKED_MARKET,
                    "quote bid equals ask",
                    location,
                    context=(("bid", str(quote.bid)), ("ask", str(quote.ask))),
                ),
            )

    if config.reject_zero_midpoint and midpoint == 0.0:
        diagnostics.append(
            _diagnostic(
                RejectionReason.ZERO_MIDPOINT,
                "quote midpoint is zero",
                location,
            ),
        )

    max_spread = config.max_relative_spread
    spread = _relative_spread(quote, enriched_quote, midpoint) if max_spread is not None else None
    if spread is not None and max_spread is not None and spread > max_spread:
        diagnostics.append(
            _diagnostic(
                RejectionReason.EXCESSIVE_SPREAD,
                "quote relative spread exceeds maximum",
                location,
                context=(
                    ("relative_spread", str(spread)),
                    ("max_relative_spread", str(max_spread)),
                ),
            ),
        )

    return diagnostics


def _enriched_diagnostics(
    enriched_quote: EnrichedOptionQuote,
    config: CleaningConfig,
    location: tuple[str, ...],
) -> list[CleaningDiagnostic]:
    diagnostics: list[CleaningDiagnostic] = []
    if _outside_range(
        enriched_quote.time_to_maturity,
        config.min_maturity,
        config.max_maturity,
    ):
        diagnostics.append(
            _diagnostic(
                RejectionReason.MATURITY_OUT_OF_RANGE,
                "quote maturity is outside configured range",
                location,
                context=(
                    ("time_to_maturity", str(enriched_quote.time_to_maturity)),
                    ("min_maturity", _optional_str(config.min_maturity)),
                    ("max_maturity", _optional_str(config.max_maturity)),
                ),
            ),
        )

    if _outside_range(
        enriched_quote.spot_moneyness,
        config.min_spot_moneyness,
        config.max_spot_moneyness,
    ):
        diagnostics.append(
            _diagnostic(
                RejectionReason.SPOT_MONEYNESS_OUT_OF_RANGE,
                "quote spot moneyness is outside configured range",
                location,
                context=(
                    ("spot_moneyness", str(enriched_quote.spot_moneyness)),
                    ("min_spot_moneyness", _optional_str(config.min_spot_moneyness)),
                    ("max_spot_moneyness", _optional_str(config.max_spot_moneyness)),
                ),
            ),
        )

    if _outside_range(
        enriched_quote.forward_moneyness,
        config.min_forward_moneyness,
        config.max_forward_moneyness,
    ):
        diagnostics.append(
            _diagnostic(
                RejectionReason.FORWARD_MONEYNESS_OUT_OF_RANGE,
                "quote forward moneyness is outside configured range",
                location,
                context=(
                    ("forward_moneyness", str(enriched_quote.forward_moneyness)),
                    (
                        "min_forward_moneyness",
                        _optional_str(config.min_forward_moneyness),
                    ),
                    (
                        "max_forward_moneyness",
                        _optional_str(config.max_forward_moneyness),
                    ),
                ),
            ),
        )
    return diagnostics


def _diagnostic(
    reason: RejectionReason,
    message: str,
    location: tuple[str, ...],
    context: tuple[tuple[str, str], ...] = (),
) -> CleaningDiagnostic:
    return CleaningDiagnostic(
        severity=ValidationSeverity.WARNING,
        reason=reason,
        message=message,
        location=location,
        context=context,
    )


def _reject_enriched_only_config(config: CleaningConfig) -> None:
    if (
        config.min_maturity is not None
        or config.max_maturity is not None
        or config.min_spot_moneyness is not None
        or config.max_spot_moneyness is not None
        or config.min_forward_moneyness is not None
        or config.max_forward_moneyness is not None
    ):
        raise ValueError("maturity and moneyness filters require enriched quotes")


def _quote_midpoint(quote: OptionQuote) -> float | None:
    if quote.bid is None or quote.ask is None:
        return None
    return (quote.bid + quote.ask) / 2.0


def _relative_spread(
    quote: OptionQuote,
    enriched_quote: EnrichedOptionQuote | None,
    midpoint: float | None,
) -> float | None:
    if enriched_quote is not None:
        return enriched_quote.relative_spread
    if midpoint is None or midpoint <= 0.0:
        return None
    if quote.bid is None or quote.ask is None:
        return None
    return (quote.ask - quote.bid) / midpoint

def _outside_range(
    value: float,
    minimum: float | None,
    maximum: float | None,
) -> bool:
    return (minimum is not None and value < minimum) or (
        maximum is not None and value > maximum
    )


def _optional_str(value: object | None) -> str:
    return "None" if value is None else str(value)


def _validate_bool(value: bool, field_name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a bool")


def _validate_non_empty_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_optional_non_negative_int(
    value: int | None,
    field_name: str,
) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer or None")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _validate_optional_non_negative_timedelta(
    value: timedelta | None,
    field_name: str,
) -> None:
    if value is None:
        return
    if not isinstance(value, timedelta):
        raise ValueError(f"{field_name} must be a timedelta or None")
    if value.total_seconds() < 0.0:
        raise ValueError(f"{field_name} must be non-negative")


def _validate_optional_finite(value: float | None, field_name: str) -> None:
    if value is not None:
        _validate_finite(value, field_name)


def _validate_optional_non_negative_finite(
    value: float | None,
    field_name: str,
) -> None:
    _validate_optional_finite(value, field_name)
    if value is not None and value < 0.0:
        raise ValueError(f"{field_name} must be non-negative")


def _validate_optional_positive_finite(
    value: float | None,
    field_name: str,
) -> None:
    _validate_optional_finite(value, field_name)
    if value is not None and value <= 0.0:
        raise ValueError(f"{field_name} must be positive")


def _validate_finite(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    if not isfinite(float(value)):
        raise ValueError(f"{field_name} must be finite")


def _validate_range(
    minimum: float | None,
    maximum: float | None,
    field_name: str,
) -> None:
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError(f"min_{field_name} must be less than or equal to max_{field_name}")
