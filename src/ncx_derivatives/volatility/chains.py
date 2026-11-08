from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from math import isfinite, isinf, isnan, log
from numbers import Real
from typing import Any

from ncx_derivatives.greeks import vega as black_scholes_vega
from ncx_derivatives.market_data import EnrichedOptionQuote, OptionType
from ncx_derivatives.utils._optional import require_pandas

from .black_scholes import (
    ImpliedVolatilityBoundsError,
    ImpliedVolatilityConvergenceError,
    ImpliedVolatilityInputError,
    _is_close_to_price_bound,
    _no_arbitrage_bounds,
    call_implied_volatility,
    put_implied_volatility,
)


class ImpliedVolatilityStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class ImpliedVolatilityFailureReason(str, Enum):
    MISSING_PRICE = "MISSING_PRICE"
    OUTSIDE_BOUNDS = "OUTSIDE_BOUNDS"
    INVALID_INPUT = "INVALID_INPUT"
    SOLVER_FAILED = "SOLVER_FAILED"
    NON_FINITE_RESULT = "NON_FINITE_RESULT"


class ImpliedVolatilityDiagnosticFlag(str, Enum):
    LOW_VEGA = "LOW_VEGA"
    VEGA_UNAVAILABLE = "VEGA_UNAVAILABLE"
    UPPER_BOUND_IV = "UPPER_BOUND_IV"


@dataclass(frozen=True, slots=True)
class ImpliedVolatilityResult:
    price: float | None
    implied_volatility: float | None
    vega: float | None
    status: ImpliedVolatilityStatus
    failure_reason: ImpliedVolatilityFailureReason | None
    diagnostic_flags: tuple[ImpliedVolatilityDiagnosticFlag, ...] = ()

    @property
    def is_success(self) -> bool:
        return self.status is ImpliedVolatilityStatus.SUCCESS


@dataclass(frozen=True, slots=True)
class ImpliedVolatilityQuote:
    enriched_quote: EnrichedOptionQuote
    bid: ImpliedVolatilityResult
    midpoint: ImpliedVolatilityResult
    ask: ImpliedVolatilityResult

    @property
    def diagnostic_flags(self) -> tuple[ImpliedVolatilityDiagnosticFlag, ...]:
        flags: list[ImpliedVolatilityDiagnosticFlag] = []
        for result in (self.bid, self.midpoint, self.ask):
            for flag in result.diagnostic_flags:
                if flag not in flags:
                    flags.append(flag)
        return tuple(flags)

    @property
    def sort_key(self) -> tuple:
        return self.enriched_quote.quote.sort_key


@dataclass(frozen=True, slots=True)
class ImpliedVolatilityChain:
    quotes: tuple[ImpliedVolatilityQuote, ...]

    def __post_init__(self) -> None:
        for quote in self.quotes:
            if not isinstance(quote, ImpliedVolatilityQuote):
                raise ValueError("quotes must contain ImpliedVolatilityQuote objects")
        object.__setattr__(
            self,
            "quotes",
            tuple(sorted(self.quotes, key=lambda quote: quote.sort_key)),
        )

    @property
    def summary(self) -> "ImpliedVolatilityChainSummary":
        return implied_volatility_chain_summary(self)


@dataclass(frozen=True, slots=True)
class ImpliedVolatilityChainSummary:
    quote_count: int
    result_count: int
    success_count: int
    failure_count: int
    missing_price_count: int
    outside_bounds_count: int
    invalid_input_count: int
    solver_failed_count: int
    non_finite_result_count: int
    low_vega_count: int
    vega_unavailable_count: int
    upper_bound_iv_count: int


def build_implied_volatility_chain(
    enriched_quotes: Iterable[EnrichedOptionQuote],
    *,
    low_vega_threshold: float = 1e-8,
) -> ImpliedVolatilityChain:
    _validate_non_negative_finite(low_vega_threshold, "low_vega_threshold")
    quotes = tuple(enriched_quotes)
    for quote in quotes:
        if not isinstance(quote, EnrichedOptionQuote):
            raise ValueError(
                "enriched_quotes must contain EnrichedOptionQuote objects",
            )

    return ImpliedVolatilityChain(
        tuple(
            ImpliedVolatilityQuote(
                enriched_quote=quote,
                bid=_solve_quote_price(
                    quote,
                    quote.quote.bid,
                    low_vega_threshold=low_vega_threshold,
                ),
                midpoint=_solve_quote_price(
                    quote,
                    quote.midpoint,
                    low_vega_threshold=low_vega_threshold,
                ),
                ask=_solve_quote_price(
                    quote,
                    quote.quote.ask,
                    low_vega_threshold=low_vega_threshold,
                ),
            )
            for quote in quotes
        ),
    )


def implied_volatility_chain_summary(
    chain: ImpliedVolatilityChain,
) -> ImpliedVolatilityChainSummary:
    if not isinstance(chain, ImpliedVolatilityChain):
        raise ValueError("chain must be an ImpliedVolatilityChain")

    success_count = 0
    failure_reasons: Counter[ImpliedVolatilityFailureReason] = Counter()
    flags: Counter[ImpliedVolatilityDiagnosticFlag] = Counter()
    for result in _iter_results(chain):
        success_count += result.is_success
        reason = result.failure_reason
        if isinstance(reason, ImpliedVolatilityFailureReason):
            failure_reasons[reason] += 1
        if result.diagnostic_flags:
            flags.update(set(result.diagnostic_flags))

    result_count = 3 * len(chain.quotes)
    return ImpliedVolatilityChainSummary(
        quote_count=len(chain.quotes),
        result_count=result_count,
        success_count=success_count,
        failure_count=result_count - success_count,
        missing_price_count=failure_reasons[ImpliedVolatilityFailureReason.MISSING_PRICE],
        outside_bounds_count=failure_reasons[ImpliedVolatilityFailureReason.OUTSIDE_BOUNDS],
        invalid_input_count=failure_reasons[ImpliedVolatilityFailureReason.INVALID_INPUT],
        solver_failed_count=failure_reasons[ImpliedVolatilityFailureReason.SOLVER_FAILED],
        non_finite_result_count=failure_reasons[ImpliedVolatilityFailureReason.NON_FINITE_RESULT],
        low_vega_count=flags[ImpliedVolatilityDiagnosticFlag.LOW_VEGA],
        vega_unavailable_count=flags[ImpliedVolatilityDiagnosticFlag.VEGA_UNAVAILABLE],
        upper_bound_iv_count=flags[ImpliedVolatilityDiagnosticFlag.UPPER_BOUND_IV],
    )


def implied_volatility_chain_to_records(
    chain: ImpliedVolatilityChain,
) -> tuple[dict[str, Any], ...]:
    return tuple(_iter_chain_records(chain))


def _iter_chain_records(
    chain: ImpliedVolatilityChain,
) -> Iterable[dict[str, Any]]:
    if not isinstance(chain, ImpliedVolatilityChain):
        raise ValueError("chain must be an ImpliedVolatilityChain")
    return (_quote_record(quote) for quote in chain.quotes)


def implied_volatility_chain_to_dataframe(chain: ImpliedVolatilityChain):
    pandas = require_pandas()
    return pandas.DataFrame.from_records(
        implied_volatility_chain_to_records(chain),
        columns=IMPLIED_VOLATILITY_CHAIN_COLUMNS,
    )


IMPLIED_VOLATILITY_CHAIN_COLUMNS = (
    "underlying_symbol",
    "quote_timestamp",
    "expiration",
    "strike",
    "option_type",
    "bid",
    "midpoint",
    "ask",
    "time_to_maturity",
    "spot_price",
    "forward_price",
    "spot_moneyness",
    "forward_moneyness",
    "log_forward_moneyness",
    "bid_iv",
    "midpoint_iv",
    "ask_iv",
    "bid_vega",
    "midpoint_vega",
    "ask_vega",
    "bid_status",
    "midpoint_status",
    "ask_status",
    "bid_failure_reason",
    "midpoint_failure_reason",
    "ask_failure_reason",
    "bid_diagnostic_flags",
    "midpoint_diagnostic_flags",
    "ask_diagnostic_flags",
    "diagnostic_flags",
)


def _solve_quote_price(
    quote: EnrichedOptionQuote,
    price: float | None,
    *,
    low_vega_threshold: float,
) -> ImpliedVolatilityResult:
    if price is None:
        return ImpliedVolatilityResult(
            price=None,
            implied_volatility=None,
            vega=None,
            status=ImpliedVolatilityStatus.FAILED,
            failure_reason=ImpliedVolatilityFailureReason.MISSING_PRICE,
        )

    try:
        validated_price = _validate_quote_price(price)
        implied_volatility = _solve_implied_volatility(quote, validated_price)
    except ImpliedVolatilityBoundsError:
        reason = ImpliedVolatilityFailureReason.OUTSIDE_BOUNDS
    except ImpliedVolatilityInputError:
        reason = ImpliedVolatilityFailureReason.INVALID_INPUT
    except (ImpliedVolatilityConvergenceError, ArithmeticError):
        reason = ImpliedVolatilityFailureReason.SOLVER_FAILED
    else:
        reason = None

    if reason is not None:
        return ImpliedVolatilityResult(
            price=_failure_price(price),
            implied_volatility=None,
            vega=None,
            status=ImpliedVolatilityStatus.FAILED,
            failure_reason=reason,
        )

    if isnan(implied_volatility):
        return _non_finite_failure(validated_price)

    upper_bound_result = (
        implied_volatility == float("inf")
        and _is_upper_bound_iv_result(quote, validated_price)
    )
    if isinf(implied_volatility) and not upper_bound_result:
        return _non_finite_failure(validated_price)

    flags: list[ImpliedVolatilityDiagnosticFlag] = []
    option_vega, vega_unavailable = _calculate_vega(quote, implied_volatility)
    if upper_bound_result:
        flags.append(ImpliedVolatilityDiagnosticFlag.UPPER_BOUND_IV)
    if vega_unavailable:
        flags.append(ImpliedVolatilityDiagnosticFlag.VEGA_UNAVAILABLE)
    if option_vega is not None and option_vega <= low_vega_threshold:
        flags.append(ImpliedVolatilityDiagnosticFlag.LOW_VEGA)

    return ImpliedVolatilityResult(
        price=validated_price,
        implied_volatility=implied_volatility,
        vega=option_vega,
        status=ImpliedVolatilityStatus.SUCCESS,
        failure_reason=None,
        diagnostic_flags=tuple(flags),
    )


def _solve_implied_volatility(
    quote: EnrichedOptionQuote,
    price: float,
) -> float:
    rate, dividend_yield = _flat_rates_from_discounts(quote)
    if quote.quote.contract.option_type is OptionType.CALL:
        return call_implied_volatility(
            price,
            quote.spot_price,
            quote.quote.contract.strike,
            quote.time_to_maturity,
            rate,
            dividend_yield,
        )
    return put_implied_volatility(
        price,
        quote.spot_price,
        quote.quote.contract.strike,
        quote.time_to_maturity,
        rate,
        dividend_yield,
    )


def _calculate_vega(
    quote: EnrichedOptionQuote,
    implied_volatility: float,
) -> tuple[float | None, bool]:
    if isinf(implied_volatility):
        return None, False
    if implied_volatility == 0.0:
        # Boundary representation policy, not a continuous sigma -> 0 limit.
        return 0.0, False
    rate, dividend_yield = _flat_rates_from_discounts(quote)
    try:
        result = black_scholes_vega(
            quote.spot_price,
            quote.quote.contract.strike,
            quote.time_to_maturity,
            rate,
            implied_volatility,
            dividend_yield,
        )
    except ValueError:
        return None, True
    if not isfinite(result) or result < 0.0:
        return None, True
    return result, False


def _flat_rates_from_discounts(quote: EnrichedOptionQuote) -> tuple[float, float]:
    maturity = quote.time_to_maturity
    if maturity == 0.0:
        return 0.0, 0.0
    return (
        -log(quote.risk_free_discount_factor) / maturity,
        -log(quote.dividend_discount_factor) / maturity,
    )


def _validate_quote_price(price: Any) -> float:
    if isinstance(price, bool) or not isinstance(price, Real):
        raise ImpliedVolatilityInputError("price must be numeric")
    try:
        result = float(price)
    except (OverflowError, TypeError, ValueError) as error:
        raise ImpliedVolatilityInputError("price must be numeric") from error
    if not isfinite(result):
        raise ImpliedVolatilityInputError("price must be finite")
    if result < 0.0:
        raise ImpliedVolatilityInputError("price must be non-negative")
    return result


def _failure_price(price: Any) -> float | None:
    if isinstance(price, bool) or not isinstance(price, Real):
        return None
    try:
        result = float(price)
    except (OverflowError, TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def _non_finite_failure(price: float) -> ImpliedVolatilityResult:
    return ImpliedVolatilityResult(
        price=price,
        implied_volatility=None,
        vega=None,
        status=ImpliedVolatilityStatus.FAILED,
        failure_reason=ImpliedVolatilityFailureReason.NON_FINITE_RESULT,
    )


def _is_upper_bound_iv_result(
    quote: EnrichedOptionQuote,
    price: float,
) -> bool:
    if quote.time_to_maturity <= 0.0:
        return False
    rate, dividend_yield = _flat_rates_from_discounts(quote)
    contract = quote.quote.contract
    _, upper_bound = _no_arbitrage_bounds(
        contract.option_type.value,
        quote.spot_price,
        contract.strike,
        quote.time_to_maturity,
        rate,
        dividend_yield,
    )
    return _is_close_to_price_bound(price, upper_bound)


def _quote_record(quote: ImpliedVolatilityQuote) -> dict[str, Any]:
    enriched = quote.enriched_quote
    contract = enriched.quote.contract
    return {
        "underlying_symbol": contract.underlying_symbol,
        "quote_timestamp": enriched.quote.quote_timestamp,
        "expiration": contract.expiration,
        "strike": contract.strike,
        "option_type": contract.option_type.value,
        "bid": quote.bid.price,
        "midpoint": quote.midpoint.price,
        "ask": quote.ask.price,
        "time_to_maturity": enriched.time_to_maturity,
        "spot_price": enriched.spot_price,
        "forward_price": enriched.forward_price,
        "spot_moneyness": enriched.spot_moneyness,
        "forward_moneyness": enriched.forward_moneyness,
        "log_forward_moneyness": enriched.log_moneyness,
        "bid_iv": quote.bid.implied_volatility,
        "midpoint_iv": quote.midpoint.implied_volatility,
        "ask_iv": quote.ask.implied_volatility,
        "bid_vega": quote.bid.vega,
        "midpoint_vega": quote.midpoint.vega,
        "ask_vega": quote.ask.vega,
        "bid_status": quote.bid.status.value,
        "midpoint_status": quote.midpoint.status.value,
        "ask_status": quote.ask.status.value,
        "bid_failure_reason": _enum_value(quote.bid.failure_reason),
        "midpoint_failure_reason": _enum_value(quote.midpoint.failure_reason),
        "ask_failure_reason": _enum_value(quote.ask.failure_reason),
        "bid_diagnostic_flags": _flags_value(quote.bid.diagnostic_flags),
        "midpoint_diagnostic_flags": _flags_value(quote.midpoint.diagnostic_flags),
        "ask_diagnostic_flags": _flags_value(quote.ask.diagnostic_flags),
        "diagnostic_flags": _flags_value(quote.diagnostic_flags),
    }


def _iter_results(
    chain: ImpliedVolatilityChain,
) -> Iterable[ImpliedVolatilityResult]:
    for quote in chain.quotes:
        yield quote.bid
        yield quote.midpoint
        yield quote.ask


def _flags_value(flags: tuple[ImpliedVolatilityDiagnosticFlag, ...]) -> str:
    return "|".join(flag.value for flag in flags)


def _enum_value(value: Enum | None) -> str | None:
    return None if value is None else value.value


def _validate_non_negative_finite(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    if not isfinite(float(value)):
        raise ValueError(f"{field_name} must be finite")
    if value < 0.0:
        raise ValueError(f"{field_name} must be non-negative")
