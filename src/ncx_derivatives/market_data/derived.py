from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from math import isfinite, log
from typing import Protocol, runtime_checkable

from .curves import CarryAssumptions
from .models import ExerciseStyle, OptionChainSnapshot, OptionQuote, OptionType


class DayCountConvention(str, Enum):
    ACT_365F = "ACT/365F"
    ACT_360 = "ACT/360"


@runtime_checkable
class DayCount(Protocol):
    convention: DayCountConvention

    def year_fraction(self, start: date | datetime, end: date | datetime) -> float:
        ...


@dataclass(frozen=True, slots=True)
class ActualFixedDayCount:
    convention: DayCountConvention

    def __post_init__(self) -> None:
        _validate_day_count_convention(self.convention)

    def year_fraction(self, start: date | datetime, end: date | datetime) -> float:
        return year_fraction(start, end, self.convention)


@dataclass(frozen=True, slots=True)
class NoArbitrageBounds:
    lower_bound: float
    upper_bound: float

    def __post_init__(self) -> None:
        _validate_non_negative_finite(self.lower_bound, "lower_bound")
        _validate_non_negative_finite(self.upper_bound, "upper_bound")
        if self.lower_bound > self.upper_bound:
            raise ValueError("lower_bound must be less than or equal to upper_bound")


@dataclass(frozen=True, slots=True)
class EnrichedOptionQuote:
    quote: OptionQuote
    valuation_timestamp: datetime
    valuation_date: date
    spot_price: float
    time_to_maturity: float
    midpoint: float | None
    absolute_spread: float | None
    relative_spread: float | None
    risk_free_discount_factor: float
    dividend_discount_factor: float
    forward_price: float
    spot_moneyness: float
    forward_moneyness: float
    log_moneyness: float
    intrinsic_value: float
    time_value: float | None
    no_arbitrage_bounds: NoArbitrageBounds | None

    def __post_init__(self) -> None:
        if not isinstance(self.quote, OptionQuote):
            raise ValueError("quote must be an OptionQuote")
        _validate_aware_datetime(self.valuation_timestamp, "valuation_timestamp")
        _validate_date(self.valuation_date, "valuation_date")
        _validate_positive_finite(self.spot_price, "spot_price")
        _validate_non_negative_finite(self.time_to_maturity, "time_to_maturity")
        _validate_optional_finite(self.midpoint, "midpoint")
        _validate_optional_finite(self.absolute_spread, "absolute_spread")
        _validate_optional_finite(self.relative_spread, "relative_spread")
        _validate_positive_finite(
            self.risk_free_discount_factor,
            "risk_free_discount_factor",
        )
        _validate_positive_finite(
            self.dividend_discount_factor,
            "dividend_discount_factor",
        )
        _validate_positive_finite(self.forward_price, "forward_price")
        _validate_positive_finite(self.spot_moneyness, "spot_moneyness")
        _validate_positive_finite(self.forward_moneyness, "forward_moneyness")
        _validate_finite(self.log_moneyness, "log_moneyness")
        _validate_non_negative_finite(self.intrinsic_value, "intrinsic_value")
        _validate_optional_finite(self.time_value, "time_value")
        if self.no_arbitrage_bounds is not None and not isinstance(
            self.no_arbitrage_bounds,
            NoArbitrageBounds,
        ):
            raise ValueError(
                "no_arbitrage_bounds must be NoArbitrageBounds or None",
            )


def year_fraction(
    start: date | datetime,
    end: date | datetime,
    convention: DayCountConvention = DayCountConvention.ACT_365F,
) -> float:
    _validate_date_or_datetime(start, "start")
    _validate_date_or_datetime(end, "end")
    _validate_day_count_convention(convention)
    if isinstance(start, datetime) != isinstance(end, datetime):
        raise ValueError("start and end must both be dates or both be datetimes")

    if isinstance(start, datetime) and isinstance(end, datetime):
        start_offset = start.utcoffset()
        end_offset = end.utcoffset()
        assert start_offset is not None and end_offset is not None
        # Subtraction with a shared tzinfo uses wall time across DST changes.
        elapsed = end.replace(tzinfo=None) - start.replace(tzinfo=None)
        seconds = (elapsed - (end_offset - start_offset)).total_seconds()
        if seconds < 0.0:
            raise ValueError("end must be greater than or equal to start")
        days = seconds / 86_400.0
    else:
        days = (end - start).days  # type: ignore[operator]
        if days < 0:
            raise ValueError("end must be greater than or equal to start")

    denominator = 365.0 if convention is DayCountConvention.ACT_365F else 360.0
    return days / denominator


def midpoint(quote: OptionQuote) -> float | None:
    _validate_quote(quote)
    if quote.bid is None or quote.ask is None:
        return None
    return (quote.bid + quote.ask) / 2.0


def absolute_spread(quote: OptionQuote) -> float | None:
    _validate_quote(quote)
    if quote.bid is None or quote.ask is None:
        return None
    return quote.ask - quote.bid


def relative_spread(quote: OptionQuote) -> float | None:
    quote_midpoint = midpoint(quote)
    if quote_midpoint is None or quote_midpoint <= 0.0:
        return None
    quote_spread = absolute_spread(quote)
    if quote_spread is None:
        return None
    return quote_spread / quote_midpoint


def intrinsic_value(quote: OptionQuote, spot: float) -> float:
    _validate_quote(quote)
    _validate_positive_finite(spot, "spot")
    strike = quote.contract.strike
    if quote.contract.option_type is OptionType.CALL:
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


def european_no_arbitrage_bounds(
    quote: OptionQuote,
    spot: float,
    risk_free_discount_factor: float,
    dividend_discount_factor: float,
) -> NoArbitrageBounds:
    _validate_quote(quote)
    _validate_positive_finite(spot, "spot")
    _validate_positive_finite(
        risk_free_discount_factor,
        "risk_free_discount_factor",
    )
    _validate_positive_finite(
        dividend_discount_factor,
        "dividend_discount_factor",
    )

    discounted_spot = spot * dividend_discount_factor
    discounted_strike = quote.contract.strike * risk_free_discount_factor
    if quote.contract.option_type is OptionType.CALL:
        return NoArbitrageBounds(
            lower_bound=max(discounted_spot - discounted_strike, 0.0),
            upper_bound=discounted_spot,
        )
    return NoArbitrageBounds(
        lower_bound=max(discounted_strike - discounted_spot, 0.0),
        upper_bound=discounted_strike,
    )


def no_arbitrage_bounds(
    quote: OptionQuote,
    spot: float,
    risk_free_discount_factor: float,
    dividend_discount_factor: float,
) -> NoArbitrageBounds | None:
    _validate_quote(quote)
    if quote.contract.exercise_style is not ExerciseStyle.EUROPEAN:
        return None
    return european_no_arbitrage_bounds(
        quote=quote,
        spot=spot,
        risk_free_discount_factor=risk_free_discount_factor,
        dividend_discount_factor=dividend_discount_factor,
    )


def enrich_option_quote(
    quote: OptionQuote,
    valuation_timestamp: datetime,
    valuation_date: date,
    spot: float,
    carry: CarryAssumptions,
    day_count: DayCountConvention | DayCount = DayCountConvention.ACT_365F,
) -> EnrichedOptionQuote:
    _validate_quote(quote)
    _validate_aware_datetime(valuation_timestamp, "valuation_timestamp")
    _validate_date(valuation_date, "valuation_date")
    _validate_positive_finite(spot, "spot")
    if not isinstance(carry, CarryAssumptions):
        raise ValueError("carry must be CarryAssumptions")

    maturity = _year_fraction_with_day_count(
        valuation_date,
        quote.contract.expiration,
        day_count,
    )
    risk_free_discount = carry.risk_free_discount_factor(maturity)
    dividend_discount = carry.dividend_discount_factor(maturity)
    _validate_positive_finite(risk_free_discount, "risk_free_discount_factor")
    _validate_positive_finite(dividend_discount, "dividend_discount_factor")

    fwd = carry.forward_price(spot, maturity)
    mid = midpoint(quote)
    intrinsic = intrinsic_value(quote, spot)
    bounds = no_arbitrage_bounds(
        quote,
        spot,
        risk_free_discount_factor=risk_free_discount,
        dividend_discount_factor=dividend_discount,
    )

    return EnrichedOptionQuote(
        quote=quote,
        valuation_timestamp=valuation_timestamp,
        valuation_date=valuation_date,
        spot_price=spot,
        time_to_maturity=maturity,
        midpoint=mid,
        absolute_spread=absolute_spread(quote),
        relative_spread=relative_spread(quote),
        risk_free_discount_factor=risk_free_discount,
        dividend_discount_factor=dividend_discount,
        forward_price=fwd,
        spot_moneyness=spot / quote.contract.strike,
        forward_moneyness=fwd / quote.contract.strike,
        log_moneyness=log(quote.contract.strike / fwd),
        intrinsic_value=intrinsic,
        time_value=None if mid is None else mid - intrinsic,
        no_arbitrage_bounds=bounds,
    )


def enrich_option_chain_snapshot(
    snapshot: OptionChainSnapshot,
    carry: CarryAssumptions,
    valuation_date: date,
    spot: float | None = None,
    day_count: DayCountConvention | DayCount = DayCountConvention.ACT_365F,
) -> tuple[EnrichedOptionQuote, ...]:
    if not isinstance(snapshot, OptionChainSnapshot):
        raise ValueError("snapshot must be an OptionChainSnapshot")
    _validate_date(valuation_date, "valuation_date")
    resolved_spot = spot
    if resolved_spot is None:
        if snapshot.underlying_quote is None or snapshot.underlying_quote.price is None:
            raise ValueError(
                "spot must be provided when snapshot has no underlying price",
            )
        resolved_spot = snapshot.underlying_quote.price

    _validate_positive_finite(resolved_spot, "spot")
    return tuple(
        enrich_option_quote(
            quote=quote,
            valuation_timestamp=snapshot.as_of,
            valuation_date=valuation_date,
            spot=resolved_spot,
            carry=carry,
            day_count=day_count,
        )
        for quote in snapshot.quotes
    )


def _year_fraction_with_day_count(
    start: date | datetime,
    end: date | datetime,
    day_count: DayCountConvention | DayCount,
) -> float:
    if isinstance(day_count, DayCountConvention):
        return year_fraction(start, end, day_count)
    if not isinstance(day_count, DayCount):
        raise ValueError("day_count must be a DayCountConvention or DayCount")
    result = day_count.year_fraction(start, end)
    _validate_non_negative_finite(result, "time_to_maturity")
    return result


def _validate_quote(value: OptionQuote) -> None:
    if not isinstance(value, OptionQuote):
        raise ValueError("quote must be an OptionQuote")


def _validate_date_or_datetime(value: date | datetime, field_name: str) -> None:
    if not isinstance(value, (date, datetime)):
        raise ValueError(f"{field_name} must be a date or datetime")
    if isinstance(value, datetime):
        _validate_aware_datetime(value, field_name)


def _validate_date(value: date, field_name: str) -> None:
    if not isinstance(value, date) or isinstance(value, datetime):
        raise ValueError(f"{field_name} must be a date")


def _validate_aware_datetime(value: datetime, field_name: str) -> None:
    if not isinstance(value, datetime):
        raise ValueError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def _validate_day_count_convention(value: DayCountConvention) -> None:
    if not isinstance(value, DayCountConvention):
        raise ValueError("convention must be a DayCountConvention")


def _validate_finite(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    if not isfinite(float(value)):
        raise ValueError(f"{field_name} must be finite")


def _validate_optional_finite(value: float | None, field_name: str) -> None:
    if value is not None:
        _validate_finite(value, field_name)


def _validate_positive_finite(value: float, field_name: str) -> None:
    _validate_finite(value, field_name)
    if value <= 0.0:
        raise ValueError(f"{field_name} must be positive")


def _validate_non_negative_finite(value: float, field_name: str) -> None:
    _validate_finite(value, field_name)
    if value < 0.0:
        raise ValueError(f"{field_name} must be non-negative")
