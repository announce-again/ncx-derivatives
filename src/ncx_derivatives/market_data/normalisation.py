from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime, timezone, tzinfo
from math import isfinite

from .models import ExerciseStyle, OptionType

DEFAULT_MISSING_VALUES = frozenset({"", "NA", "N/A", "NULL", "NONE", "NAN"})

_OPTION_TYPE_ALIASES = {
    "C": OptionType.CALL,
    "CALL": OptionType.CALL,
    "P": OptionType.PUT,
    "PUT": OptionType.PUT,
}

_EXERCISE_STYLE_ALIASES = {
    "A": ExerciseStyle.AMERICAN,
    "AMERICAN": ExerciseStyle.AMERICAN,
    "E": ExerciseStyle.EUROPEAN,
    "EUROPEAN": ExerciseStyle.EUROPEAN,
    "B": ExerciseStyle.BERMUDAN,
    "BERMUDAN": ExerciseStyle.BERMUDAN,
}


def normalise_missing_value(
    value: object,
    missing_values: Iterable[str] = DEFAULT_MISSING_VALUES,
) -> object | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    normalized_missing = {item.strip().upper() for item in missing_values}
    if stripped.upper() in normalized_missing:
        return None
    return stripped


def normalise_symbol(value: str) -> str:
    return _required_text(value, "symbol").upper()


def normalise_exchange(value: str) -> str:
    return _required_text(value, "exchange").upper()


def normalise_option_type(value: str) -> OptionType:
    text = _required_text(value, "option_type").upper()
    mapped = _OPTION_TYPE_ALIASES.get(text)
    if mapped is None:
        raise ValueError(f"invalid option type {value!r}")
    return mapped


def normalise_exercise_style(value: str | None) -> ExerciseStyle | None:
    normalized = normalise_missing_value(value)
    if normalized is None:
        return None
    if not isinstance(normalized, str):
        raise ValueError(f"invalid exercise style {value!r}")
    text = normalized.upper()
    mapped = _EXERCISE_STYLE_ALIASES.get(text)
    if mapped is None:
        raise ValueError(f"invalid exercise style {value!r}")
    return mapped


def normalise_float(value: object) -> float | None:
    normalized = normalise_missing_value(value)
    if normalized is None:
        return None
    if isinstance(normalized, bool):
        raise ValueError("float value must not be bool")
    if isinstance(normalized, (int, float)):
        parsed = float(normalized)
        if not isfinite(parsed):
            raise ValueError(f"float value must be finite: {value!r}")
        return parsed
    if not isinstance(normalized, str):
        raise ValueError(f"invalid float value {value!r}")
    try:
        parsed = float(normalized)
    except ValueError as error:
        raise ValueError(f"invalid float value {value!r}") from error
    if not isfinite(parsed):
        raise ValueError(f"float value must be finite: {value!r}")
    return parsed


def normalise_int(value: object) -> int | None:
    normalized = normalise_missing_value(value)
    if normalized is None:
        return None
    if isinstance(normalized, bool):
        raise ValueError("integer value must not be bool")
    if isinstance(normalized, int):
        return normalized
    if not isinstance(normalized, str):
        raise ValueError(f"invalid integer value {value!r}")
    try:
        return int(normalized)
    except ValueError as error:
        raise ValueError(f"invalid integer value {value!r}") from error


def normalise_contract_multiplier(value: object) -> float | None:
    parsed = normalise_float(value)
    if parsed is None:
        return None
    if parsed <= 0.0:
        raise ValueError("contract multiplier must be positive")
    return parsed


def normalise_date(value: object) -> date | None:
    normalized = normalise_missing_value(value)
    if normalized is None:
        return None
    if isinstance(value, datetime):
        raise ValueError("date value must not be datetime")
    if isinstance(value, date):
        return value
    if not isinstance(normalized, str):
        raise ValueError(f"invalid ISO date value {value!r}")
    try:
        return date.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"invalid ISO date value {value!r}") from error


def normalise_datetime_utc(
    value: object,
    assume_timezone: tzinfo = timezone.utc,
) -> datetime | None:
    text = normalise_missing_value(value)
    if text is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        if not isinstance(text, str):
            raise ValueError(f"invalid ISO datetime value {value!r}")
        normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as error:
            raise ValueError(
                f"invalid ISO datetime value {value!r}",
            ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _validate_tzinfo(assume_timezone)
        wall_time = parsed.replace(tzinfo=None)
        first = wall_time.replace(tzinfo=assume_timezone, fold=0)
        second = wall_time.replace(tzinfo=assume_timezone, fold=1)
        if first.utcoffset() != second.utcoffset():
            utc = first.astimezone(timezone.utc)
            if utc.astimezone(assume_timezone).replace(tzinfo=None) != wall_time:
                raise ValueError("nonexistent local datetime; supply an explicit UTC offset")
            raise ValueError("ambiguous local datetime; supply an explicit UTC offset")
        parsed = first
    return parsed.astimezone(timezone.utc)


def standard_option_type_value_map() -> dict[str, OptionType]:
    values: dict[str, OptionType] = {}
    for alias, option_type in _OPTION_TYPE_ALIASES.items():
        values[alias] = option_type
        values[alias.lower()] = option_type
        values[alias.capitalize()] = option_type
    return values


def standard_exercise_style_value_map() -> dict[str, ExerciseStyle]:
    values: dict[str, ExerciseStyle] = {}
    for alias, exercise_style in _EXERCISE_STYLE_ALIASES.items():
        values[alias] = exercise_style
        values[alias.lower()] = exercise_style
        values[alias.capitalize()] = exercise_style
    return values


def _required_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _validate_tzinfo(value: tzinfo) -> None:
    if not isinstance(value, tzinfo):
        raise ValueError("assume_timezone must be a tzinfo")
    probe = datetime(2000, 1, 1, tzinfo=value)
    if probe.utcoffset() is None:
        raise ValueError("assume_timezone must produce aware datetimes")
