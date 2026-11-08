from types import ModuleType


def require_pandas() -> ModuleType:
    """Load pandas only when a DataFrame interface is used."""
    try:
        import pandas
    except ImportError as error:
        raise RuntimeError(
            "pandas interoperability requires pandas to be installed",
        ) from error
    return pandas
