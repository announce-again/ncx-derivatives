from types import ModuleType


def require_pandas() -> ModuleType:
    try:
        import pandas
    except ImportError as error:
        raise RuntimeError("Research 001 requires pandas") from error
    return pandas
