__all__ = ["ConfigError", "DatasetError"]


class ConfigError(Exception):
    """Error type for handling config errors."""

    ...


class DatasetError(Exception):
    """Error type for handling dataset errors."""

    ...
