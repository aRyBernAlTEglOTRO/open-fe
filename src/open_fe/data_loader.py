import pandas as pd

from open_fe.base import LoaderBase
from open_fe.utils import get_extension

__all__ = ["DataLoader"]


class CsvLoader(LoaderBase):
    """Data loader for reading csv files."""

    @classmethod
    def load(cls, file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)


class ExcelLoader(LoaderBase):
    """Data loader for reading excel files."""

    @classmethod
    def load(cls, file_path: str) -> pd.DataFrame:
        return pd.read_excel(file_path)


class DataLoader:
    """Route file loading request to corresponding loader."""

    _data_handler = {
        ".csv": CsvLoader,
        ".xlsx": ExcelLoader,
        ".xls": ExcelLoader,
    }

    @classmethod
    def load(cls, file_path: str) -> pd.DataFrame:
        ext = get_extension(file_path)
        if ext not in cls.support_data_handlers():
            raise ValueError(f"Unsupported file type: {ext}")
        return cls._data_handler[ext].load(file_path)

    @classmethod
    def support_data_handlers(cls) -> list[str]:
        return list(cls._data_handler.keys())
