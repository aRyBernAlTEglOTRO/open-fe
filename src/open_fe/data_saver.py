import pandas as pd

from open_fe.base import SaverBase
from open_fe.utils import get_extension

__all__ = ["DataSaver"]


class CsvSaver(SaverBase):
    """Data saver for reading csv files."""

    @classmethod
    def save(cls, data: pd.DataFrame, file_path: str) -> None:
        data.to_csv(file_path, index=False)


class ExcelSaver(SaverBase):
    """Data saver for saving excel files."""

    @classmethod
    def save(cls, data: pd.DataFrame, file_path: str) -> None:
        data.to_excel(file_path, index=False)


class DataSaver:
    """Route file saving request to corresponding saver."""

    _data_handler = {
        ".csv": CsvSaver,
        ".xlsx": ExcelSaver,
        ".xls": ExcelSaver,
    }

    @classmethod
    def save(cls, data: pd.DataFrame, file_path: str) -> pd.DataFrame:
        ext = get_extension(file_path)
        if ext not in cls.support_data_handlers():
            raise ValueError(f"Unsupported file type: {ext}")
        return cls._data_handler[ext].save(data, file_path)

    @classmethod
    def support_data_handlers(cls) -> list[str]:
        return list(cls._data_handler.keys())
