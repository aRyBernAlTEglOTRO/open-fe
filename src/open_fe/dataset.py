from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from open_fe.constant import ValueType
from open_fe.ops import Operator

__all__ = ["Dataset"]

DEFAULT_ORDINAL_THRESHOLD = 100
DUMMY_DATA = pd.DataFrame()

Config = dict[str, Any]
Feature = Operator


class Dataset:
    """Data structure for storing information about dataset."""

    def _check_target_column(self) -> None:
        if self._target_column and self._target_column not in self.columns:
            raise ValueError(f"Target column {self._target_column} not found in dataset")

    def _load_data(self) -> None:
        from open_fe.data_loader import DataLoader

        self._data = DataLoader.load(self._file_path)

    def _setup(self) -> None:
        self._load_data()
        self._check_target_column()

    def __init__(self, file_path: str, target_column: str = "") -> None:
        self._file_path: str = file_path
        self._data: pd.DataFrame = DUMMY_DATA
        self._data_snapshot: pd.DataFrame | None = None
        self._category_columns: list[str] = []
        self._numeric_columns: list[str] = []
        self._target_column: str = target_column
        self._data_blocks: list[pd.DataFrame] = []
        self._setup()

    @classmethod
    def build_from_config(cls, cfg: Config) -> Dataset:
        """Build a dataset from a given dataset config. The format of config is bellow:
        {
            "file_path": "the/input/path/to/dataset.csv",
            "target_column"(optional, only need for train dataset): "target",
            "unwanted_columns"(optional): ["unwanted_column_1", "unwanted_column_2"],
            "is_submit": False (only True for submit dataset)
        }
        Parameters
        ----------
        params_dict : Config
            parameters dictionary.
        Returns
        -------
        Dataset
            data structure for storing the information about the dataset.
        """
        # 1.load dataset.
        dataset = cls(cfg["file_path"], cfg.get("target_column", ""))

        # if this is a config for submit, then just return.
        if cfg["is_submit"]:
            return dataset

        # 2.drop unwanted columns if necessary.
        if cfg.get("unwanted_columns") is not None:
            dataset._drop(cfg["unwanted_columns"])

        return dataset

    def save(self, file_path: str) -> None:
        from open_fe.data_saver import DataSaver

        DataSaver.save(self._data, file_path)

    def generate_features(self, features: list[Feature]) -> None:
        # If we don't have the snapshot of original data, then we need to deep copy one
        # because the generated feature will change the original data. We need to make
        # sure that each time we call this function, features are generated from the same data.
        if self._data_snapshot is None:
            self._data_snapshot = self._data.copy()
        feature_value_list = [feature.calculate_base_on(self._data_snapshot) for feature in features]
        if self._target_column:
            target_value = self._data[self._target_column]
            self._data = pd.concat([*feature_value_list, target_value], axis=1)
        else:
            self._data = pd.concat(feature_value_list, axis=1)

    # TODO: add `StratifiedKFold` to get better split dataset?
    def split_into_blocks(self, data_blocks_number: int) -> None:
        data = self._data.copy()
        old_data_size = len(data)
        new_data_size = old_data_size // data_blocks_number * data_blocks_number
        data = data[:new_data_size]
        self._data_blocks = np.array_split(data, data_blocks_number)  # type:ignore

    def merge_k_blocks(self, block_number: int) -> Dataset:
        dataset_copy = deepcopy(self)
        new_inner_data = pd.concat(self._data_blocks[:block_number])
        dataset_copy._data = new_inner_data
        return dataset_copy

    def apply(self, features: list[Operator]) -> Dataset:
        self.generate_features(features)
        return self

    def get_feature_value_type(self, feature_name: str) -> ValueType:
        return ValueType.CAT if self._data[feature_name].dtype in ["object", "category"] else ValueType.NUM

    @property
    def size(self) -> int:
        return len(self._data)

    def _parse_column_type(self) -> None:
        self._category_columns = [
            col
            for col in self.columns
            if self._data[col].dtype in ["category", "object"] and col != self._target_column
        ]
        self._numeric_columns = [
            col for col in self.columns if col not in self._category_columns and col != self._target_column
        ]

    @property
    def category_columns(self) -> list[str]:
        self._parse_column_type()
        return self._category_columns

    @property
    def numeric_columns(self) -> list[str]:
        self._parse_column_type()
        return self._numeric_columns

    @property
    def features(self) -> pd.DataFrame:
        return self._data[self.feature_columns].copy()

    @property
    def target(self) -> pd.Series:
        if not self._target_column:
            raise ValueError("No target column")
        return self._data[self._target_column].copy()

    @property
    def feature_columns(self) -> list[str]:
        return self.category_columns + self.numeric_columns

    @property
    def columns(self) -> list[str]:
        return self._data.columns.tolist()

    def _drop(self, columns: list[str]) -> None:
        self._data = self._data.drop(columns, axis=1)

    def __setitem__(self, key: str, value: ArrayLike) -> None:
        self._data[key] = value
