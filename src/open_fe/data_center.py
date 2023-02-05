from __future__ import annotations

from typing import Any

from open_fe.base import DataCenterBase
from open_fe.dataset import Dataset
from open_fe.exceptions import ConfigError, DatasetError

Config = dict[str, Any]
DatasetDict = dict[str, Dataset]

__all__ = ["DataCenter"]


class DataCenter(DataCenterBase):
    """Use to generate datasets from config file."""

    __section__: str = "data"
    _data_cfg: Config = {}

    @classmethod
    def _check_config(cls) -> None:
        """Check the configuration for building dataset. this function follow the following rules to check the
        configuration.

        1. check `train`, `test`, `submit` section exist.
        2. check `file_path` in previous section exist.
        3. for `train` section, check `target_column` exist
        """
        for data_type in ["train", "test", "submit"]:
            if data_type not in cls._data_cfg:
                raise ConfigError(f"missing {data_type} in config file.")
            if "file_path" not in cls._data_cfg[data_type]:
                raise ConfigError(f"missing file_path of {data_type} data.")
            if data_type == "train" and "target_column" not in cls._data_cfg[data_type]:
                raise ConfigError(f"missing target_column of {data_type} data.")

    @classmethod
    def _update_config(cls) -> None:
        """update `train`, `test` `submit` section configuration."""
        cls._data_cfg["train"].update({"is_submit": False})
        cls._data_cfg["test"].update({"is_submit": False})
        cls._data_cfg["submit"].update({"is_submit": True})

    @classmethod
    def parse(cls, cfg: Config) -> DataCenter:
        cls._data_cfg = cfg[cls.__section__]
        cls._check_config()
        cls._update_config()
        return cls  # type: ignore

    @staticmethod
    def _contain_same_features(dataset_a: Dataset, dataset_b: Dataset) -> bool:
        """Check if two datasets contain same features.

        Parameters
        ----------
        dataset_a : Dataset
            dataset for comparison
        dataset_b : Dataset
            dataset for comparison

        Returns
        -------
        bool
            boolean flag indicating whether `dataset_a` and `dataset_b` contain same features.
        """
        a_features = set(dataset_a.feature_columns)
        b_features = set(dataset_b.feature_columns)
        return not ((a_features - b_features) or (b_features - a_features))

    @classmethod
    def _check_datasets(cls, dataset_dict: DatasetDict) -> None:
        if not cls._contain_same_features(dataset_dict["train"], dataset_dict["test"]):
            raise DatasetError("some columns in train dataset and test dataset do not match.")

    @staticmethod
    def _build_dataset(cfg: Config) -> Dataset:
        return Dataset.build_from_config(cfg)

    @classmethod
    def build(cls) -> DatasetDict:
        dataset_dict = {data_type: cls._build_dataset(data_params) for data_type, data_params in cls._data_cfg.items()}
        cls._check_datasets(dataset_dict)
        return dataset_dict
