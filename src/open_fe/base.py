from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

import pandas as pd
from numpy.typing import ArrayLike

from open_fe.exceptions import ConfigError
from open_fe.split_method import SplitMethodBuilder

if TYPE_CHECKING:
    from open_fe.dataset import Dataset


__all__ = [
    "DataCenterBase",
    "ModelCenterBase",
    "TrainerCenterBase",
    "LoaderBase",
    "SaverBase",
    "TrainerBase",
]


Config = dict[str, Any]


class CenterBase(ABC):
    """Base class."""

    @classmethod
    @abstractmethod
    def parse(cls, cfg: Config) -> CenterBase:
        ...


class DataCenterBase(CenterBase):
    """Base class for data center."""

    @classmethod
    @abstractmethod
    def build(cls) -> dict[str, Any]:
        ...


# TODO: modify this to return `Model` instance instead.
class ModelCenterBase(CenterBase):
    """Base class for model center."""

    @classmethod
    @abstractmethod
    def build(cls) -> Any:
        ...


class TrainerCenterBase(CenterBase):
    """Base class for trainer center."""

    @classmethod
    @abstractmethod
    def build(cls) -> Any:
        ...


class LoaderBase(ABC):
    """Base class for loader."""

    @classmethod
    @abstractmethod
    def load(cls, file_path: str) -> pd.DataFrame:
        ...


class SaverBase(ABC):
    """Base class for saver."""

    @classmethod
    @abstractmethod
    def save(cls, data: pd.DataFrame, file_path: str) -> pd.DataFrame:
        ...


class TrainerBase(ABC):
    """Base class for trainer."""

    def _setup(self) -> None:
        self._split_method = SplitMethodBuilder.build_from_config(self._cfg["split_method"])

    def _check_config(self) -> None:
        """Check the configuration for building trainer. this function follow the following rules to check the
        configuration.

        1. check `split_method` exist.
        """
        if "split_method" not in self._cfg:
            raise ConfigError("split_method must be specified in the trainer params.")

    def __init__(self, config: Config) -> None:
        self._cfg = config
        self._model = None
        self._check_config()
        self._setup()

    @classmethod
    def build_from_config(cls, config: Config) -> TrainerBase:
        """Build a trainer from a given trainer config. The format of config is bellow:
        {
            "split_method": {
                "name": "KFold",
                "params": {
                    "n_splits": 5,
                    "random_seed": 2023,
                    "shuffle": True
                }
            }
        }
        Parameters
        ----------
        params_dict : Config
            parameters dictionary.
        Returns
        -------
        TrainerBase
            data structure for model training.
        """
        return cls(config)

    @abstractmethod
    def fit(self, dataset: Dataset) -> TrainerBase:
        ...

    @abstractmethod
    def predict(self, dataset: Dataset) -> ArrayLike:
        ...

    def fit_then_predict(self, train_dataset: Dataset, test_dataset: Dataset) -> ArrayLike:
        return self.fit(train_dataset).predict(test_dataset)

    def attach(self, model: Any) -> TrainerBase:
        self._model = model
        return self
