from typing import Any

from sklearn.model_selection import BaseCrossValidator, KFold, StratifiedKFold

from open_fe.constant import Task
from open_fe.exceptions import ConfigError

Config = dict[str, Any]

__all__ = ["SplitMethodBuilder"]


class SplitMethodBuilder:
    _supported_methods = {
        "KFold": KFold,
        "StratifiedKFold": StratifiedKFold,
    }

    @classmethod
    def build_from_config(cls, config: Config) -> BaseCrossValidator:
        split_method: str = config["name"]
        if split_method not in cls._supported_methods:
            raise ConfigError(f"{split_method} is not supported.")
        return cls._supported_methods[config["name"]](**config["params"])

    @classmethod
    def supported_methods(cls) -> list[str]:
        return list(cls._supported_methods.keys())

    @classmethod
    def build_for_task(cls, task: Task, n_splits: int, random_state: int, shuffle: bool) -> BaseCrossValidator:
        if task == Task.CLASSIFICATION:
            return StratifiedKFold(n_splits, shuffle=shuffle, random_state=random_state)
        if task == Task.REGRESSION:
            return KFold(n_splits, shuffle=shuffle, random_state=random_state)
        raise ValueError(f"Invalid task: {task}")
