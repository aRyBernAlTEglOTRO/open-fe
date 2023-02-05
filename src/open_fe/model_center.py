from __future__ import annotations

from typing import Any

from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

from open_fe.base import ModelCenterBase
from open_fe.exceptions import ConfigError

Config = dict[str, Any]

__all__ = ["ModelCenter"]

# TODO: wrap model into self define class, only supply fit and predict methods
class ModelCenter(ModelCenterBase):
    """Use to generate model from config file."""

    __section__: str = "model"
    _model_cfg: Config = {}
    _supported_models = {
        "LGBMClassifier": LGBMClassifier,
        "LGBMRegressor": LGBMRegressor,
        "XGBClassifier": XGBClassifier,
        "XGBRegressor": XGBRegressor,
    }

    @classmethod
    def _check_config(cls) -> None:
        """Check the configuration for building model. this function follow the following rules to check the
        configuration.

        1. check whether contain the `name` attribute
        2. check whether contain the `params` attribute
        3. check whether contain the `name` attribute in supported models.
        """
        for section in ["name", "params"]:
            if section not in cls._model_cfg:
                raise ConfigError(f"missing section {section} in model configuration.")
        if cls._model_cfg["name"] not in cls._supported_models:
            raise ConfigError(f"model {cls._model_cfg['name']} is not supported.")

    @classmethod
    def parse(cls, cfg: Config) -> ModelCenter:
        cls._model_cfg = cfg[cls.__section__]
        cls._check_config()
        return cls  # type: ignore

    @classmethod
    def build(cls) -> Any:
        name = cls._model_cfg["name"]
        params = cls._model_cfg["params"]
        return cls._supported_models[name](**params)

    @classmethod
    def supported_models(cls) -> list[str]:
        return list(cls._supported_models.keys())
