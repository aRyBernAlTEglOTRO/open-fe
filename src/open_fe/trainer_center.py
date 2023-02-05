from __future__ import annotations

from typing import Any

from open_fe.base import TrainerBase, TrainerCenterBase
from open_fe.exceptions import ConfigError
from open_fe.trainer import MLTrainer

Config = dict[str, Any]

__all__ = ["TrainerCenter"]


class TrainerCenter(TrainerCenterBase):
    """Use to generate trainer from config file."""

    __section__: str = "trainer"
    _trainer_cfg: Config = {}
    _supported_trainers = {
        "MLTrainer": MLTrainer,
    }

    @classmethod
    def _check_config(cls) -> None:
        """Check the configuration for building trainer. this function follow the following rules to check the
        configuration.

        1. check whether contain the `name` attribute
        2. check whether contain the `params` attribute
        3. check whether contain the `name` attribute in supported trainers.
        """
        for section in ["name", "params"]:
            if section not in cls._trainer_cfg:
                raise ConfigError(f"missing section {section} in trainer configuration.")
        if cls._trainer_cfg["name"] not in cls._supported_trainers:
            raise ConfigError(f"trainer {cls._trainer_cfg['name']} is not supported.")

    @classmethod
    def parse(cls, cfg: Config) -> TrainerCenter:
        cls._trainer_cfg = cfg[cls.__section__]
        cls._check_config()
        return cls  # type: ignore

    @classmethod
    def build(cls) -> TrainerBase:
        name = cls._trainer_cfg["name"]
        sub_cfg = cls._trainer_cfg["params"]
        return cls._supported_trainers[name].build_from_config(sub_cfg)

    @classmethod
    def supported_trainers(cls) -> list[str]:
        return list(cls._supported_trainers.keys())
