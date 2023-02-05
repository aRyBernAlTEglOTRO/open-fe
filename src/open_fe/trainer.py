from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any

from lightgbm import early_stopping
from numpy.typing import ArrayLike

from open_fe.base import TrainerBase
from open_fe.dataset import Dataset
from open_fe.utils import encode_category_features

__all__ = ["MLTrainer"]


class MLTrainer(TrainerBase):
    """Trainer class for machine learning model."""

    def fit(self, dataset: Dataset) -> MLTrainer:
        def build_callbacks() -> list[Callable]:
            """_summary_

            Returns
            -------
            List[Callable]
                _description_
            """
            callbacks = []
            callbacks.append(early_stopping(200, verbose=False))
            return callbacks

        self._model_dict: dict[str, Any] = {}
        features = dataset.features
        target = dataset.target
        features = encode_category_features(features)
        for idx, (train_index, valid_index) in enumerate(self._split_method.split(features, target)):
            x_train, x_valid = (
                features.iloc[train_index].copy(),
                features.iloc[valid_index].copy(),
            )
            y_train, y_valid = (
                target.iloc[train_index].copy(),
                target.iloc[valid_index].copy(),
            )
            model_copy = deepcopy(self._model)
            model_copy.fit(
                x_train,
                y_train,
                eval_set=[(x_valid, y_valid)],
                callbacks=build_callbacks(),
            )
            self._model_dict[f"fold_{idx}"] = model_copy
        return self

    def predict(self, dataset: Dataset) -> ArrayLike:
        features = encode_category_features(dataset.features)
        return sum((model.predict(features) / self._split_method.get_n_splits()) for model in self._model_dict.values())
