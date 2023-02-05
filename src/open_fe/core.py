from __future__ import annotations

import warnings
from collections.abc import Callable
from copy import deepcopy
from multiprocessing import cpu_count
from typing import Any, Union

import numpy as np
import pandas as pd
from lightgbm import early_stopping, LGBMClassifier, LGBMRegressor
from scipy.special import expit, softmax
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import BaseCrossValidator, train_test_split
from tqdm import tqdm

from open_fe.constant import Task
from open_fe.dataset import Dataset
from open_fe.exceptions import ConfigError
from open_fe.feature_generator import FeatureGenerator
from open_fe.ops import Operator, OperatorCenter
from open_fe.split_method import SplitMethodBuilder
from open_fe.utils import encode_category_features

warnings.filterwarnings("ignore")

__all__ = ["OpenFE"]

Feature = Operator
Config = dict[str, Any]
LGBMModel = Union[LGBMClassifier, LGBMRegressor]
BasePredictions = Union[pd.DataFrame, pd.Series]


class OpenFE:
    """Implementation of paper:
    `OPENFE: AUTOMATED FEATURE GENERATION BEYOND EXPERT-LEVEL PERFORMANCE`_
    .. _`OPENFE: AUTOMATED FEATURE GENERATION BEYOND EXPERT-LEVEL PERFORMANCE`:
    https://arxiv.org/pdf/2211.12507.pdf

    Parameters
    ----------
    max_order : int
        predefined max order mentioned in `Algorithm 1: OpenFE` part of the paper.
    data_block_number : int
        number of data blocks mentioned in `Algorithm 2: Successive Pruning` part of the paper.
    """

    __section__ = "algorithm"

    def __init__(
        self,
        max_order: int = 5,
        data_block_number: int = 3,
        task: Task = Task.CLASSIFICATION,
        random_state: int = 2023,
        shuffle: bool = True,
        n_splits: int = 5,
        min_candidate_features: int = 100,
    ) -> None:
        self._max_order = max_order
        self._data_blocks_number = data_block_number
        self._best_features: list[Feature] = []
        self._task = task
        self._random_state = random_state
        self._shuffle = shuffle
        self._n_splits = n_splits
        self._min_candidate_features = min_candidate_features

    @classmethod
    def _check_config(cls) -> None:
        if cls._algorithm_cfg["name"] != cls.__name__:
            raise ConfigError("Unmatched configuration for OpenFE class.")

    @classmethod
    def parse(cls, cfg: Config) -> OpenFE:
        cls._algorithm_cfg = cfg[cls.__section__]
        cls._check_config()
        return cls  # type: ignore

    @classmethod
    def build(cls) -> OpenFE:
        params = cls._algorithm_cfg["params"]
        params["task"] = Task.CLASSIFICATION if params["task"] == "classification" else Task.REGRESSION
        return cls(**params)  # type: ignore

    def _setup_metric(self, dataset: Dataset) -> None:
        if self._task == Task.CLASSIFICATION:
            target_nunique = dataset.target.nunique()
            self._metric = "binary_logloss" if target_nunique == 2 else "multi_logloss"
        else:
            self._metric = "rmse"

    def _generate_base_predictions(self, base_features: list[Feature], dataset: Dataset) -> BasePredictions:
        def get_split_method() -> BaseCrossValidator:
            return SplitMethodBuilder.build_for_task(self._task, self._n_splits, self._random_state, self._shuffle)

        def init_base_predictions(target: pd.Series) -> BasePredictions:
            target_nunique = target.nunique()
            sample_number = len(target)
            if self._task == Task.CLASSIFICATION and target_nunique > 2:
                return pd.DataFrame(np.zeros((sample_number, target_nunique)))
            return 0 * target

        def build_model_for_base_prediction() -> LGBMModel:
            params = {
                "n_estimators": 10000,
                "learning_rate": 0.1,
                "metric": self._metric,
                "seed": self._random_state,
                "n_jobs": cpu_count() // 2,
                "verbosity": -1,
            }

            return LGBMClassifier(**params) if self._task == Task.CLASSIFICATION else LGBMRegressor(**params)

        def build_callbacks() -> list[Callable]:
            callbacks = []
            callbacks.append(early_stopping(200, verbose=False))
            return callbacks

        dataset.generate_features(base_features)
        features = dataset.features
        target = dataset.target
        features = encode_category_features(features)
        split_method = get_split_method()
        base_predictions = init_base_predictions(target)
        for train_index, valid_index in split_method.split(features, target):
            x_train, x_valid = (
                features.iloc[train_index].copy(),
                features.iloc[valid_index].copy(),
            )
            y_train, y_valid = (
                target.iloc[train_index].copy(),
                target.iloc[valid_index].copy(),
            )
            model = build_model_for_base_prediction()
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_valid, y_valid)],
                callbacks=build_callbacks(),
            )
            base_predictions.iloc[valid_index] = (  # type: ignore
                model.predict_proba(x_valid, raw_score=True)  # type: ignore
                if self._task == Task.CLASSIFICATION
                else model.predict(x_valid)
            )
        return base_predictions

    def _feature_boosting(
        self,
        dataset: Dataset,
        candidate_feature: Operator,
        base_predictions: BasePredictions,
    ) -> float:
        def get_metric(y_true: pd.Series, y_pred: pd.Series) -> float:
            if self._metric == "binary_logloss":
                return log_loss(y_true, expit(y_pred))
            if self._metric == "multi_logloss":
                return log_loss(y_true, softmax(y_pred, axis=1))
            if self._metric == "rmse":
                return mean_squared_error(y_true, y_pred, squared=False)
            raise NotImplementedError(
                f"Metric {self._metric} is not supported. "
                f"Please select metric from ['binary_logloss', 'multi_logloss'"
                f"'rmse']."
            )

        def get_feature_metric(
            y_train: pd.Series,
            y_valid: pd.Series,
            init_score_train: BasePredictions,
            init_score_valid: BasePredictions,
            candidate_feature_train: pd.Series,
            candidate_feature_valid: pd.Series,
        ) -> float:
            model = build_model_for_feature_boosting()
            model.fit(
                candidate_feature_train,
                y_train,
                init_score=init_score_train,
                eval_init_score=[init_score_valid],
                eval_set=[(candidate_feature_valid, y_valid)],
                callbacks=[early_stopping(3, verbose=False)],
            )
            return model._best_score["valid_0"][self._metric]  # type: ignore

        def build_model_for_feature_boosting() -> LGBMModel:
            params = {
                "n_estimators": 100,
                "importance_type": "gain",
                "num_leaves": 16,
                "seed": 1,
                "deterministic": True,
                "metric": self._metric,
                "n_jobs": cpu_count() // 2,
                "verbosity": -1,
            }

            return LGBMClassifier(**params) if self._task == Task.CLASSIFICATION else LGBMRegressor(**params)

        # get init score
        num_samples = dataset.size
        init_score = base_predictions[:num_samples]

        # calculate feature value
        feature_value = candidate_feature.calculate_base_on(dataset.features)

        # encode feature if needed
        if feature_value.dtype in ["object", "category"]:
            feature_value = feature_value.astype("category").cat.codes

        # reshape feature value
        feature_value = feature_value.to_numpy().reshape(-1, 1)

        # split into train & valid
        (
            y_train,
            y_valid,
            feature_value_train,
            feature_value_valid,
            init_score_train,
            init_score_valid,
        ) = train_test_split(
            dataset.target,
            feature_value,
            init_score,
            test_size=0.2,
            random_state=self._random_state,
        )

        # calculate feature boosting
        init_metric = get_metric(y_valid, init_score_valid)  # type: ignore
        feature_metric = get_feature_metric(
            y_train,  # type: ignore
            y_valid,  # type: ignore
            init_score_train,  # type: ignore
            init_score_valid,  # type: ignore
            feature_value_train,  # type: ignore
            feature_value_valid,  # type: ignore
        )
        return init_metric - feature_metric

    @staticmethod
    def _delete_negative_or_same(feature_delta_pairs: list[tuple[Operator, float]]) -> list[Operator]:
        sorted_feature_delta_pairs = sorted(feature_delta_pairs, key=lambda x: x[1], reverse=True)
        sorted_feature_delta_pairs = [
            feature_delta_pair for feature_delta_pair in sorted_feature_delta_pairs if feature_delta_pair[1] > 0
        ]
        remaining_features = []
        prev_delta = None
        for feature, delta in sorted_feature_delta_pairs:
            if prev_delta is None or prev_delta != delta:
                remaining_features.append(feature)
                prev_delta = delta
        return remaining_features

    @staticmethod
    def _top_half(cur_candidate_features: list[Operator]) -> list[Operator]:
        feature_number = len(cur_candidate_features)
        return cur_candidate_features[: feature_number // 2]

    def _successive_pruning(
        self,
        candidate_features: list[Operator],
        dataset: Dataset,
        base_predictions: BasePredictions,
    ) -> list[Operator]:
        dataset.split_into_blocks(2**self._data_blocks_number)
        cur_candidate_features = deepcopy(candidate_features)
        cur_idx = 0
        while cur_idx < self._data_blocks_number + 1 and len(cur_candidate_features) >= self._min_candidate_features:
            cur_iteration_feature_delta_pairs: list[tuple[Operator, float]] = []
            cur_iteration_dataset: Dataset = dataset.merge_k_blocks(2**cur_idx)
            for candidate_feature in tqdm(cur_candidate_features, desc=f"iteration {cur_idx}"):  # type: ignore
                delta = self._feature_boosting(cur_iteration_dataset, candidate_feature, base_predictions)
                cur_iteration_feature_delta_pairs.append((candidate_feature, delta))
            cur_candidate_features = self._delete_negative_or_same(cur_iteration_feature_delta_pairs)
            cur_candidate_features = self._top_half(cur_candidate_features)
            cur_idx += 1
        return cur_candidate_features

    def _get_feature_importance(
        self,
        dataset: Dataset,
        features: list[Operator],
        base_predictions: BasePredictions,
    ) -> np.ndarray:
        def build_model_for_feature_importance() -> LGBMModel:
            params = {
                "n_estimators": 1000,
                "importance_type": "gain",
                "num_leaves": 16,
                "seed": 1,
                "metric": self._metric,
                "n_jobs": cpu_count() // 2,
                "verbosity": -1,
            }

            return LGBMClassifier(**params) if self._task == Task.CLASSIFICATION else LGBMRegressor(**params)

        # build features
        dataset.generate_features(features)
        features_df = dataset.features
        target_df = dataset.target
        features_df = encode_category_features(features_df)

        # split into train & valid
        (
            y_train,
            y_valid,
            features_value_train,
            features_value_valid,
            init_score_train,
            init_score_valid,
        ) = train_test_split(
            target_df,
            features_df,
            base_predictions,
            test_size=0.2,
            random_state=self._random_state,
        )

        model = build_model_for_feature_importance()
        model.fit(
            features_value_train,
            y_train,
            init_score=init_score_train,
            eval_init_score=[init_score_valid],
            eval_set=[(features_value_valid, y_valid)],
            callbacks=[early_stopping(50, verbose=False)],
        )
        return model.feature_importances_

    @staticmethod
    def _filter_importance(feature_importance: np.ndarray, candidate_features: list[Operator]) -> np.ndarray:
        return feature_importance[-len(candidate_features) :]

    @staticmethod
    def _sort_feature_by_importance(
        candidate_features: list[Operator], candidate_importance: np.ndarray
    ) -> list[Operator]:
        return [
            candidate_feature
            for _, candidate_feature in sorted(zip(candidate_importance, candidate_features), key=lambda x: x[0])
        ]

    def _feature_attribution(
        self,
        candidate_features: list[Operator],
        base_features: list[Feature],
        dataset: Dataset,
        base_predictions: BasePredictions,
    ) -> list[Operator]:
        feature_importance = self._get_feature_importance(dataset, base_features + candidate_features, base_predictions)
        candidate_importance = self._filter_importance(feature_importance, candidate_features)
        candidate_features = self._sort_feature_by_importance(candidate_features, candidate_importance)
        return candidate_features

    @staticmethod
    def _top_k(candidate_features: list[Operator], top_k: int = 100) -> list[Operator]:
        return candidate_features[-top_k:]

    def fit(self, dataset: Dataset) -> OpenFE:
        # pre-stage preprocessing
        self._setup_metric(dataset)

        cur_order = 1
        base_features: list[Feature] = FeatureGenerator.extract_base_features(dataset)
        operators = OperatorCenter.supported_operators()
        while cur_order < self._max_order:
            base_predictions = self._generate_base_predictions(base_features, dataset)
            candidate_features = FeatureGenerator.enumerate_candidate_features(base_features, operators)
            print(f"cur order: {cur_order}")
            print(f">>>> base features size: {len(base_features)}")
            print(f">>>> candidate features size: {len(candidate_features)}")

            # two-stage evaluation
            candidate_features = self._successive_pruning(candidate_features, dataset, base_predictions)
            print(f">>>> after successive pruning, candidate features size: {len(candidate_features)}")

            candidate_features = self._feature_attribution(candidate_features, base_features, dataset, base_predictions)
            print(f">>>> after feature attribution, candidate features size: {len(candidate_features)}")

            base_features += self._top_k(candidate_features)
            cur_order += 1

            print("")
        self._best_features = base_features
        return self

    @staticmethod
    def _calculate_feature(candidate_feature: Operator, dataset: pd.DataFrame) -> pd.Series:
        return candidate_feature.calculate_base_on(dataset)

    def transform(self, dataset: Dataset) -> Dataset:
        return dataset.apply(self._best_features)
