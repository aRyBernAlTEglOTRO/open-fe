from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from open_fe.constant import OperatorType, ValueType


# TODO: add cache to all `Operator`
class Operator(ABC):
    value_type = ValueType.UNKNOWN

    @abstractmethod
    def calculate_base_on(self, df: pd.DataFrame) -> pd.Series:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def __call__(self, *args, **kwargs) -> Any:  # type: ignore
        ...


class UnaryOperator(Operator):
    def __init__(self, operand: Operator) -> None:
        self._operand = operand

    def calculate_base_on(self, df: pd.DataFrame) -> pd.Series:
        feature = self.generate(self._operand.calculate_base_on(df))
        feature.name = self.name
        return feature

    @staticmethod
    @abstractmethod
    def generate(feature: pd.Series) -> pd.Series:
        ...

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__} {self._operand.name}"


class BinaryOperator(Operator):
    def __init__(self, operand_1: Operator, operand_2: Operator) -> None:
        self._operand_1 = operand_1
        self._operand_2 = operand_2

    def calculate_base_on(self, df: pd.DataFrame) -> pd.Series:
        feature = self.generate(self._operand_1.calculate_base_on(df), self._operand_2.calculate_base_on(df))
        feature.name = self.name
        return feature

    @staticmethod
    @abstractmethod
    def generate(feature_1: pd.Series, feature_2: pd.Series) -> pd.Series:
        ...

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__} {self._operand_1.name} {self._operand_2.name}"


class OperatorCenter:
    _supported_operators: dict[OperatorType, list[Operator]] = defaultdict(list)

    @classmethod
    def register(cls, operator_type: OperatorType) -> Callable:
        def decorator(cls_: Operator) -> Operator:
            cls._supported_operators[operator_type].append(cls_)
            return cls_

        return decorator

    @classmethod
    def supported_operators(cls) -> dict[OperatorType, list[Operator]]:
        return cls._supported_operators


class Column(Operator):
    def __init__(self, column_name: str) -> None:
        self._column_name = column_name

    def calculate_base_on(self, df: pd.DataFrame) -> pd.Series:
        return df[self._column_name]  # type: ignore

    @property
    def name(self) -> str:
        return self._column_name


@OperatorCenter.register(OperatorType.ALL)
class Freq(UnaryOperator):
    value_type = ValueType.NUM

    @staticmethod
    def generate(feature: pd.Series) -> pd.Series:
        return feature.map(feature.value_counts())


@OperatorCenter.register(OperatorType.NUM)
class Abs(UnaryOperator):
    value_type = ValueType.NUM

    @staticmethod
    def generate(feature: pd.Series) -> pd.Series:
        return feature.abs()


@OperatorCenter.register(OperatorType.NUM)
class Log(UnaryOperator):
    value_type = ValueType.NUM

    @staticmethod
    def generate(feature: pd.Series) -> pd.Series:
        return np.log(feature.abs().replace(0, np.nan))  # type: ignore


@OperatorCenter.register(OperatorType.NUM)
class Sqrt(UnaryOperator):
    value_type = ValueType.NUM

    @staticmethod
    def generate(feature: pd.Series) -> pd.Series:
        return np.sqrt(feature.abs())  # type: ignore


@OperatorCenter.register(OperatorType.NUM)
class Square(UnaryOperator):
    value_type = ValueType.NUM

    @staticmethod
    def generate(feature: pd.Series) -> pd.Series:
        return np.square(feature)  # type: ignore


@OperatorCenter.register(OperatorType.NUM)
class Sigmoid(UnaryOperator):
    value_type = ValueType.NUM

    @staticmethod
    def generate(feature: pd.Series) -> pd.Series:
        return 1 / (1 + np.exp(-feature))  # type: ignore


@OperatorCenter.register(OperatorType.NUM)
class Round(UnaryOperator):
    value_type = ValueType.NUM

    @staticmethod
    def generate(feature: pd.Series) -> pd.Series:
        return np.floor(feature)  # type: ignore


@OperatorCenter.register(OperatorType.NUM)
class Residual(UnaryOperator):
    value_type = ValueType.NUM

    @staticmethod
    def generate(feature: pd.Series) -> pd.Series:
        return feature - np.floor(feature)  # type: ignore


@OperatorCenter.register(OperatorType.NUM_NUM)
class Min(BinaryOperator):
    value_type = ValueType.NUM

    @staticmethod
    def generate(feature_1: pd.Series, feature_2: pd.Series) -> pd.Series:
        return pd.concat([feature_1, feature_2], axis=1).apply(min, axis=1)


@OperatorCenter.register(OperatorType.NUM_NUM)
class Max(BinaryOperator):
    value_type = ValueType.NUM

    @staticmethod
    def generate(feature_1: pd.Series, feature_2: pd.Series) -> pd.Series:
        return pd.concat([feature_1, feature_2], axis=1).apply(max, axis=1)


@OperatorCenter.register(OperatorType.NUM_NUM)
class Add(BinaryOperator):
    value_type = ValueType.NUM

    @staticmethod
    def generate(feature_1: pd.Series, feature_2: pd.Series) -> pd.Series:
        return feature_1 + feature_2


@OperatorCenter.register(OperatorType.NUM_NUM)
class Sub(BinaryOperator):
    value_type = ValueType.NUM

    @staticmethod
    def generate(feature_1: pd.Series, feature_2: pd.Series) -> pd.Series:
        return feature_1 - feature_2


@OperatorCenter.register(OperatorType.NUM_NUM)
class Mul(BinaryOperator):
    value_type = ValueType.NUM

    @staticmethod
    def generate(feature_1: pd.Series, feature_2: pd.Series) -> pd.Series:
        return feature_1 * feature_2


@OperatorCenter.register(OperatorType.NUM_NUM)
class Div(BinaryOperator):
    value_type = ValueType.NUM

    @staticmethod
    def generate(feature_1: pd.Series, feature_2: pd.Series) -> pd.Series:
        return feature_1 / feature_2


@OperatorCenter.register(OperatorType.CAT_NUM)
class GroupByThenMin(BinaryOperator):
    value_type = ValueType.NUM

    @staticmethod
    def generate(feature_1: pd.Series, feature_2: pd.Series) -> pd.Series:
        df = pd.concat([feature_1, feature_2], axis=1)
        df.columns = ["cat", "num"]
        return df.groupby("cat")["num"].transform(min)


@OperatorCenter.register(OperatorType.CAT_NUM)
class GroupByThenMax(BinaryOperator):
    value_type = ValueType.NUM

    @staticmethod
    def generate(feature_1: pd.Series, feature_2: pd.Series) -> pd.Series:
        df = pd.concat([feature_1, feature_2], axis=1)
        df.columns = ["cat", "num"]
        return df.groupby("cat")["num"].transform(max)


@OperatorCenter.register(OperatorType.CAT_NUM)
class GroupByThenMean(BinaryOperator):
    value_type = ValueType.NUM

    @staticmethod
    def generate(feature_1: pd.Series, feature_2: pd.Series) -> pd.Series:
        df = pd.concat([feature_1, feature_2], axis=1)
        df.columns = ["cat", "num"]
        return df.groupby("cat")["num"].transform(np.mean)


@OperatorCenter.register(OperatorType.CAT_NUM)
class GroupByThenMedian(BinaryOperator):
    value_type = ValueType.NUM

    @staticmethod
    def generate(feature_1: pd.Series, feature_2: pd.Series) -> pd.Series:
        df = pd.concat([feature_1, feature_2], axis=1)
        df.columns = ["cat", "num"]
        return df.groupby("cat")["num"].transform(np.median)


@OperatorCenter.register(OperatorType.CAT_NUM)
class GroupByThenStd(BinaryOperator):
    value_type = ValueType.NUM

    @staticmethod
    def generate(feature_1: pd.Series, feature_2: pd.Series) -> pd.Series:
        df = pd.concat([feature_1, feature_2], axis=1)
        df.columns = ["cat", "num"]
        return df.groupby("cat")["num"].transform(np.std)


@OperatorCenter.register(OperatorType.CAT_NUM)
class GroupByThenRank(BinaryOperator):
    value_type = ValueType.NUM

    @staticmethod
    def generate(feature_1: pd.Series, feature_2: pd.Series) -> pd.Series:
        df = pd.concat([feature_1, feature_2], axis=1)
        df.columns = ["cat", "num"]
        return df.groupby("cat")["num"].transform(np.argsort)


@OperatorCenter.register(OperatorType.CAT_CAT)
class Combine(BinaryOperator):
    value_type = ValueType.CAT

    @staticmethod
    def generate(feature_1: pd.Series, feature_2: pd.Series) -> pd.Series:
        return feature_1.astype(str) + "_" + feature_2.astype(str)


@OperatorCenter.register(OperatorType.CAT_CAT)
class CombineThenFreq(BinaryOperator):
    value_type = ValueType.NUM

    @staticmethod
    def generate(feature_1: pd.Series, feature_2: pd.Series) -> pd.Series:
        return Freq.generate(Combine.generate(feature_1, feature_2))


@OperatorCenter.register(OperatorType.CAT_CAT)
class GroupByThenNUnique(BinaryOperator):
    value_type = ValueType.NUM

    @staticmethod
    def generate(feature_1: pd.Series, feature_2: pd.Series) -> pd.Series:
        df = pd.concat([feature_1, feature_2], axis=1)
        df.columns = ["cat", "num"]
        return df.groupby("cat")["num"].transform(lambda x: len(np.unique(x)))
