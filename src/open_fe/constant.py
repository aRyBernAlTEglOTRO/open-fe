from enum import auto, Enum

__all__ = ["Task", "OperatorType", "ValueType"]


class Task(Enum):
    CLASSIFICATION = auto()
    REGRESSION = auto()


class OperatorType(Enum):
    ALL = auto()
    NUM = auto()
    NUM_NUM = auto()
    CAT_NUM = auto()
    CAT_CAT = auto()


class ValueType(Enum):
    NUM = auto()
    CAT = auto()
    UNKNOWN = auto()
