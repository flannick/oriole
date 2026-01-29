from enum import Enum


class Action(str, Enum):
    TRAIN = "train"
    CLASSIFY = "classify"


class names:
    TRAIN = "train"
    CLASSIFY = "classify"
