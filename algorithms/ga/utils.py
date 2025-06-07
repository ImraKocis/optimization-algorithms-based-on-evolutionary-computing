from enum import Enum


class SelectionMethod(Enum):
    TOURNAMENT = "tournament"
    ROULETTE_WHEEL = "roulette_wheel"


class CrossoverMethod(Enum):
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"


class MutationMethod(Enum):
    GAUSSIAN = "gaussian"
    FLIP_BIT = "flip_bit"
