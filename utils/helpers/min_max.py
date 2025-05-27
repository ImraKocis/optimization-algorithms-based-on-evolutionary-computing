from typing import NamedTuple


class MinMax(NamedTuple):
    min: float
    max: float

    def contains(self, value):
        return self.min <= value <= self.max

    def restrict(self, value):
        return max(self.min, min(self.max, value))
