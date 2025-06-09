from typing import Callable

import numpy as np


class FitnessTransformer:
    def __init__(self, minimize_function: Callable, method: str = "negative"):
        self.minimize_function = minimize_function
        self.method = method
        self.best_known_value = None

    def __call__(self, x: np.ndarray) -> float:
        min_value = self.minimize_function(x)

        if self.method == "negative":
            return -min_value
