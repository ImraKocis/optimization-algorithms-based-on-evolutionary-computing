from typing import Callable

import numpy as np


class FitnessTransformer:
    """Transforms minimization problems to maximization"""

    def __init__(self, minimize_function: Callable, method: str = "negative"):
        self.minimize_function = minimize_function
        self.method = method
        self.best_known_value = None

    def __call__(self, x: np.ndarray) -> float:
        """Transform minimization to maximization"""
        min_value = self.minimize_function(x)

        if self.method == "negative":
            # Simple negation - most common approach
            return -min_value
