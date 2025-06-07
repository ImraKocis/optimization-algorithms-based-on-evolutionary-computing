import numpy as np
from typing import List, Tuple


class KnapsackProblem:
    """0-1 Knapsack Problem implementation"""

    def __init__(self, weights: List[float], values: List[float], capacity: float):
        self.weights = np.array(weights)
        self.values = np.array(values)
        self.capacity = capacity
        self.num_items = len(weights)

    def fitness_function(self, genes: np.ndarray) -> float:
        """
        Calculate fitness for knapsack solution
        Genes represent binary selection: 1 = item selected, 0 = not selected
        """
        # Convert to binary (in case of floating point representation)
        selection = (genes > 0.5).astype(int)

        total_weight = np.sum(selection * self.weights)
        total_value = np.sum(selection * self.values)

        # Penalty for exceeding capacity
        if total_weight > self.capacity:
            # Heavy penalty proportional to excess weight
            penalty = (total_weight - self.capacity) * max(self.values)
            return total_value - penalty

        return total_value

    def decode_solution(self, genes: np.ndarray) -> Tuple[List[int], float, float]:
        """Decode binary genes to readable solution"""
        selection = (genes > 0.5).astype(int)
        selected_items = [i for i, selected in enumerate(selection) if selected]
        total_weight = np.sum(selection * self.weights)
        total_value = np.sum(selection * self.values)

        return selected_items, total_weight, total_value
