from typing import Callable
import numpy as np


class Individual:

    def __init__(self, genes: np.ndarray, fitness: float = None):
        self.genes = genes.copy()
        self.fitness = fitness
        self._fitness_calculated = fitness is not None

    def calculate_fitness(self, fitness_function: Callable):
        if not self._fitness_calculated:
            self.fitness = fitness_function(self.genes)
            self._fitness_calculated = True
        return self.fitness

    def copy(self):
        return Individual(self.genes.copy(), self.fitness)

    def __str__(self):
        return f"Individual(genes={self.genes}, fitness={self.fitness})"
