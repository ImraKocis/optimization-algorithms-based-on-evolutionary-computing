from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from algorithms.ga.individual import Individual


class PopulationInitializer(ABC):
    @abstractmethod
    def initialize(self, population_size: int, **kwargs) -> List[Individual]:
        pass


class RandomContinuousInitializer(PopulationInitializer):
    def __init__(self, dimensions: int, bounds: Tuple[float, float]):
        self.dimensions = dimensions
        self.lower_bound, self.upper_bound = bounds

    def initialize(self, population_size: int, **kwargs) -> List[Individual]:
        population = []
        for _ in range(population_size):
            genes = np.random.uniform(
                self.lower_bound,
                self.upper_bound,
                self.dimensions
            )
            population.append(Individual(genes))
        return population


class RandomBinaryInitializer(PopulationInitializer):
    def __init__(self, length: int):
        self.length = length

    def initialize(self, population_size: int, **kwargs) -> List[Individual]:
        population = []
        for _ in range(population_size):
            genes = np.random.randint(0, 2, self.length).astype(float)
            population.append(Individual(genes))
        return population
