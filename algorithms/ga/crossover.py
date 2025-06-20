from abc import ABC, abstractmethod
from typing import Tuple
import random

import numpy as np

from algorithms.ga.individual import Individual


class CrossoverOperator(ABC):
    @abstractmethod
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        pass


class SinglePointCrossover(CrossoverOperator):
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        length = len(parent1.genes)
        if length <= 1:
            return parent1.copy(), parent2.copy()

        crossover_point = random.randint(1, length - 1)

        offspring1_genes = np.concatenate([
            parent1.genes[:crossover_point],
            parent2.genes[crossover_point:]
        ])
        offspring2_genes = np.concatenate([
            parent2.genes[:crossover_point],
            parent1.genes[crossover_point:]
        ])

        return Individual(offspring1_genes), Individual(offspring2_genes)


class TwoPointCrossover(CrossoverOperator):
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        length = len(parent1.genes)
        if length <= 2:
            return parent1.copy(), parent2.copy()

        point1 = random.randint(1, length - 2)
        point2 = random.randint(point1 + 1, length - 1)

        offspring1_genes = np.concatenate([
            parent1.genes[:point1],
            parent2.genes[point1:point2],
            parent1.genes[point2:]
        ])
        offspring2_genes = np.concatenate([
            parent2.genes[:point1],
            parent1.genes[point1:point2],
            parent2.genes[point2:]
        ])

        return Individual(offspring1_genes), Individual(offspring2_genes)


class UniformCrossover(CrossoverOperator):
    def __init__(self, crossover_probability: float = 0.5):
        self.crossover_probability = crossover_probability

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        length = len(parent1.genes)

        offspring1_genes = np.zeros(length)
        offspring2_genes = np.zeros(length)

        for i in range(length):
            if random.random() < self.crossover_probability:
                offspring1_genes[i] = parent1.genes[i]
                offspring2_genes[i] = parent2.genes[i]
            else:
                offspring1_genes[i] = parent2.genes[i]
                offspring2_genes[i] = parent1.genes[i]

        return Individual(offspring1_genes), Individual(offspring2_genes)
