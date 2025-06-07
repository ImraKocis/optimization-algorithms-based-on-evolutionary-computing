import numpy as np
from abc import ABC, abstractmethod
import random


from algorithms.ga.individual import Individual


class MutationOperator(ABC):
    """Abstract base class for mutation operators"""

    @abstractmethod
    def mutate(self, individual: Individual) -> Individual:
        pass


class GaussianMutation(MutationOperator):
    """Gaussian mutation for continuous optimization"""

    def __init__(self, mutation_rate: float = 0.1, sigma: float = 0.1):
        self.mutation_rate = mutation_rate
        self.sigma = sigma

    def mutate(self, individual: Individual) -> Individual:
        mutated_genes = individual.genes.copy()

        for i in range(len(mutated_genes)):
            if random.random() < self.mutation_rate:
                # Add Gaussian noise to the gene
                mutated_genes[i] += np.random.normal(0, self.sigma)

        return Individual(mutated_genes)


class FlipBitMutation(MutationOperator):
    """Flip-bit mutation for binary optimization"""

    def __init__(self, mutation_rate: float = 0.01):
        self.mutation_rate = mutation_rate

    def mutate(self, individual: Individual) -> Individual:
        mutated_genes = individual.genes.copy()

        for i in range(len(mutated_genes)):
            if random.random() < self.mutation_rate:
                # Flip the bit (0 becomes 1, 1 becomes 0)
                mutated_genes[i] = 1.0 - mutated_genes[i]

        return Individual(mutated_genes)
