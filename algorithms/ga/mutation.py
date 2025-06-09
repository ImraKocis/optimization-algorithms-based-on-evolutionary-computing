import numpy as np
from abc import ABC, abstractmethod
import random


from algorithms.ga.individual import Individual


class MutationOperator(ABC):
    @abstractmethod
    def mutate(self, individual: Individual) -> Individual:
        pass


class GaussianMutation(MutationOperator):
    def __init__(self, mutation_rate: float = 0.1, sigma: float = 0.1):
        self.mutation_rate = mutation_rate
        self.sigma = sigma

    def mutate(self, individual: Individual) -> Individual:
        mutated_genes = individual.genes.copy()

        for i in range(len(mutated_genes)):
            if random.random() < self.mutation_rate:
                mutated_genes[i] += np.random.normal(0, self.sigma)

        return Individual(mutated_genes)


class AdaptiveGaussianMutation(MutationOperator):
    def __init__(self, initial_mutation_rate=0.1, initial_sigma=0.5,
                 final_sigma=0.01, max_generations=500):
        self.initial_mutation_rate = initial_mutation_rate
        self.initial_sigma = initial_sigma
        self.final_sigma = final_sigma
        self.max_generations = max_generations
        self.current_generation = 0

    def update_generation(self, generation):
        self.current_generation = generation

    def mutate(self, individual: Individual) -> Individual:
        # starts high, decreases over time
        progress = self.current_generation / self.max_generations
        current_sigma = self.initial_sigma * (1 - progress) + self.final_sigma * progress

        mutated_genes = individual.genes.copy()

        for i in range(len(mutated_genes)):
            if random.random() < self.initial_mutation_rate:
                mutated_genes[i] += np.random.normal(0, current_sigma)

        return Individual(mutated_genes)


class FlipBitMutation(MutationOperator):
    def __init__(self, mutation_rate: float = 0.01):
        self.mutation_rate = mutation_rate

    def mutate(self, individual: Individual) -> Individual:
        mutated_genes = individual.genes.copy()

        for i in range(len(mutated_genes)):
            if random.random() < self.mutation_rate:
                mutated_genes[i] = 1.0 - mutated_genes[i]

        return Individual(mutated_genes)
