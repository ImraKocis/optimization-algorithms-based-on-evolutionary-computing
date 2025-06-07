from abc import ABC, abstractmethod
import random

import numpy as np
from typing import Callable, Tuple, List

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg') # remove for jupiter


class Individual(ABC):
    """Abstract base class for all individuals"""

    def __init__(self, genes: np.ndarray):
        self.genes = genes
        self.fitness = float('-inf')

    @abstractmethod
    def mutate(self, mutation_rate: float):
        pass

    @abstractmethod
    def create_child(self, genes: np.ndarray) -> 'Individual':
        pass


class ContinuousIndividual(Individual):
    def __init__(self,
                 genes: np.ndarray,
                 mutation_strength: float = 0.1,
                 gene_bounds: Tuple[float, float] = (-5.12, 5.12)):
        super().__init__(genes)
        self.mutation_strength = mutation_strength
        self.gene_bounds = gene_bounds  # Add bounds parameter

    def mutate(self, mutation_rate: float):
        mask = np.random.random(size=len(self.genes)) < mutation_rate
        self.genes[mask] += np.random.normal(0, self.mutation_strength, size=np.sum(mask))
        self.genes = np.clip(self.genes, *self.gene_bounds)  # Clip to bounds

    def create_child(self, genes: np.ndarray) -> 'ContinuousIndividual':
        return ContinuousIndividual(genes, self.mutation_strength)


class DiscreteIndividual(Individual):
    def mutate(self, mutation_rate: float):
        mask = np.random.random(size=len(self.genes)) < mutation_rate
        self.genes[mask] = 1 - self.genes[mask]

    def create_child(self, genes: np.ndarray) -> 'DiscreteIndividual':
        return DiscreteIndividual(genes)


class GeneticAlgorithm:
    def __init__(self,
                 population_size: int,
                 gene_length: int,
                 fitness_func: Callable,
                 problem_type: str,
                 mutation_rate: float = 0.01,
                 elitism: int = 1,
                 selection_method: str = 'tournament',
                 crossover_method: str = 'single_point'):  # New config parameter

        if not isinstance(population_size, int) or population_size <= 0:
            raise ValueError("population_size must be positive integer")
        if not isinstance(elitism, int) or elitism < 0 or elitism >= population_size:
            raise ValueError("elitism must be 0 <= elitism < population_size")
        if problem_type not in ['continuous', 'discrete']:
            raise ValueError(f"Invalid problem_type: {problem_type}. Must be 'continuous' or 'discrete'.")
        if selection_method not in ['tournament', 'roulette']:
            raise ValueError(f"Invalid selection_method: {selection_method}. Must be 'tournament' or 'roulette'.")

        self.population_size = population_size
        self.gene_length = gene_length
        self.fitness_func = fitness_func
        self.problem_type = problem_type
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.population: List[Individual] = []
        self.best_fitness_history = []
        self._crossover_strategies = {
            'single_point': self._single_point_crossover,
            'two_point': self._two_point_crossover,
            'uniform': self._uniform_crossover
        }

    def select_parents(self, tournament_size: int = 3) -> List[Individual]:
        if self.selection_method == 'tournament':
            return self._tournament_selection(tournament_size)
        elif self.selection_method == 'roulette':
            return self._roulette_wheel_selection()
        else:
            raise ValueError("Invalid selection method")

    def _tournament_selection(self, tournament_size: int) -> List[Individual]:
        selected = []
        for _ in range(self.population_size - self.elitism):
            contestants = random.choices(self.population, k=tournament_size)
            winner = max(contestants, key=lambda x: x.fitness)
            selected.append(winner)
        return selected

    def _roulette_wheel_selection(self) -> List[Individual]:
        fitnesses = [ind.fitness for ind in self.population]

        # Handle negative fitness values
        min_fitness = min(fitnesses)
        if min_fitness < 0:
            fitnesses = [f - min_fitness + 1e-8 for f in fitnesses]

        # Handle all-zero case
        if sum(fitnesses) == 0:
            return random.choices(self.population, k=self.population_size - self.elitism)

        return random.choices(
            population=self.population,
            weights=fitnesses,
            k=self.population_size - self.elitism
        )

    def evaluate_fitness(self):
        for individual in self.population:
            individual.fitness = self.fitness_func(individual.genes)

    def run_generation(self):
        self.evaluate_fitness()

        # Handle empty population case
        if not self.population:
            raise RuntimeError("Population went extinct")

        current_best = max(self.population, key=lambda x: x.fitness)
        self.best_fitness_history.append(current_best.fitness)

        new_population = []

        # Elitism
        elites = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:self.elitism]
        new_population.extend(elites)

        # Calculate required parents
        needed_children = self.population_size - len(elites)
        needed_parents = needed_children * 2  # Each parent pair produces 2 children

        # Selection
        parents = self.select_parents()

        # Ensure even number of parents for crossover
        if len(parents) % 2 != 0:
            parents = parents[:-1]

        # Limit parents to what's needed
        parents = parents[:needed_parents]

        # Crossover and mutation
        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[i + 1]
            child1, child2 = self._crossover(parent1, parent2)
            child1.mutate(self.mutation_rate)
            child2.mutate(self.mutation_rate)
            new_population.extend([child1, child2])

        # Maintain population size with new random individuals if needed
        remaining = self.population_size - len(new_population)
        if remaining > 0:
            if self.problem_type == 'continuous':
                new_individuals = [ContinuousIndividual(np.random.uniform(-5.12, 5.12, self.gene_length))
                                   for _ in range(remaining)]
            else:
                new_individuals = [DiscreteIndividual(np.random.randint(0, 2, self.gene_length))
                                   for _ in range(remaining)]
            new_population.extend(new_individuals)

        self.population = new_population[:self.population_size]

    def optimize(self, generations: int):
        for _ in range(generations):
            self.run_generation()
        return max(self.population, key=lambda x: x.fitness)

    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Select crossover strategy based on configuration"""
        return self._crossover_strategies[self.crossover_method](parent1, parent2)

    @staticmethod
    def _single_point_crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """One-point crossover implementation"""
        crossover_point = np.random.randint(1, len(parent1.genes))
        child1_genes = np.concatenate([parent1.genes[:crossover_point], parent2.genes[crossover_point:]])
        child2_genes = np.concatenate([parent2.genes[:crossover_point], parent1.genes[crossover_point:]])
        return parent1.create_child(child1_genes), parent2.create_child(child2_genes)

    @staticmethod
    def _two_point_crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        points = sorted(np.random.choice(len(parent1.genes), 2, replace=False))
        child1_genes = np.concatenate([
            parent1.genes[:points[0]],
            parent2.genes[points[0]:points[1]],
            parent1.genes[points[1]:]
        ])
        child2_genes = np.concatenate([
            parent2.genes[:points[0]],
            parent1.genes[points[0]:points[1]],
            parent2.genes[points[1]:]
        ])
        return parent1.create_child(child1_genes), parent2.create_child(child2_genes)

    @staticmethod
    def _uniform_crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Uniform crossover implementation"""
        mask = np.random.randint(0, 2, size=len(parent1.genes)).astype(bool)
        child1_genes = np.where(mask, parent1.genes, parent2.genes)
        child2_genes = np.where(mask, parent2.genes, parent1.genes)
        return parent1.create_child(child1_genes), parent2.create_child(child2_genes)

    def plot_progress(self):
        plt.plot(self.best_fitness_history)
        plt.title('Best Fitness Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.show()
