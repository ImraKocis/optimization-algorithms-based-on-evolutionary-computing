from abc import ABC, abstractmethod
import random
from typing import List

from algorithms.ga.individual import Individual


class SelectionOperator(ABC):
    @abstractmethod
    def select(self, population: List[Individual], num_parents: int) -> List[Individual]:
        pass


class TournamentSelection(SelectionOperator):
    def __init__(self, tournament_size: int = 3, minimize: bool = False):
        self.tournament_size = tournament_size
        self.minimize = minimize
        self.comparison_func = min if minimize else max

    def select(self, population: List[Individual], num_parents: int) -> List[Individual]:
        parents = []
        for _ in range(num_parents):
            tournament = random.sample(population,
                                       min(self.tournament_size, len(population)))
            winner = self.comparison_func(tournament, key=lambda ind: ind.fitness)
            parents.append(winner.copy())
        return parents


class RouletteWheelSelection(SelectionOperator):
    def __init__(self, minimize: bool = False):
        self.minimize = minimize

    def select(self, population: List[Individual], num_parents: int) -> List[Individual]:
        if self.minimize:
            return self._select_for_minimization(population, num_parents)
        else:
            return self._select_for_maximization(population, num_parents)

    def _select_for_maximization(self, population: List[Individual], num_parents: int) -> List[Individual]:
        min_fitness = min(ind.fitness for ind in population)
        if min_fitness < 0:
            adjusted_fitnesses = [ind.fitness - min_fitness + 1e-10 for ind in population]
        else:
            adjusted_fitnesses = [ind.fitness + 1e-10 for ind in population]

        total_fitness = sum(adjusted_fitnesses)
        probabilities = [f / total_fitness for f in adjusted_fitnesses]

        return self._spin_wheel(population, probabilities, num_parents)

    def _select_for_minimization(self, population: List[Individual], num_parents: int) -> List[Individual]:
        fitness_values = [ind.fitness for ind in population]
        max_fitness = max(fitness_values)
        min_fitness = min(fitness_values)

        if min_fitness >= 0:
            inverted_fitnesses = [(max_fitness + 1e-10) - f for f in fitness_values]
        else:
            shifted_max = max_fitness - min_fitness + 1e-10
            inverted_fitnesses = [shifted_max - (f - min_fitness) for f in fitness_values]

        # Ensure all inverted fitnesses are positive
        inverted_fitnesses = [max(f, 1e-10) for f in inverted_fitnesses]

        total_fitness = sum(inverted_fitnesses)
        probabilities = [f / total_fitness for f in inverted_fitnesses]

        return self._spin_wheel(population, probabilities, num_parents)

    @staticmethod
    def _spin_wheel(population: List[Individual], probabilities: List[float], num_parents: int) -> List[Individual]:
        parents = []
        for _ in range(num_parents):
            r = random.random()
            cumulative_prob = 0
            for i, prob in enumerate(probabilities):
                cumulative_prob += prob
                if r <= cumulative_prob:
                    parents.append(population[i].copy())
                    break
        return parents
