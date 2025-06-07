from abc import ABC, abstractmethod
import random
from typing import List

from algorithms.ga.individual import Individual


class SelectionOperator(ABC):
    """Abstract base class for selection operators"""

    @abstractmethod
    def select(self, population: List[Individual], num_parents: int) -> List[Individual]:
        pass


class TournamentSelection(SelectionOperator):
    """Tournament selection implementation"""

    def __init__(self, tournament_size: int = 3):
        self.tournament_size = tournament_size

    def select(self, population: List[Individual], num_parents: int) -> List[Individual]:
        parents = []
        for _ in range(num_parents):
            # Select random individuals for tournament
            tournament = random.sample(population,
                                       min(self.tournament_size, len(population)))
            # Choose the best individual from tournament
            winner = max(tournament, key=lambda ind: ind.fitness)
            parents.append(winner.copy())
        return parents


class RouletteWheelSelection(SelectionOperator):
    """Roulette wheel selection implementation"""

    def select(self, population: List[Individual], num_parents: int) -> List[Individual]:
        # Handle negative fitness values by shifting
        min_fitness = min(ind.fitness for ind in population)
        if min_fitness < 0:
            adjusted_fitnesses = [ind.fitness - min_fitness + 1e-10 for ind in population]
        else:
            adjusted_fitnesses = [ind.fitness + 1e-10 for ind in population]

        total_fitness = sum(adjusted_fitnesses)
        probabilities = [f / total_fitness for f in adjusted_fitnesses]

        parents = []
        for _ in range(num_parents):
            # Spin the roulette wheel
            r = random.random()
            cumulative_prob = 0
            for i, prob in enumerate(probabilities):
                cumulative_prob += prob
                if r <= cumulative_prob:
                    parents.append(population[i].copy())
                    break
        return parents
