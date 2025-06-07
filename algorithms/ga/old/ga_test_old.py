import numpy as np

from algorithms.ga.old.ga_old import GeneticAlgorithm


def knapsack_fitness(genes: np.ndarray,
                    values: list,
                    weights: list,
                    max_weight: float) -> float:
    total_value = np.dot(genes, values)
    total_weight = np.dot(genes, weights)
    return total_value if total_weight <= max_weight else 0


# Problem setup
values = [60, 100, 120, 50, 70, 90, 110]
weights = [10, 20, 30, 15, 5, 25, 40]
max_weight = 75

ga = GeneticAlgorithm(
    population_size=100,
    gene_length=len(values),
    fitness_func=lambda genes: knapsack_fitness(genes, values, weights, max_weight),
    problem_type='discrete',
    crossover_method='uniform',
    mutation_rate=0.05,
    elitism=2
)
ga.optimize(100)
ga.plot_progress()
