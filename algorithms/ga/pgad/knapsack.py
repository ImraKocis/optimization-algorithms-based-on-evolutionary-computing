import numpy as np
import pygad


class KnapsackPyGAD:
    def __init__(self, values, weights, capacity, sol_per_pop=50, num_generations=100):
        self.values = values
        self.weights = weights
        self.capacity = capacity

        self.num_generations = num_generations
        self.num_parents_mating = 4
        self.init_range_low = 0
        self.init_range_high = 1
        self.sol_per_pop = sol_per_pop
        self.num_genes = len(values)

        self.ga = pygad.GA(
            num_generations=self.num_generations,
            num_parents_mating=self.num_parents_mating,
            sol_per_pop=self.sol_per_pop,  # sol_per_pop is num of solutions (e.e. chromosomes) in the population
            num_genes=self.num_genes,
            init_range_low=self.init_range_low,  # lower bound for gene initialization
            init_range_high=self.init_range_high,  # upper bound for gene initialization
            gene_type=int,
            gene_space=[0, 1],  # binary
            fitness_func=self.fitness_func,
            crossover_type="uniform",
            mutation_type="random",
            mutation_percent_genes=5,
            keep_parents=0,  # no parents are kept in population
            random_seed=42,  # for reproducibility
            suppress_warnings=True
        )

    def fitness_func(self, gad, solution, solution_idx):
        solution = np.array(solution)

        total_weight = np.sum(self.weights * solution)
        total_value = np.sum(self.values * solution)

        if total_weight > self.capacity:
            return 0

        return total_value

    def run(self):
        self.ga.run()

        solution, solution_fitness, _ = self.ga.best_solution()

        print(f"Number of generations: {self.num_generations}")
        print(f"Population size: {self.sol_per_pop}")
        print(f"Number of genes: {self.num_genes}")
        print(f"Best solution: {solution}")
        print(f"Best solution fitness: {solution_fitness}")
        print(f"Selected items: {np.sum(solution)}/{len(solution)}")
        print(f"Weight constraint check: {np.sum(self.weights * solution)}/{self.capacity}")

        self.ga.plot_fitness()
        return solution, solution_fitness
