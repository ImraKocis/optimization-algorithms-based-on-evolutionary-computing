from pygad import pygad
import numpy as np

from utils.objective_functions import rastrigin_objective_function


class RastriginPyGAD:
    def __init__(self, num_generations=200, sol_per_pop=100, dimensions=10):
        self.num_generations = num_generations
        self.sol_per_pop = sol_per_pop
        self.dimensions = dimensions

        self.ga = pygad.GA(
            num_generations=self.num_generations,
            num_parents_mating=10,
            sol_per_pop=self.sol_per_pop,
            num_genes=self.dimensions,
            init_range_low=-5.12,
            init_range_high=5.12,
            gene_type=float,
            fitness_func=self.fitness_function,
            parent_selection_type="tournament",
            crossover_type="uniform",
            mutation_type="adaptive",
            mutation_percent_genes=[20, 5],  # must be tuple id adaptive
            keep_parents=4,
            K_tournament=7,
            random_mutation_min_val=-1.0,
            random_mutation_max_val=1.0,
            random_seed=42,  # for reproducibility
            suppress_warnings=True,
        )

    def fitness_function(slef, gad, solution, solution_idx):
        value = rastrigin_objective_function(solution)
        return -value

    def run(self):
        self.ga.run()

        solution, solution_fitness, _ = self.ga.best_solution()

        print(f"Number of generations: {self.num_generations}")
        print(f"Population size: {self.sol_per_pop}")
        print(f"Number of genes or dimensions: {self.dimensions}")
        print(f"Best solution: {solution}")
        print(f"Best solution fitness: {-solution_fitness}")
        print(f"Distance from global optimum: {np.sqrt(np.sum(solution**2))}")

        self.ga.plot_fitness()

        return solution, -solution_fitness
