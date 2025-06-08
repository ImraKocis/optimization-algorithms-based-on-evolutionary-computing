from algorithms.ga.fitness_transformer import FitnessTransformer
from algorithms.ga.ga import GeneticAlgorithm
from algorithms.ga.knapsack_problem import KnapsackProblem, create_random_knapsack_problem
from algorithms.ga.population import RandomContinuousInitializer, RandomBinaryInitializer
from algorithms.ga.utils import SelectionMethod, CrossoverMethod, MutationMethod
from utils.objective_functions import rastrigin_objective_function


def solve_rastrigin():
    print("=== Solving Rastrigin Function ===")

    dimensions = 10
    bounds = (-5.12, 5.12)

    fitness_transformer = FitnessTransformer(rastrigin_objective_function, method="negative")

    # Initialize GA
    ga = GeneticAlgorithm(
        population_size=200,
        crossover_rate=0.8,
        mutation_rate=0.1,
        elitism_count=2,
        max_generations=500,
        stagnation_limit=100,
        population_initializer=RandomContinuousInitializer(dimensions, bounds),
        selection_method=SelectionMethod.TOURNAMENT,
        crossover_method=CrossoverMethod.TWO_POINT,
        mutation_method=MutationMethod.ADAPTIVE_GAUSSIAN,
        tournament_size=3,
        mutation_sigma=0.5
    )

    # Evolve solution
    best_solution = ga.evolve(fitness_transformer)

    minimized_value = rastrigin_objective_function(best_solution.genes)

    print(f"Best solution found: {best_solution.genes}")
    print(f"Minimized Rastrigin value: {minimized_value:.6f}")
    print(f"Best fitness: {best_solution.fitness}")

    # Plot evolution
    ga.plot_evolution()


# Example 2: Solving Knapsack Problem (Discrete Optimization)
def solve_knapsack():
    print("\n=== Solving Knapsack Problem ===")

    # Create a knapsack problem
    knapsack = create_random_knapsack_problem(
        num_items=20,
        min_value=10.0,
        max_value=100.0,
        min_weight=5.0,
        max_weight=30.0,
        capacity_factor=0.65,
        seed=42  # For reproducible results
    )

    knapsack.print_problem_info()

    ga = GeneticAlgorithm(
        population_size=50,
        crossover_rate=0.85,
        mutation_rate=0.02,
        elitism_count=3,
        max_generations=200,
        stagnation_limit=30,
        population_initializer=RandomBinaryInitializer(knapsack.num_items),
        selection_method=SelectionMethod.ROULETTE_WHEEL,
        crossover_method=CrossoverMethod.TWO_POINT,
        mutation_method=MutationMethod.FLIP_BIT,
        tournament_size=3
    )

    print("\n=== Running Genetic Algorithm ===")
    best_solution = ga.evolve(knapsack.fitness_function)

    print(f"\n=== Final Results ===")
    print(f"Best fitness: {best_solution.fitness:.2f}")

    # Print detailed solution
    knapsack.print_solution(best_solution.genes)

    # Manual verification
    selected_items, total_weight, total_value = knapsack.decode_solution(best_solution.genes)
    print(f"\nManual verification:")
    print(f"Calculated total value: {total_value:.2f}")
    print(f"GA fitness value: {best_solution.fitness:.2f}")
    print(f"Values match: {'Yes' if abs(total_value - best_solution.fitness) < 0.01 else 'No'}")

    # Plot evolution
    ga.plot_evolution()
