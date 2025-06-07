from algorithms.ga.ga import GeneticAlgorithm
from algorithms.ga.knapsack_problem import KnapsackProblem
from algorithms.ga.population import RandomContinuousInitializer, RandomBinaryInitializer
from algorithms.ga.utils import SelectionMethod, CrossoverMethod, MutationMethod
from utils.objective_functions import rastrigin_objective_function


def solve_rastrigin():
    print("=== Solving Rastrigin Function ===")

    # Problem setup
    dimensions = 2
    bounds = (-5.12, 5.12)

    # Initialize GA
    ga = GeneticAlgorithm(
        population_size=100,
        crossover_rate=0.8,
        mutation_rate=0.1,
        elitism_count=5,
        max_generations=500,
        stagnation_limit=50,
        population_initializer=RandomContinuousInitializer(dimensions, bounds),
        selection_method=SelectionMethod.TOURNAMENT,
        crossover_method=CrossoverMethod.UNIFORM,
        mutation_method=MutationMethod.GAUSSIAN,
        tournament_size=3,
        mutation_sigma=0.1
    )

    # Evolve solution
    best_solution = ga.evolve(rastrigin_objective_function)

    print(f"Best solution found: {best_solution.genes}")
    print(f"Best fitness: {best_solution.fitness}")

    # Plot evolution
    ga.plot_evolution()


# Example 2: Solving Knapsack Problem (Discrete Optimization)
def solve_knapsack():
    print("\n=== Solving Knapsack Problem ===")

    # Problem setup - classic example
    weights = [7, 2, 1, 9, 3]
    values = [5, 4, 7, 2, 6]
    capacity = 15

    knapsack = KnapsackProblem(weights, values, capacity)

    # Initialize GA
    ga = GeneticAlgorithm(
        population_size=50,
        crossover_rate=0.85,
        mutation_rate=0.02,
        elitism_count=3,
        max_generations=200,
        stagnation_limit=100,
        population_initializer=RandomBinaryInitializer(len(weights)),
        selection_method=SelectionMethod.ROULETTE_WHEEL,
        crossover_method=CrossoverMethod.TWO_POINT,
        mutation_method=MutationMethod.FLIP_BIT,
        tournament_size=3
    )

    # Evolve solution
    best_solution = ga.evolve(knapsack.fitness_function)

    # Decode and display solution
    selected_items, total_weight, total_value = knapsack.decode_solution(best_solution.genes)

    print(f"Best solution encoding: {best_solution.genes}")
    print(f"Selected items (0-indexed): {selected_items}")
    print(f"Total weight: {total_weight}/{capacity}")
    print(f"Total value: {total_value}")
    print(f"Fitness: {best_solution.fitness}")

    # Show item details
    print("\nItem details:")
    for i, (w, v) in enumerate(zip(weights, values)):
        status = "SELECTED" if i in selected_items else "not selected"
        print(f"Item {i}: weight={w}, value={v} - {status}")

    # Plot evolution
    ga.plot_evolution()
