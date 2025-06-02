from algorithms.ant_colony.ant_colony import generate_random_cities, create_distance_matrix, AntColonyOptimizer, \
    plot_tsp_solution, plot_convergence

num_cities = 20
cities = generate_random_cities(num_cities, seed=42)
distances = create_distance_matrix(cities)

# Create and run ACO
aco = AntColonyOptimizer(
    distances=distances,
    num_iterations=100,
    alpha=1.0,  # Pheromone importance
    beta=2.0,  # Heuristic importance
    evaporation_rate=0.5,
    Q=100,
    elite_factor=1.0
)

print("Running Ant Colony Optimization...")
best_route, best_distance, history = aco.optimize(verbose=True)

print(f"\nOptimization completed!")
print(f"Best route: {best_route}")
print(f"Best distance: {best_distance:.2f}")

# Visualize results
plot_tsp_solution(cities, best_route, best_distance)
plot_convergence(history)
