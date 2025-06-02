from algorithms.ant_colony.ant_colony import create_distance_matrix, AntColonyOptimizer, \
    plot_tsp_solution, plot_convergence
from algorithms.ant_colony.berlin52 import berlin52_coords

berlin52_distances = create_distance_matrix(berlin52_coords)

aco = AntColonyOptimizer(
    distances=berlin52_distances,
    num_iterations=300,
    alpha=1.0,
    beta=3.0,
    evaporation_rate=0.3,
    Q=100,
    elite_factor=2.0
)

print("Running Ant Colony Optimization...")
best_route, best_distance, history = aco.optimize(verbose=True)

gap = ((best_distance - 7542) / 7542) * 100
print(f"\nTuned ACO Results:")
print(f"Distance: {best_distance:.2f}")
print(f"Gap from optimal: {gap:.2f}%")

# Visualize results
plot_tsp_solution(berlin52_coords, best_route, best_distance)
plot_convergence(history)
