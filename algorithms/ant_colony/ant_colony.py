import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import matplotlib
matplotlib.use('TkAgg') # remove for jupiter


class AntColonyOptimizer:
    def __init__(
            self,
            distances: np.ndarray,
            num_ants: Optional[int] = None,
            num_iterations: int = 100,
            alpha: float = 1.0,  # usually 1
            beta: float = 5.0,  # usually 5
            evaporation_rate: float = 0.5,  # usually [0.5, 0.99]
            Q: float = 100,  # usually 100
            elite_factor: float = 1.0
    ):
        """
        Initialize Ant Colony Optimizer for TSP.

        Parameters:
        - distances: Distance matrix between cities
        - num_ants: Number of ants in the colony, default number of cities
        - num_iterations: Number of optimization iterations
        - alpha: Pheromone importance factor, default = 1.0
        - beta: Heuristic information importance factor, default = 5.0
        - evaporation_rate: Rate at which pheromones evaporate (0-1), best option [0.5, 0.99]
        - Q: Pheromone deposit factor, default = 100
        - elite_factor: Bonus factor for best solution pheromone deposit, default = 1.0
        """
        self.distances = distances
        self.num_cities = distances.shape[0]
        if num_ants is None:
            self.num_ants = self.num_cities
        else:
            self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.Q = Q
        self.elite_factor = elite_factor

        # Validation
        assert distances.shape[0] == distances.shape[1], "Distance matrix must be square"
        assert self.num_ants > 0, "Number of ants must be positive"
        assert 0 <= evaporation_rate <= 1, "Evaporation rate must be between 0 and 1"

        # Initialize pheromone matrix with small positive values
        self.pheromone = np.ones((self.num_cities, self.num_cities)) * 1e-6

        # Heuristic information: inverse of distance (visibility)
        # Add small epsilon to avoid division by zero
        self.heuristic = 1.0 / (distances + 1e-10)
        # Set diagonal to zero (no self-loops)
        np.fill_diagonal(self.heuristic, 0)

        # Best solution tracking
        self.best_route: Optional[List[int]] = None
        self.best_distance: float = float('inf')
        self.distance_history: List[float] = []

    def _calculate_probabilities(self, current_city: int, unvisited: List[int]) -> np.ndarray:
        """
        Calculate probability of moving to each unvisited city.

        Probability formula: P_ij = (τ_ij^α * η_ij^β) / Σ(τ_ik^α * η_ik^β)
        where τ is pheromone, η is heuristic info, k ∈ unvisited cities
        """
        if not unvisited:
            return np.array([])

        unvisited_array = np.array(unvisited)

        # Calculate pheromone and heuristic components
        pheromone_component = self.pheromone[current_city, unvisited_array] ** self.alpha
        heuristic_component = self.heuristic[current_city, unvisited_array] ** self.beta

        # Combined attractiveness
        attractiveness = pheromone_component * heuristic_component

        # Normalize to get probabilities
        total_attractiveness = np.sum(attractiveness)
        if total_attractiveness == 0:
            # If no pheromone/heuristic info, use uniform distribution
            return np.ones(len(unvisited)) / len(unvisited)

        return attractiveness / total_attractiveness

    def _select_next_city(self, current_city: int, unvisited: List[int]) -> int:
        """Select next city based on probability distribution."""
        probabilities = self._calculate_probabilities(current_city, unvisited)

        # Roulette wheel selection
        selected_idx = np.random.choice(len(unvisited), p=probabilities)
        return unvisited[selected_idx]

    def _construct_ant_solution(self) -> Tuple[List[int], float]:
        """
        Construct a complete tour for a single ant.

        Returns:
        - route: List of cities in order visited
        - total_distance: Total distance of the route
        """
        # Start from random city
        start_city = np.random.randint(0, self.num_cities)
        route = [start_city]
        unvisited = list(range(self.num_cities))
        unvisited.remove(start_city)

        current_city = start_city

        # Visit all remaining cities
        while unvisited:
            next_city = self._select_next_city(current_city, unvisited)
            route.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city

        # Return to start city to complete the tour
        route.append(start_city)

        # Calculate total distance
        total_distance = self._calculate_route_distance(route)

        return route, total_distance

    def _calculate_route_distance(self, route: List[int]) -> float:
        """Calculate total distance for a given route."""
        total_distance = 0.0
        for i in range(len(route) - 1):
            total_distance += self.distances[route[i], route[i + 1]]
        return total_distance

    def _update_pheromones(self, all_routes: List[List[int]], all_distances: List[float]):
        """
        Update pheromone matrix based on ant solutions.

        Steps:
        1. Evaporate existing pheromones
        2. Deposit new pheromones based on solution quality
        3. Apply elite ant strategy (bonus for best solution)
        """
        # Step 1: Pheromone evaporation
        self.pheromone *= (1.0 - self.evaporation_rate)

        # Step 2: Pheromone deposition by all ants
        for route, distance in zip(all_routes, all_distances):
            # Amount of pheromone to deposit (inversely proportional to distance)
            deposit_amount = self.Q / distance

            # Deposit pheromone on each edge of the route
            for i in range(len(route) - 1):
                city_a, city_b = route[i], route[i + 1]
                self.pheromone[city_a, city_b] += deposit_amount
                self.pheromone[city_b, city_a] += deposit_amount  # Symmetric TSP

        # Step 3: Elite ant strategy - extra pheromone for best solution
        if self.best_route is not None:
            elite_deposit = (self.elite_factor * self.Q) / self.best_distance
            for i in range(len(self.best_route) - 1):
                city_a, city_b = self.best_route[i], self.best_route[i + 1]
                self.pheromone[city_a, city_b] += elite_deposit
                self.pheromone[city_b, city_a] += elite_deposit

    def optimize(self, verbose: bool = False) -> Tuple[List[int], float, List[float]]:
        """
        Run the complete ACO optimization process.

        Returns:
        - best_route: Best route found
        - best_distance: Distance of best route
        - distance_history: History of best distances per iteration
        """
        for iteration in range(self.num_iterations):
            # Generate solutions for all ants
            iteration_routes = []
            iteration_distances = []

            for ant in range(self.num_ants):
                route, distance = self._construct_ant_solution()
                iteration_routes.append(route)
                iteration_distances.append(distance)

            # Update global best solution
            iteration_best_distance = min(iteration_distances)
            if iteration_best_distance < self.best_distance:
                best_idx = iteration_distances.index(iteration_best_distance)
                self.best_distance = iteration_best_distance
                self.best_route = iteration_routes[best_idx].copy()

            # Update pheromone matrix
            self._update_pheromones(iteration_routes, iteration_distances)

            # Track progress
            self.distance_history.append(self.best_distance)

            if verbose and (iteration + 1) % 10 == 0:
                avg_distance = np.mean(iteration_distances)
                print(f"Iteration {iteration + 1:3d}: "
                      f"Best = {self.best_distance:.2f}, "
                      f"Avg = {avg_distance:.2f}, "
                      f"Improvement = {len([d for d in iteration_distances if d < self.best_distance * 1.1])}/{self.num_ants}")

        return self.best_route, self.best_distance, self.distance_history


def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """Calculate Euclidean distance between two points."""
    return np.sqrt(np.sum((point1 - point2) ** 2))


def create_distance_matrix(cities: np.ndarray) -> np.ndarray:
    """Create distance matrix from city coordinates."""
    num_cities = len(cities)
    distances = np.zeros((num_cities, num_cities))

    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                distances[i, j] = euclidean_distance(cities[i], cities[j])

    return distances


def generate_random_cities(num_cities: int, seed: Optional[int] = None) -> np.ndarray:
    """Generate random city coordinates for testing."""
    if seed is not None:
        np.random.seed(seed)

    cities = np.random.uniform(0, 100, (num_cities, 2))
    return cities


# Visualization function
def plot_tsp_solution(cities: np.ndarray, route: List[int], distance: float, title: str = "TSP Solution"):
    """Plot the TSP solution."""
    plt.figure(figsize=(10, 8))

    # Plot cities
    plt.scatter(cities[:, 0], cities[:, 1], c='red', s=100, zorder=3)

    # Label cities
    for i, (x, y) in enumerate(cities):
        plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', fontsize=12)

    # Plot route
    route_coords = cities[route]
    plt.plot(route_coords[:, 0], route_coords[:, 1], 'b-', linewidth=2, alpha=0.7)

    # Highlight start city
    start_city = route[0]
    plt.scatter(cities[start_city, 0], cities[start_city, 1], c='green', s=200, marker='*', zorder=4)

    plt.title(f"{title}\nTotal Distance: {distance:.2f}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_convergence(distance_history: List[float]):
    """Plot convergence curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(distance_history, 'b-', linewidth=2)
    plt.title("ACO Convergence Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Best Distance Found")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
