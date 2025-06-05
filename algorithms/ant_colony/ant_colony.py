import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
# import matplotlib
# matplotlib.use('TkAgg') # remove for jupiter


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
            elite_factor: float = 1.0,
            patience: int = 20,
            convergence_threshold: float = 1e-8,
            verbose=False
    ):
        """
        Parameters:
        - distances: Distance matrix between cities
        - num_ants: Number of ants in the colony, default number of cities
        - num_iterations: Number of optimization iterations
        - alpha: Pheromone importance factor, default = 1.0
        - beta: Heuristic information importance factor, default = 5.0
        - evaporation_rate: Rate at which pheromones evaporate (0-1), best option [0.5, 0.99]
        - Q: Pheromone deposit factor, default = 100
        - elite_factor: Bonus factor for best solution pheromone deposit, default = 1.0
        - patience: Number of iterations without improvement before stopping, default = 20
        - convergence_threshold: Minimum improvement to consider an iteration successful, default = 1e-8
        - verbose: default = False
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
        self.patience = patience
        self.convergence_threshold = convergence_threshold
        self.verbose = verbose

        assert distances.shape[0] == distances.shape[1], "Distance matrix must be square"
        assert self.num_ants > 0, "Number of ants must be positive"
        assert 0 <= evaporation_rate <= 1, "Evaporation rate must be between 0 and 1"

        self.pheromone = np.ones((self.num_cities, self.num_cities)) * 1e-6

        # Heuristic matrix for each value => Lk = 1/distance
        # if distance is zero, set to a small value to avoid division by zero
        self.heuristic = 1.0 / (np.where(distances > 0, distances, 1e-12))
        np.fill_diagonal(self.heuristic, 0)

        self.best_route: Optional[List[int]] = None
        self.best_distance: float = float('inf')
        self.distance_history: List[float] = []
        self.convergence_iteration: Optional[int] = None

    def _calculate_probabilities(self, current_city: int, unvisited: List[int]) -> np.ndarray:
        """
        Probability formula: P_ij = (τ_ij^α * η_ij^β) / Σ(τ_ik^α * η_ik^β)
        where:
        τ_ij: amount if pheromones between i-city and j-city
        η_ij: distance or proximity between i-city and j-city
        k ∈ unvisited cities

        :returns: Probability array over unvisited cities from current city.
        """
        if not unvisited:
            return np.array([])

        unvisited_array = np.array(unvisited)

        # τ_ij^α for unvisited cities
        pheromone_component = self.pheromone[current_city, unvisited_array] ** self.alpha
        # η_ij^β for unvisited cities
        heuristic_component = self.heuristic[current_city, unvisited_array] ** self.beta

        # τ_ij^α * η_ij^β
        attractiveness = pheromone_component * heuristic_component

        # Σ(τ_ik^α * η_ik^β)
        total_attractiveness = np.sum(attractiveness)
        if total_attractiveness == 0:
            # If no pheromone/heuristic info, use uniform distribution
            return np.ones(len(unvisited)) / len(unvisited)

        return attractiveness / total_attractiveness

    def _select_next_city(self, current_city: int, unvisited: List[int]) -> int:
        probabilities = self._calculate_probabilities(current_city, unvisited)

        # roulette wheel
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

        if self.verbose:
            print(f"\nAnt starting at city {start_city}")
            print(f"Initial unvisited cities: {unvisited}")

        while unvisited:
            next_city = self._select_next_city(current_city, unvisited)
            if self.verbose:
                prob = self._calculate_probabilities(current_city, unvisited)
                prob_str = ", ".join([f"{c}: {p * 100:.2f}%" for c, p in zip(unvisited, prob)])
                print(f"  Current city: {current_city} | Unvisited: {unvisited}")
                print(f"  Probabilities: {prob_str}")
                print(f"  Selected next city: {next_city}")
            route.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city

        # return to start city to complete the tour
        route.append(start_city)

        total_distance = self._calculate_route_distance(route)

        if self.verbose:
            print(f"Ant completed route: {route}")
            print(f"Total distance: {total_distance:.2f}\n")

        return route, total_distance

    def _calculate_route_distance(self, route: List[int]) -> float:
        # [1, 3, 0, 2, 4, 1]
        # 1st iteration example, calculate distance from 1 to 3
        # in distance matrix it is distances[1, 3]
        total_distance = 0.0
        for i in range(len(route) - 1):
            total_distance += self.distances[route[i], route[i + 1]]
        return total_distance

    def _update_pheromones(self, all_routes: List[List[int]], all_distances: List[float]):
        """
        Formula: τ_ij = (1 - ρ) * τ_ij + Σ(Δτ_ij)

        Where:
        - τ_ij: pheromone level on edge (i, j)
        - ρ: evaporation rate (0-1)
        - Σ(Δτ_ij): sum of pheromone deposits from all ants
        - Q: pheromone deposit factor

        """
        # Pheromone evaporation - (1 - ρ)
        self.pheromone *= (1.0 - self.evaporation_rate)

        for route, distance in zip(all_routes, all_distances):
            # Amount of pheromone to deposit (inversely proportional to distance)
            deposit_amount = self.Q / distance
            if self.verbose:
                print(f"\nAnt with distance {distance:.2f} deposits {deposit_amount:.4f} pheromone")
            # Deposit pheromone on each edge of the route
            for i in range(len(route) - 1):
                city_a, city_b = route[i], route[i + 1]
                self.pheromone[city_a, city_b] += deposit_amount
                self.pheromone[city_b, city_a] += deposit_amount  # Symmetric TSP

        # Elite ant strategy - extra pheromone for best solution
        if self.best_route is not None:
            elite_deposit = (self.elite_factor * self.Q) / self.best_distance
            if self.verbose:
                print(f"\nElite ant (distance {self.best_distance:.2f}) deposits {elite_deposit:.4f} pheromone")
            for i in range(len(self.best_route) - 1):
                city_a, city_b = self.best_route[i], self.best_route[i + 1]
                self.pheromone[city_a, city_b] += elite_deposit
                self.pheromone[city_b, city_a] += elite_deposit

    def optimize(self) -> Tuple[List[int], float, List[float], int]:
        stagnation_counter = 0
        iterations_completed = 0

        for iteration in range(self.num_iterations):
            iterations_completed = iteration + 1
            iteration_routes = []
            iteration_distances = []

            for ant in range(self.num_ants):
                if self.verbose:
                    print(f"\nAnt {ant + 1}/{self.num_ants}")
                route, distance = self._construct_ant_solution()
                iteration_routes.append(route)
                iteration_distances.append(distance)

            # Update global best solution
            iteration_best_distance = min(iteration_distances)
            prev_best = self.best_distance
            if iteration_best_distance < self.best_distance:
                best_idx = iteration_distances.index(iteration_best_distance)
                self.best_distance = iteration_best_distance
                self.best_route = iteration_routes[best_idx].copy()
                stagnation_counter = 0
                if self.verbose:
                    print(f"\nNew best distance found: {self.best_distance:.2f}")
                    print(f"Best route: {self.best_route}")
            else:
                stagnation_counter += 0

            improvement = abs(prev_best - self.best_distance)
            if improvement < self.convergence_threshold:
                stagnation_counter += 1

            if stagnation_counter >= self.patience:
                self.convergence_iteration = iterations_completed
                print(f"\nEarly stopping at iteration {iteration + 1}")
                print(f"No improvement for {self.patience} consecutive iterations")
                break

            # Update pheromone matrix
            self._update_pheromones(iteration_routes, iteration_distances)

            # Track progress
            self.distance_history.append(self.best_distance)
            self.distance_history = self.distance_history[:iterations_completed]

            if self.verbose and (iteration + 1) % 10 == 0:
                avg_distance = np.mean(iteration_distances)
                print(f"\nIteration summary:")
                print(f"Best distance: {self.best_distance:.2f}")
                print(f"Average distance: {avg_distance:.2f}")
                print(f"Improvement candidates: {len([d for d in iteration_distances if d < self.best_distance * 1.1])}/{self.num_ants}")

        return self.best_route, self.best_distance, self.distance_history, self.convergence_iteration


def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    return np.sqrt(np.sum((point1 - point2) ** 2))


def create_distance_matrix(cities: np.ndarray) -> np.ndarray:
    num_cities = len(cities)
    distances = np.zeros((num_cities, num_cities))

    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                distances[i, j] = euclidean_distance(cities[i], cities[j])

    return distances


def generate_random_cities(num_cities: int, seed: Optional[int] = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)

    cities = np.random.uniform(0, 100, (num_cities, 2))
    return cities


def plot_tsp_solution(
        cities: np.ndarray,
        route: List[int], distance: float,
        distance_history: List[float],
        title: str = "TSP Solution"
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    ax1.scatter(cities[:, 0], cities[:, 1], c='red', s=100, zorder=3)

    for i, (x, y) in enumerate(cities):
        ax1.annotate(str(i), (x, y), xytext=(5, 5),
                     textcoords='offset points', fontsize=12)

    route_coords = cities[route]
    ax1.plot(route_coords[:, 0], route_coords[:, 1], 'b-', linewidth=2, alpha=0.7)

    start_city = route[0]
    ax1.scatter(cities[start_city, 0], cities[start_city, 1],
                c='green', s=200, marker='*', zorder=4)

    ax1.set_title(f"{title}\nTotal Distance: {distance:.2f}")
    ax1.set_xlabel("X Coordinate")
    ax1.set_ylabel("Y Coordinate")
    ax1.grid(True, alpha=0.3)

    ax2.plot(distance_history, 'b-', linewidth=2)
    ax2.set_title("ACO Convergence Curve")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Best Distance Found")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
