from typing import Tuple

import numpy as np


class Firefly:
    def __init__(self, dimensions: int, bounds: Tuple[float, float]):
        """
        Args:
            dimensions: num of dimensions in the search space
            bounds: (min, max)
        """
        self.dimensions = dimensions
        self.bounds = bounds

        self.position = np.random.uniform(
            bounds[0], bounds[1], size=dimensions
        )

        self.fitness = float('inf')
        self.intensity = 0.0

    def calculate_intensity(self, fitness_value: float, is_minimization: bool = True):
        """
        Calculate light intensity based on fitness value.

        For minimization: better fitness (lower) = higher intensity
        For maximization: better fitness (higher) = higher intensity
        """
        self.fitness = fitness_value

        if is_minimization:
            # For minimization problems, lower fitness means higher intensity
            self.intensity = 1.0 / (1.0 + fitness_value)
        else:
            # For maximization problems, higher fitness means higher intensity
            self.intensity = fitness_value

    def distance_to(self, other_firefly: 'Firefly') -> float:
        """Calculate Euclidean distance to another firefly."""
        return float(np.linalg.norm(self.position - other_firefly.position))

    def move_towards(self, brighter_firefly: 'Firefly', beta0: float,
                     gamma: float, alpha: float, random_factor: np.ndarray):
        """
        Move this firefly towards a brighter firefly using the FA update equation.

        The movement equation is:
        x_i = x_i + β₀ * exp(-γ * r²) * (x_j - x_i) + α * ε

        Where:
        - β₀: base attractiveness
        - γ: absorption coefficient
        - r: distance between fireflies
        - α: randomization parameter
        - ε: random vector
        """
        # Calculate distance and attractiveness
        distance = self.distance_to(brighter_firefly)
        attractiveness = beta0 * np.exp(-gamma * distance ** 2)

        # Update position using the firefly algorithm formula
        self.position = (
                self.position +
                attractiveness * (brighter_firefly.position - self.position) +
                alpha * random_factor
        )

    def random_walk(self, alpha: float, random_factor: np.ndarray):
        """Perform random walk when no brighter firefly is found."""
        self.position += alpha * random_factor

    def clip_to_bounds(self, bounds: Tuple[float, float]):
        """Ensure firefly position stays within problem bounds."""
        self.position = np.clip(self.position, bounds[0], bounds[1])

    def __str__(self):
        return f"Firefly(pos={self.position}, fitness={self.fitness:.6f}, intensity={self.intensity:.6f})"
