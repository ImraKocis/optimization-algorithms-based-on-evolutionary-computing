import numpy as np


class Particle:
    def __init__(self, bounds, velocity_factor=0.1, vmax_factor=0.2):
        self.bounds = bounds
        self.dimensions = bounds.shape[0]
        self.position = np.random.uniform(bounds[:, 0], bounds[:, 1], self.dimensions)

        # Velocity initialized as fraction of search range
        search_range = bounds[:, 1] - bounds[:, 0]
        self.max_velocity = vmax_factor * search_range
        self.velocity = np.random.uniform(
            -velocity_factor * search_range,
            velocity_factor * search_range,
            self.dimensions
        )

        self.personal_best_position = self.position.copy()
        self.personal_best_value = None  # Will be set on first evaluation

    def update_personal_best(self, value, is_better_func):
        if self.personal_best_value is None or is_better_func(value, self.personal_best_value):
            self.personal_best_value = value
            self.personal_best_position = self.position.copy()

    def update_velocity(self, w, c1, c2, global_best_position):
        r1 = np.random.rand(self.dimensions)
        r2 = np.random.rand(self.dimensions)
        # c1 * r1 * (pbest - x(t))
        cognitive = c1 * r1 * (self.personal_best_position - self.position)
        # c2 * r2 * (gbest - x(t))
        social = c2 * r2 * (global_best_position - self.position)
        # v(t+1) = w * v(t) + c1 * r1 * (pbest - x(t)) + c2 * r2 * (gbest - x(t))
        self.velocity = w * self.velocity + cognitive + social
        # clip velocity to max 20% of search space
        self.velocity = np.clip(self.velocity, -self.max_velocity, self.max_velocity)

    def update_position(self):
        new_position = self.position + self.velocity
        for i in range(self.dimensions):
            if new_position[i] > self.bounds[i, 1]:  # upper bound violation
                # overshoot = new_position - upper_bound
                # reflected_position = upper_bound - overshoot
                overshoot = new_position[i] - self.bounds[i, 1]
                reflected_position = self.bounds[i, 1] - overshoot
                new_position[i] = reflected_position
                new_position[i] = np.clip(new_position[i], self.bounds[i, 0], self.bounds[i, 1])
                self.velocity[i] = -self.velocity[i]
            elif new_position[i] < self.bounds[i, 0]:  # lower bound violation
                overshoot = self.bounds[i, 0] - new_position[i]
                reflected_position = self.bounds[i, 0] + overshoot
                new_position[i] = reflected_position
                new_position[i] = np.clip(new_position[i], self.bounds[i, 0], self.bounds[i, 1])
                self.velocity[i] = -self.velocity[i]
        self.position = new_position

    def is_within_bounds(self):
        return np.all(self.position >= self.bounds[:, 0]) and np.all(self.position <= self.bounds[:, 1])


class ParticleSwarmOptimizer:
    def __init__(
        self,
        objective_func,
        bounds,
        num_particles=30,
        max_iter=1000,
        w_min=0.2,
        w_max=0.9,
        c1=2.0,
        c2=2.0,
        convergence_threshold=1e-6,
        patience=50,
        maximize=False,
        verbose=False
    ):
        self.objective_func = objective_func
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w_max  # initial weight is max
        self.w_min = w_min
        self.w_max = w_max
        self.c1 = c1
        self.c2 = c2
        self.convergence_threshold = convergence_threshold
        self.patience = patience
        self.maximize = maximize
        self.verbose = verbose

        assert bounds.shape[0] > 0, "Bounds must be at least one dimensional."
        assert num_particles > 0, "Number of particles must be a positive integer."
        assert max_iter > 0, "Maximum iterations must be a positive integer."
        assert w_min > 0 and w_max > 0, "Inertia weights must be non-negative."
        assert w_max > w_min, "And w_max must be greater than w_min."
        assert c1 > 0 and c2 > 0, "Cognitive and social coefficients must be positive."
        assert convergence_threshold > 0, "Convergence threshold must be positive."
        assert patience > 0, "Patience must be a positive integer."

        self.dimensions = bounds.shape[0]
        self.particles = [Particle(bounds) for _ in range(num_particles)]
        self.global_best_position = None
        self.global_best_value = None
        self.is_better_func = (lambda new, current: new > current) if maximize else (lambda new, current: new < current)
        self.best_value_history = []
        self.particle_velocities_history = []
        self.stagnation_counter = 0
        self.window = 10*self.dimensions

    def optimize(self):
        self.ensure_initialized()

        for i in range(self.max_iter):
            # particle iteration
            self.perform_particle_iteration()

            self.update_weight_linearly(i)
            self.best_value_history.append(self.global_best_value)

            if self.check_convergence():
                if self.verbose:
                    print(f"Convergence reached at iteration {i}.")
                break

            if self.verbose and i % (self.max_iter // 10) == 0:
                print(f"Iter {i}, Best Value: {self.global_best_value:.6f}, Best Position: {self.global_best_position}, Inertia Weight: {self.w:.4f}")
                diversity = self.calculate_diversity()
                print(f"Iter {i}, Best: {self.global_best_value:.6f}, Diversity: {diversity:.4f}")

        return self.global_best_position, self.global_best_value, self.best_value_history

    def update_weight_linearly(self, iteration):
        self.w = self.w_max - (self.w_max - self.w_min) * (iteration / self.max_iter)

    def update_global_best(self, value, particle):
        if self.global_best_position is None or self.is_better_func(value, self.global_best_value):
            self.global_best_value = value
            self.global_best_position = particle.position.copy()

    def check_convergence(self):
        if len(self.best_value_history) < self.window + 5:
            return False

        recent_window = self.best_value_history[-self.window:]
        older_window = self.best_value_history[-self.window * 2:-self.window]

        recent_best = min(recent_window) if not self.maximize else max(recent_window)
        older_best = min(older_window) if not self.maximize else max(older_window)

        improvement = abs(recent_best - older_best)

        if improvement < self.convergence_threshold:
            if self.verbose:
                print(f"Stagnation detected: Improvement {improvement:.10f} is below threshold {self.convergence_threshold:.10f}.")
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0

        return self.stagnation_counter >= self.patience

    def initialize_global_best(self):
        for particle in self.particles:
            value = self.objective_func(particle.position)
            self.update_global_best(value, particle)

    def ensure_initialized(self):
        if self.global_best_value is None:
            if self.verbose:
                print("Initializing global best value and position.")
            self.initialize_global_best()

    def perform_particle_iteration(self):
        for particle in self.particles:
            value = self.objective_func(particle.position)

            particle.update_personal_best(value, self.is_better_func)
            self.update_global_best(value, particle)

            particle.update_velocity(self.w, self.c1, self.c2, self.global_best_position)
            particle.update_position()

    def calculate_diversity(self):
        """Calculate swarm diversity to detect premature convergence."""
        positions = np.array([p.position for p in self.particles])
        center = np.mean(positions, axis=0)
        distances = [np.linalg.norm(pos - center) for pos in positions]
        return np.mean(distances)
