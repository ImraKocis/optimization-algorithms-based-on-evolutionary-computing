import numpy as np


class Particle:
    def __init__(self, bounds, algo_type='minimizer'):
        self.dimensions = bounds.shape[0]
        self.position = np.random.uniform(bounds[:, 0], bounds[:, 1], self.dimensions)
        self.velocity = np.random.uniform(-abs(bounds[:, 1] - bounds[:, 0]), abs(bounds[:, 1] - bounds[:, 0]), self.dimensions)
        self.personal_best_position = self.position.copy()
        self.algo_type = algo_type
        if self.algo_type == 'maximizer':
            self.personal_best_value = 0
        elif self.algo_type == 'minimizer':
            self.personal_best_value = np.inf
        else:
            raise ValueError("Invalid algorithm type. Use 'minimizer' or 'maximizer'.")

    def update_personal_best(self, value):
        if self.algo_type == 'maximizer':
            if value > self.personal_best_value:
                self.personal_best_value = value
                self.personal_best_position = self.position.copy()

        else:
            if value < self.personal_best_value:
                self.personal_best_value = value
                self.personal_best_position = self.position.copy()


class ParticleSwarmOptimizer:
    def __init__(
        self,
        objective_func,
        bounds,
        num_particles=30,
        max_iter=1000,
        w=0.9,
        c1=2,
        c2=2,
        linear_weight_decay=True,
        linear_weight_decay_alpha=0.0005,
        algo_type='minimizer'
    ):
        self.objective_func = objective_func
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.initial_w = w
        self.c1 = c1
        self.c2 = c2
        self.linear_weight_decay = linear_weight_decay
        self.linear_weight_decay_alpha = linear_weight_decay_alpha
        self.algo_type = algo_type

        self.particles = [Particle(bounds) for _ in range(num_particles)]
        self.global_best_position = None
        if self.algo_type == 'maximizer':
            self.global_best_value = 0
        elif self.algo_type == 'minimizer':
            self.global_best_value = np.inf
        else:
            raise ValueError("Invalid algorithm type. Use 'minimizer' or 'maximizer'.")
        self.history = []

    def optimize(self, verbose=False):
        for iter in range(self.max_iter):
            for particle in self.particles:
                value = self.objective_func(particle.position)
                particle.update_personal_best(value)
                self.update_global_best(value, particle)

            for particle in self.particles:
                self.update_particle_velocity(particle)
                self.update_particle_position(particle)

            # Update inertia weight linearly if requested
            self.update_weight_linearly(iter)
            self.history.append(self.global_best_value)
            if verbose and iter % (self.max_iter // 10) == 0:
                print(f"Iter {iter}, Best Value: {self.global_best_value:.6f}, Best Position: {self.global_best_position}, Inertia Weight: {self.w:.4f}")

        return self.global_best_position, self.global_best_value, self.history

    def update_particle_velocity(self, particle):
        # v(t+1) = w * v(t) + c1 * r1 * (pbest - x(t)) + c2 * r2 * (gbest - x(t))
        r1 = np.random.rand(particle.dimensions)
        r2 = np.random.rand(particle.dimensions)
        cognitive = self.c1 * r1 * (particle.personal_best_position - particle.position)
        social = self.c2 * r2 * (self.global_best_position - particle.position)
        particle.velocity = self.w * particle.velocity + cognitive + social

    def update_particle_position(self, particle):
        # x(t+1) = x(t) + v(t+1)
        particle.position += particle.velocity
        particle.position = np.clip(particle.position, self.bounds[:, 0], self.bounds[:, 1])

    def update_weight_linearly(self, iteration):
        if not self.linear_weight_decay:
            return
        w_min, w_max = 0.2, 0.9
        new_w = self.initial_w - (self.linear_weight_decay_alpha * iteration)
        self.w = np.clip(new_w, w_min, w_max)

    def update_global_best(self, value, particle):
        if self.algo_type == 'maximizer':
            if value > self.global_best_value:
                self.global_best_value = value
                self.global_best_position = particle.position.copy()
        elif self.algo_type == 'minimizer':
            if value < self.global_best_value:
                self.global_best_value = value
                self.global_best_position = particle.position.copy()
