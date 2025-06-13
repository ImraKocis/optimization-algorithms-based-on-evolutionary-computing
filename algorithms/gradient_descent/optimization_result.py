class OptimizationResult:
    def __init__(self, x_optimal, f_optimal, iterations, converged, history):
        self.x_optimal = x_optimal
        self.f_optimal = f_optimal
        self.iterations = iterations
        self.converged = converged
        self.history = history
