class OptimizationHistory:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x_values = []
        self.f_values = []
        self.gradients = []
        self.gradient_norms = []

    def record_iteration(self, x, f_val, gradient, grad_norm):
        self.x_values.append(x.copy())
        self.f_values.append(f_val)
        if gradient.size > 0:  # Skip empty gradient for initial point
            self.gradients.append(gradient.copy())
            self.gradient_norms.append(grad_norm)
