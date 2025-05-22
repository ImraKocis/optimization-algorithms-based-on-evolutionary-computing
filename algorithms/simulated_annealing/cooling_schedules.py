import numpy as np


def linear_cooling(T, k, T0=1.0, alpha=0.001):
    """
    Reduces temperature by a constant amount each iteration,
    providing steady, predictable cooling regardless of current temperature.
    """
    return max(T0 - alpha * k, 1e-10)


def logarithmic_cooling(T, k, T0=1.0):
    """
    Maintains exploration capability for extended period by very slowly
    transitioning to exploitation, theoretically optimal for finding
    global optima.
    """
    if k == 0:
        return T0  # avoid division by zero
    return T0 / np.log(k + 1)


def exponential_cooling(T, k, T0=1.0, beta=0.8):
    """
    Multiplies temperature by a constant factor less than 1,
    creating rapid initial cooling that slows down over time.
    """
    return T0 * (beta ** k)


def adaptive_cooling(T, k, acceptance_rate=0.44, adapt_factor=0.95, last_accepts=0, window=50, **kwargs):
    # Every 'window' iterations, adjust the temperature based on acceptance rate
    if (k+1) % window == 0 and last_accepts is not None:
        rate = last_accepts / window
        if rate > acceptance_rate:
            return T / adapt_factor  # Cool slower (raise T)
        else:
            return T * adapt_factor  # Cool faster (lower T)
    return T  # Otherwise, keep T unchanged


def custom_cooling(T, k, T0=1.0):
    """
    Custom cooling: Oscillating schedule to encourage periodic exploration bursts.
    Tk = T0 / (1 + 0.1*k) + 0.1 * T0 * np.sin(0.1 * k)
    Explanation: This schedule cools down but occasionally increases temperature,
    allowing the algorithm to periodically escape local minima.
    """
    return T0 / (1 + 0.1 * k) + 0.1 * T0 * np.sin(0.1 * k)


COOLING_SCHEDULES = {
    "linear": linear_cooling,
    "logarithmic": logarithmic_cooling,
    "exponential": exponential_cooling,
    "adaptive": adaptive_cooling,
    "custom": custom_cooling,
}