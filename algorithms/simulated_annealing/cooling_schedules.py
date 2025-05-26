import numpy as np


def linear_cooling(T, k, T0, alpha=0.001, verbose=False):
    """
    Reduces temperature by a constant amount each iteration,
    providing steady, predictable cooling regardless of current temperature.
    """
    res = T0 - alpha * k
    print(f"Linear cooling at iteration {k}: temperature={res:.4f}") if verbose else None
    return res


def logarithmic_cooling(T, k, T0, verbose=False):
    """
    Maintains exploration capability for extended period by very slowly
    transitioning to exploitation, theoretically optimal for finding
    global optima.
    """
    if k == 0:
        return T0*2  # avoid division by zero, and make huge jump at k=0
    res = T0 / np.log(k + 1)
    print(f"Logarithmic cooling at iteration {k}: temperature={res:.4f}, log_value={np.log(k+1)}") if verbose else None
    return res


def exponential_cooling(T, k, T0, beta=0.95, verbose=False):
    """
    Multiplies temperature by a constant factor less than 1,
    creating rapid initial cooling that slows down over time.
    """
    res = T0 * (beta ** k)
    print(f"Exponential cooling at iteration {k}: temperature={res:.4f}") if verbose else None
    return res


def adaptive_cooling(T, k, acceptance_rate=0.44, adapt_factor=0.95, last_accepts=0, window=50, verbose=False):
    # Every 'window' iterations, adjust the temperature based on acceptance rate
    res = T
    if (k+1) % window == 0 and last_accepts is not None:
        rate = last_accepts / window
        if rate > acceptance_rate:
            res = T / adapt_factor  # Cool slower (raise T)
            print(f"Adaptive cooling at iteration {k}: temperature={res:.4f}, acceptance rate={rate:.2f}") if verbose else None
            return res
        else:
            res = T * adapt_factor  # Cool faster (lower T)
            print(f"Adaptive cooling at iteration {k}: temperature={res:.4f}, acceptance rate={rate:.2f}") if verbose else None
            return res
    print(f"Adaptive cooling at iteration {k}: temperature={res:.4f}, no adjustment") if verbose else None
    return res  # Otherwise, keep T unchanged


def custom_cooling(T, k, T0, verbose=False):
    """
    Custom cooling: Oscillating schedule to encourage periodic exploration bursts.
    Tk = T0 / (1 + 0.1*k) + 0.1 * T0 * np.sin(0.1 * k)
    Explanation: This schedule cools down but occasionally increases temperature,
    allowing the algorithm to periodically escape local minima.
    """
    res = T0 / (1 + 0.1 * k) + 0.1 * T0 * np.sin(0.1 * k)
    print(f"Custom cooling at iteration {k}: temperature={res:.4f}") if verbose else None
    return res


COOLING_SCHEDULES = {
    "linear": linear_cooling,
    "logarithmic": logarithmic_cooling,
    "exponential": exponential_cooling,
    "adaptive": adaptive_cooling,
    "custom": custom_cooling,
}