import numpy as np

from utils.helpers.min_max import MinMax


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


def adaptive_cooling(
    T,
    k,
    T0,
    current_acceptance_rate,
    target_acceptance_rate=0.44,
    learning_rate=0.1,
    cooling_rate=0.95,
    optimal_range=MinMax(0.4, 0.5),
    max_temperature_change=MinMax(-0.5, 0.5),
    T_min=1e-6,
    verbose=False
):
    """
    Adaptive cooling schedule:
    - If acceptance rate is within optimal range, cool by base rate.
    - If acceptance rate is too low, heat up.
    - If acceptance rate is too high, cool down faster.
    - Temperature change is clamped to max_temperature_change.
    """
    assert T >= 0, "temperature must be non-negative"
    assert T0 >= 0, "initial temperature must be non-negative"
    assert 0 < target_acceptance_rate < 1
    assert learning_rate > 0
    assert 0 < cooling_rate < 1
    assert 0 <= optimal_range.min < optimal_range.max <= 1
    assert -1 < max_temperature_change.min < 0 < max_temperature_change.max < 1
    assert optimal_range.min <= target_acceptance_rate <= optimal_range.max

    T_max = T0 * 1.44

    if current_acceptance_rate > optimal_range.max:
        # Too much exploration, cool down
        T = T * (1 - learning_rate * (current_acceptance_rate - target_acceptance_rate))
    elif current_acceptance_rate < optimal_range.min:
        # Too little exploration, heat up
        T = T * (1 + learning_rate * (target_acceptance_rate - current_acceptance_rate))
    else:
        # In the optimal range, cool normally
        T = T * cooling_rate
    print(f"Adaptive cooling at iteration {k}: temperature={max(T_min, min(T_max, T)):.4f}") if verbose else None
    return max(T_min, min(T_max, T))


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