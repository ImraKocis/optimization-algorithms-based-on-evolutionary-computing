from enum import Enum


class SelectionMethod(Enum):
    TOURNAMENT = "tournament"
    ROULETTE_WHEEL = "roulette_wheel"


class CrossoverMethod(Enum):
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"


class MutationMethod(Enum):
    GAUSSIAN = "gaussian"
    ADAPTIVE_GAUSSIAN = "adaptive_gaussian"
    FLIP_BIT = "flip_bit"


def find_convergence_generation(best_fitness_history, stagnation_limit, minimize=False):
    if len(best_fitness_history) < 2:
        return 0

    last_improvement_gen = 0
    current_stagnation = 0
    best_so_far = best_fitness_history[0]

    for gen in range(1, len(best_fitness_history)):
        current_fitness = best_fitness_history[gen]

        if minimize:
            improved = current_fitness < best_so_far
        else:
            improved = current_fitness > best_so_far

        if improved:
            best_so_far = current_fitness
            last_improvement_gen = gen
            current_stagnation = 0
        else:
            current_stagnation += 1
            if current_stagnation >= stagnation_limit:
                break

    return last_improvement_gen
