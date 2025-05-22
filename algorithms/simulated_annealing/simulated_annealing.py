import numpy as np

from algorithms.simulated_annealing.cooling_schedules import linear_cooling, COOLING_SCHEDULES
from utils.objective_functions import rastrigin_objective_function

"""
kako radi algo:

Inicijaliziramo pocetnu temp, broj iteracija, step size te zeljenu funkciju hladenja i iterirammo,
u nasem slucaju sve dok ne dodemo do n iteracija.

U svakoj iteraciji kreiraom novog kandidata koji ovisi o trenutnom rjesenju i step size-u. Koji
je uvijek u obliku [num] * dimensions - to su koordiante novog kandidata u prostoru.

Nakon toga ubacimo kandidata (koordiante) u zeljenu objective funckiju te dobijemo vrijednost te točke

Nakon toga algo odlucuje zeli li prihvatiti kandidata tako što će napraviti usporedbu izmedu trenutne najbolje
vrijednosti i trenutne vrijednosti koju smo dobili od objective funckije - singularne vrijednosti.

Ukoliko je, u nasem slucaju, delta manja od 0, automatksi znamo da smo nasli bolju točku tj. vrijednost i 
automatski cemo ju prihvatiti.

Ukoliko je delta veća od 0. Tada nam simulated annealing dolzai sa svojom logikom te radimo usporedbu
nekog radnom broja [0, 1) i exp(-delta / T) - gdje je delta razlika izmedu vrijednosti kandidata i trenutne najbolje
vrijednosti, a T je trenutna temperatura.

Što je delta manja i temperatura veća, to je veća vjerojatnost da ćemo prihvatiti kandidata. 

Ukoliko je temperatura jako visoka doči ćemo do faze da konst. prihvaćamo kandidate bez obzira na njihovu vrijednost. 
Opet ako je temp premala nećemo više vjerjatno prihvaćati niti jednog kandidata. Kod ekstrema u temperaturi delta nam
vise ne igra veliku ulogu.

Ako je random broj mani od probabilitija, uzimamo vrijednost kandidata i postavljmo tog kandidata kao najbolju 
vrijednost.
"""


def simulated_annealing(
    objective_func,
    bounds,
    n_iterations=1000,
    step_size=0.1,
    T0=1.0,
    cooling_schedule="linear",
    cooling_kwargs=None,
    verbose=False
):
    dimensions = bounds.shape[0]
    solution = np.random.uniform(bounds[:, 0], bounds[:, 1], size=dimensions)
    value = objective_func(solution)
    best_solution = solution.copy()
    best_value = value
    history = [best_value]
    T = T0
    accepted = 0

    cooling_func = COOLING_SCHEDULES.get(cooling_schedule, linear_cooling)
    if cooling_kwargs is None:
        cooling_kwargs = {}

    window = cooling_kwargs.get('window', 50)

    for k in range(n_iterations):
        candidate = solution + np.random.randn(dimensions) * step_size
        candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
        candidate_value = objective_func(candidate)
        delta = candidate_value - value

        print(f"Iteration {k}: Current Value: {value}, Candidate Value: {candidate_value}, Delta: {delta}, Current Temp: {T}")

        print(f"Probability: {np.exp(-delta / T) * 100:.2f}%")

        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            solution = candidate
            value = candidate_value
            accepted += 1
            if value < best_value:
                best_solution = solution.copy()
                best_value = value

        history.append(best_value)

        # Update temperature
        if cooling_schedule == "adaptive":
            if (k + 1) % window == 0:
                cooling_kwargs['last_accepts'] = accepted
                T = cooling_func(T, k, **cooling_kwargs)
                if verbose:
                    print(f"Iter {k}, T={T:.4f}, accepted in window={accepted}, best_value={best_value:.6f}")
                accepted = 0  # Reset for next window
            else:
                pass
        elif cooling_schedule == "logarithmic":
            T = cooling_func(T, k, T0=T0)
        else:
            print("Other cooling")
            T = cooling_func(T, k, T0=T0, **cooling_kwargs)

        if verbose and cooling_schedule != "adaptive" and (k % (n_iterations // 10) == 0):
            print(f"Iter {k}, T={T:.4f}, best_value={best_value:.6f}")

    return best_solution, best_value, history


# for testing:
# bounds = np.array([[-5.12, 5.12]] * 2)
# best_sol, best_val, history = simulated_annealing(
#         rastrigin_objective_function,
#         bounds,
#         n_iterations=100,
#         step_size=0.4,
#         T0=5.0,
#         cooling_schedule="adaptive",
#         cooling_kwargs={'acceptance_rate': 0.44, 'adapt_factor': 0.95, 'window': 20, 'last_accepts': 0},
#         verbose=True
#     )
# print("Best Solution:", best_sol)
# print("Best Value:", best_val)
