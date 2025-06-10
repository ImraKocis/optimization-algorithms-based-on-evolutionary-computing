import itertools
from dataclasses import dataclass
from typing import Dict

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from algorithms.ga.ga import GeneticAlgorithm
from algorithms.ga.population import RandomContinuousInitializer
from algorithms.ga.utils import SelectionMethod, CrossoverMethod, MutationMethod
from utils.objective_functions import rastrigin_objective_function


@dataclass
class ExperimentResult:
    params: Dict[str, any]
    best_fitness: float
    generations: int
    convergence_gen: int


class GAExperiment:
    def __init__(self, dimensions: int = 10, bounds: (float, float) = (-5.12, 5.12), runs_per_config: int = 5):
        self.dimensions = dimensions
        self.bounds = bounds
        self.runs_per_config = runs_per_config

        self.population_sizes = [50, 100, 200]
        self.tournament_sizes = [2, 5, 7]
        self.mutation_rates = [0.01, 0.05, 0.1]
        self.crossover_rates = [0.5, 0.8, 0.99]

        self.selection_methods = {
            'tournament': SelectionMethod.TOURNAMENT,
            'roulette_wheel': SelectionMethod.ROULETTE_WHEEL,
        }

        self.crossover_methods = {
            'single_point': CrossoverMethod.SINGLE_POINT,
            'two_point': CrossoverMethod.TWO_POINT,
            'uniform': CrossoverMethod.UNIFORM,
        }

    def run_single_experiment(self, params: Dict[str, any]) -> ExperimentResult:
        ga_exp = GeneticAlgorithm(
            population_size=params['population_size'],
            crossover_rate=params['crossover_rate'],
            mutation_rate=params['mutation_rate'],
            elitism_count=3,
            max_generations=300,
            stagnation_limit=100,
            population_initializer=RandomContinuousInitializer(self.dimensions, self.bounds),
            selection_method=self.selection_methods[params['selection_method']],
            crossover_method=self.crossover_methods[params['crossover_method']],
            mutation_method=MutationMethod.GAUSSIAN,
            tournament_size=params['tournament_size'],
            mutation_sigma=0.15,
            minimize=True,
            verbose=False
        )

        best_solution_ga = ga_exp.evolve(rastrigin_objective_function)
        convergence_info = ga_exp.get_convergence_info()
        return ExperimentResult(
            params=params,
            best_fitness=best_solution_ga.fitness,
            generations=convergence_info['total_generations'],
            convergence_gen=convergence_info['convergence_generation']
        )

    def run_experiments(self) -> pd.DataFrame:
        results = []

        param_combinations = [
            {
                'population_size': ps,
                'tournament_size': ts,
                'mutation_rate': mr,
                'crossover_rate': cr,
                'selection_method': st,
                'crossover_method': cm
            }
            for ps, ts, mr, cr, st, cm in itertools.product(
                self.population_sizes,
                self.tournament_sizes,
                self.mutation_rates,
                self.crossover_rates,
                self.selection_methods,
                self.crossover_methods
            )
        ]

        total_runs = len(param_combinations) * self.runs_per_config
        with tqdm(total=total_runs, desc="Running GA Experiments") as pbar:
            for params in param_combinations:
                for _ in range(self.runs_per_config):
                    result = self.run_single_experiment(params)
                    results.append(result)
                    pbar.update(1)

        df = pd.DataFrame([
            {
                **r.params,
                'best_fitness': r.best_fitness,
                'generations': r.generations,
                'convergence_gen': r.convergence_gen
            }
            for r in results
        ])

        return df

    @staticmethod
    def analyze_results(results_df: pd.DataFrame):
        print("\nTop 5 Best Performing Configurations:")
        best_configs = results_df.groupby(
            ['population_size', 'tournament_size', 'mutation_rate',
             'crossover_rate', 'selection_method', 'crossover_method']
        )['best_fitness'].mean().sort_values()
        print(best_configs.head())

        print("\nTop 5 Worst Performing Configurations:")
        print(best_configs.tail())

        plt.figure(figsize=(20, 10))

        plt.subplot(2, 3, 1)
        sns.boxplot(data=results_df, x='population_size', y='best_fitness')
        plt.title('Population Size Impact')

        plt.subplot(2, 3, 2)
        sns.boxplot(data=results_df, x='mutation_rate', y='best_fitness')
        plt.title('Mutation Rate Impact')

        plt.subplot(2, 3, 3)
        sns.boxplot(data=results_df, x='crossover_rate', y='best_fitness')
        plt.title('Crossover Rate Impact')

        plt.subplot(2, 3, 4)
        sns.boxplot(data=results_df, x='tournament_size', y='best_fitness')
        plt.title('Tournament Size Impact')

        plt.subplot(2, 3, 5)
        sns.boxplot(data=results_df, x='selection_method', y='best_fitness')
        plt.title('Selection Type Impact')

        plt.subplot(2, 3, 6)
        sns.boxplot(data=results_df, x='crossover_method', y='best_fitness')
        plt.title('Crossover Type Impact')

        plt.tight_layout()
        plt.show()
