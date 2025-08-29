from differential_evolution import differential_evolution
from evolutionary_strategies import evolutionary_strategies
from nsga2 import nsga2
from eva_core import filter_feasible_solutions
from pymoo.problems import get_problem
import matplotlib.pyplot as plt
import numpy as np

def run_bnh(verbose):
    problem_name = 'bnh'

    if verbose: print(f"---- Running {problem_name} experiment ----")

    problem = get_problem(problem_name)
    if verbose: print("Differential evolution... pop_size=100, generations=500")
    de_pop, de_fit = differential_evolution(problem, pop_size=100, generations=500, verbose=verbose)
    if verbose: print("Evolutionary strategies... pop_size=50, generations=100")
    es_pop, es_fit = evolutionary_strategies(problem, pop_size=50, generations=100, verbose=verbose)
    if verbose: print("NSGA-II... pop_size=50, generations=100")
    nsga2_pop, nsga2_fit = nsga2(problem, pop_size=50, generations=100, verbose=verbose)

    if verbose: print(f"---- Finished {problem_name} experiment ----")

    plot(problem, problem_name, de_pop, de_fit, es_pop, es_fit, nsga2_pop, nsga2_fit)


def run_osy(verbose):
    problem_name = 'osy'

    if verbose: print(f"---- Running {problem_name} experiment ----")

    problem = get_problem(problem_name)
    if verbose: print("Differential evolution... pop_size=100, generations=500")
    de_pop, de_fit = differential_evolution(problem, pop_size=100, generations=500, F=0.8, CR=0.9, verbose=verbose)
    if verbose: print("Evolutionary strategies... pop_size=100, generations=200")
    es_pop, es_fit = evolutionary_strategies(problem, pop_size=100, generations=200, verbose=verbose)
    if verbose: print("NSGA-II... pop_size=100, generations=200")
    nsga2_pop, nsga2_fit = nsga2(problem, pop_size=100, generations=200, verbose=verbose)

    if verbose: print(f"---- Finished {problem_name} experiment ----")

    plot(problem, problem_name, de_pop, de_fit, es_pop, es_fit, nsga2_pop, nsga2_fit)

def plot(problem, problem_name, de_pop, de_fit, es_pop, es_fit, nsga2_pop, nsga2_fit):

    # Get feasible solutions
    _, de_obj = filter_feasible_solutions(de_pop, de_fit)
    _, es_obj = filter_feasible_solutions(es_pop, es_fit)
    _, nsga2_obj = filter_feasible_solutions(nsga2_pop, nsga2_fit)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    if de_obj:
        de_obj = np.array(de_obj)
        plt.scatter(de_obj[:, 0], de_obj[:, 1], c='blue', label='DE', alpha=0.7)
    
    if es_obj:
        es_obj = np.array(es_obj)
        plt.scatter(es_obj[:, 0], es_obj[:, 1], c='green', label='ES', alpha=0.7)
    
    if nsga2_obj:
        nsga2_obj = np.array(nsga2_obj)
        plt.scatter(nsga2_obj[:, 0], nsga2_obj[:, 1], c='red', label='NSGA-II', alpha=0.7)
    
    # Pareto front
    pf = problem.pareto_front()
    if pf is not None:
        plt.plot(pf[:, 0], pf[:, 1], 'k-', linewidth=2, label='Pareto Front')
    
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.title(f'Algorithm Comparison - {problem_name.upper()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    verbose = True
    # run_bnh(verbose)
    run_osy(verbose)