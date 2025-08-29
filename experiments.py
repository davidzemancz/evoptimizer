from differential_evolution import  differential_evolution_feasible
from evolutionary_strategies import evolutionary_strategies_feasible
from nsga2 import nsga2, nsga2_feasible
from eva_core import filter_feasible_solutions
from pymoo.problems import get_problem
import matplotlib.pyplot as plt
import numpy as np

def run_bnh(verbose):
    problem_name = 'bnh'

    if verbose: print(f"---- Running {problem_name} experiment ----")

    problem = get_problem(problem_name)
    if verbose: print("Differential evolution... pop_size=100, generations=500")
    de_pop, de_fit = differential_evolution_feasible(problem, pop_size=100, generations=500, verbose=verbose)
    if verbose: print("Evolutionary strategies... pop_size=50, generations=100")
    es_pop, es_fit = evolutionary_strategies_feasible(problem, pop_size=50, generations=100, verbose=verbose)
    if verbose: print("NSGA-II... pop_size=50, generations=100")
    nsga2_pop, nsga2_fit = nsga2_feasible(problem, pop_size=50, generations=100, verbose=verbose)

    if verbose: print(f"---- Finished {problem_name} experiment ----")

    plot_pareto_front(problem, problem_name, de_fit, es_fit, nsga2_fit)


def run_osy(verbose):
    problem_name = 'osy'

    if verbose: print(f"---- Running {problem_name} experiment ----")

    problem = get_problem(problem_name)
    if verbose: print("Differential evolution... pop_size=100, generations=500")
    de_pop, de_fit = differential_evolution_feasible(problem, pop_size=100, generations=100, verbose=verbose)
    if verbose: print("Evolutionary strategies... pop_size=100, generations=200")
    es_pop, es_fit = evolutionary_strategies_feasible(problem, pop_size=100, generations=300, verbose=verbose)
    if verbose: print("NSGA-II... pop_size=100, generations=200")
    nsga2_pop, nsga2_fit = nsga2_feasible(problem, pop_size=100, generations=300, verbose=verbose)

    if verbose: print(f"---- Finished {problem_name} experiment ----")

    plot_pareto_front(problem, problem_name, de_fit, es_fit, nsga2_fit)

def run_tnk(verbose):
    problem_name = 'tnk'

    if verbose: print(f"---- Running {problem_name} experiment ----")

    problem = get_problem(problem_name)
    if verbose: print("Differential evolution... pop_size=100, generations=500")
    de_pop, de_fit = differential_evolution_feasible(problem, pop_size=100, generations=500, verbose=verbose)
    if verbose: print("Evolutionary strategies... pop_size=50, generations=100")
    es_pop, es_fit = evolutionary_strategies_feasible(problem, pop_size=50, generations=100, verbose=verbose)
    if verbose: print("NSGA-II... pop_size=50, generations=100")
    nsga2_pop, nsga2_fit = nsga2_feasible(problem, pop_size=50, generations=100, verbose=verbose)

    if verbose: print(f"---- Finished {problem_name} experiment ----")

    plot_pareto_front(problem, problem_name, de_fit, es_fit, nsga2_fit)

def plot_pareto_front(problem, problem_name,  de_fit,  es_fit,  nsga2_fit):

    # Algorithms now return only feasible solutions
    de_obj = de_fit
    es_obj = es_fit
    nsga2_obj = nsga2_fit
    
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
    
    # Pareto fronta
    pf = problem.pareto_front()
    if pf is not None:
        plt.plot(pf[:, 0], pf[:, 1], 'k-', linewidth=2, label='Pareto Front')
    
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.title(f'Algorithm Comparison - {problem_name.upper()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # ulozim plot
    # plt.savefig(f"algorithm_comparison_{problem_name}.png")
    # plt.close()

    # Calculate and print quality metrics
    # print_quality_metrics(problem, de_obj, es_obj, nsga2_obj)


def print_quality_metrics(problem, problem_name, de_obj, es_obj, nsga2_obj):

    pf = problem.pareto_front()
    if pf is None:
        print("No reference Pareto front available for quality assessment")
        return

    print(f"---- Quality Metrics for {problem_name} ----")

    algorithms = [
        ("DE", de_obj),
        ("ES", es_obj), 
        ("NSGA-II", nsga2_obj)
    ]
    
    for alg_name, solutions in algorithms:
        solutions = np.array(solutions)
        
        # Spocitam metriky
        gd = generational_distance(solutions, pf)
        igd = inverted_generational_distance(solutions, pf)
        spread = diversity_spread(solutions)

        print(f"{alg_name:8} | #: {len(solutions):4} | GD: {gd:.4f} | IGD: {igd:.4f} | Spread: {spread:.4f}")


    # print("="*60)
    # print("Lower GD/IGD = better convergence")
    # print("Lower Spread = better diversity")
    # print("="*60)


def generational_distance(solutions, pareto_front):

    # Spocitam minimalni vzdalenost kazdeho jedince k pareto fronte a vratim prumer
    min_distances = []
    for solution in solutions:
        distances = np.sqrt(np.sum((pareto_front - solution) ** 2, axis=1))
        min_distances.append(np.min(distances))
    
    return np.mean(min_distances)


def inverted_generational_distance(solutions, pareto_front):

    # Spocitam minimalni vzdalenost kazdeho z pareto fronty k resenim a vratim prumer

    min_distances = []
    for pf_point in pareto_front:
        distances = np.sqrt(np.sum((solutions - pf_point) ** 2, axis=1))
        min_distances.append(np.min(distances))
    
    return np.mean(min_distances)

def diversity_spread(solutions):

    # Setridim podle prvni objective
    sorted_solutions = solutions[np.argsort(solutions[:, 0])]
    distances = []

    # Vypocitam vzdalenosti mezi sousednimi resenimi
    for i in range(1, len(sorted_solutions)):
        dist = np.sqrt(np.sum((sorted_solutions[i] - sorted_solutions[i-1]) ** 2))
        distances.append(dist)
    
    # Spocitam spread jako stdev distance
    mean_dist = np.mean(distances)
    spread = np.sqrt(np.mean((distances - mean_dist) ** 2))
    
    return spread


if __name__ == "__main__":
    verbose = True
    run_bnh(verbose)
    # run_osy(verbose)
    # run_tnk(verbose)