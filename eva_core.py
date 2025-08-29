import numpy as np


def evaluate(problem, x):
    """
    Evaluates the fitness of a solution, if feasible.
    Otherwise, returns None.
    """
    # Vyhodnoceni
    F, G = problem.evaluate(x, return_values_of=["F", "G"])

    # Jsou splneny constrainty
    if G is not None and np.any(G > 0):
        return None

    # Vratim fitness
    return F


def dominates(f1, f2):
    """
    Check if f1 dominates f2 (Pareto dominance)
    f1 dominates f2 if f1 is better in all objectives and strictly better in at least one
    """
    if f1 is None or f2 is None:
        return False
    
    # For minimization: f1 dominates f2 if all f1[i] <= f2[i] and at least one f1[i] < f2[i]
    better_equal = np.all(f1 <= f2)
    strictly_better = np.any(f1 < f2)
    
    return better_equal and strictly_better


def print_progress(generation, pop_size, fitness_pop, algorithm_name="Algorithm", extra_info=None):
    """
    Print algorithm progress
    """
    feasible_count = sum(1 for f in fitness_pop if f is not None)
    print(f"Generation {generation}: {feasible_count}/{pop_size} feasible solutions", end="")
    
    if extra_info:
        print(f", {extra_info}", end="")
    
    print()


def filter_feasible_solutions(population, fitness_pop):
    """
    Filter out infeasible solutions and return feasible ones
    
    Returns:
        feasible_solutions: List of feasible solution vectors
        feasible_objectives: List of feasible objective values
    """
    feasible_solutions = []
    feasible_objectives = []
    
    for i, fit in enumerate(fitness_pop):
        if fit is not None:
            feasible_solutions.append(population[i])
            feasible_objectives.append(fit)
    
    return feasible_solutions, feasible_objectives


def pareto_ranking(objectives):
    """
    Simple Pareto ranking - assign rank based on domination
    Lower rank is better (0 = non-dominated)
    """
    n = len(objectives)
    ranks = [0] * n
    
    for i in range(n):
        for j in range(n):
            if i != j and dominates(objectives[j], objectives[i]):
                ranks[i] += 1
    
    return ranks


def initialize_population(problem, pop_size):
    """
    Initialize random population within problem bounds
    
    Returns:
        population: List of individuals
        fitness_pop: List of fitness values
    """
    population = []
    fitness_pop = []
    
    for n in range(pop_size):
        individual = np.random.uniform(problem.xl, problem.xu, problem.n_var)
        population.append(individual)
        
        fit = evaluate(problem, individual)
        fitness_pop.append(fit)
    
    return population, fitness_pop


def clip_to_bounds(individual, problem):
    """
    Ensure individual stays within problem bounds
    """
    return np.clip(individual, problem.xl, problem.xu)
