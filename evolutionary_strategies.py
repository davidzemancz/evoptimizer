import numpy as np
from eva_core import evaluate, dominates, print_progress, filter_feasible_solutions, environmental_selection, clip_to_bounds


def evolutionary_strategies(problem, pop_size=100, generations=300, verbose=False):
    """
    Optimized Evolutionary Strategies (μ+λ) with adaptive operators
    
    Args:
        problem: Problem instance from pymoo
        pop_size: Population size (μ = pop_size, λ = pop_size)
        generations: Number of generations
        verbose: Print progress
        
    Returns:
        population: Final population
        fitness_pop: Final fitness values
    """
    
    mu = pop_size  # Parent population size
    lambda_size = pop_size  # Offspring population size
    
    # Populace
    population = np.random.uniform(problem.xl, problem.xu, (mu, problem.n_var))
    
    # Optimalizacni parametry
    sigma_range = 0.1 * (problem.xu - problem.xl)
    sigma_pop = np.tile(sigma_range, (mu, 1))
    
    # Predpocitam fintss
    fitness_pop = [evaluate(problem, individual) for individual in population]
    
    # Adaptivni parametry
    tau = 1.0 / np.sqrt(2 * np.sqrt(problem.n_var))  # Global learning rate
    tau_prime = 1.0 / np.sqrt(2 * problem.n_var)     # Local learning rate
    search_center = (problem.xl + problem.xu) / 2
    center_bias = 0.1
    sigma_min = 1e-5 * (problem.xu - problem.xl)
    
    # Evoluce
    for g in range(generations):
        # Vyberu parenty
        parent_indices = np.random.randint(mu, size=lambda_size)
        parents = population[parent_indices]
        parent_sigmas = sigma_pop[parent_indices]
        
        # Self-adaptatace
        global_factors = np.exp(tau * np.random.normal(size=lambda_size))
        local_factors = np.exp(tau_prime * np.random.normal(size=(lambda_size, problem.n_var)))
        
        # Prepocitam sigmy
        new_sigmas = parent_sigmas * global_factors[:, np.newaxis] * local_factors
        new_sigmas = np.maximum(new_sigmas, sigma_min)
        
        # Biased mutace
        bias_vectors = center_bias * (search_center - parents) / (problem.xu - problem.xl)
        mutations = np.random.normal(0, 1, (lambda_size, problem.n_var)) * new_sigmas
        offspring = parents + mutations + bias_vectors

        # Kontrola boundu
        offspring = np.clip(offspring, problem.xl, problem.xu)
        
        # Spoctam fitness offspringu
        offspring_fitness = [evaluate(problem, individual) for individual in offspring]
        
        # Vyberu nejlepsi z populace
        combined_pop = np.vstack([population, offspring])
        combined_sigma = np.vstack([sigma_pop, new_sigmas])
        combined_fitness = fitness_pop + offspring_fitness
        selected_indices = environmental_selection(combined_fitness, mu)
        population = combined_pop[selected_indices]
        sigma_pop = combined_sigma[selected_indices]
        fitness_pop = [combined_fitness[i] for i in selected_indices]
        
        # Print progress (reduced frequency for speed)
        if verbose and (g + 1) % 100 == 0:
            avg_sigma = np.mean(sigma_pop)
            extra_info = f"avg σ: {avg_sigma:.4f}"
            print_progress(g + 1, mu, fitness_pop, "ES", extra_info)

    return population, fitness_pop


