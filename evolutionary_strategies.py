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
    
    # Vectorized initialization
    population = np.random.uniform(problem.xl, problem.xu, (mu, problem.n_var))
    
    # Initialize strategy parameters (sigma for each variable) - vectorized
    sigma_range = 0.1 * (problem.xu - problem.xl)
    sigma_pop = np.tile(sigma_range, (mu, 1))
    
    # Evaluate initial population - batch if possible
    fitness_pop = [evaluate(problem, individual) for individual in population]
    
    # Pre-compute adaptive parameters
    tau = 1.0 / np.sqrt(2 * np.sqrt(problem.n_var))  # Global learning rate
    tau_prime = 1.0 / np.sqrt(2 * problem.n_var)     # Local learning rate
    search_center = (problem.xl + problem.xu) / 2
    center_bias = 0.1
    sigma_min = 1e-6 * (problem.xu - problem.xl)
    
    # Evolution loop
    for g in range(generations):
        # Vectorized parent selection
        parent_indices = np.random.randint(mu, size=lambda_size)
        parents = population[parent_indices]
        parent_sigmas = sigma_pop[parent_indices]
        
        # Vectorized self-adaptation
        global_factors = np.exp(tau * np.random.normal(size=lambda_size))
        local_factors = np.exp(tau_prime * np.random.normal(size=(lambda_size, problem.n_var)))
        
        # Update sigma with minimum threshold - vectorized
        new_sigmas = parent_sigmas * global_factors[:, np.newaxis] * local_factors
        new_sigmas = np.maximum(new_sigmas, sigma_min)
        
        # Vectorized biased mutation
        bias_vectors = center_bias * (search_center - parents) / (problem.xu - problem.xl)
        mutations = np.random.normal(0, 1, (lambda_size, problem.n_var)) * new_sigmas
        offspring = parents + mutations + bias_vectors
        
        # Vectorized bounds clipping
        offspring = np.clip(offspring, problem.xl, problem.xu)
        
        # Evaluate offspring (this is the bottleneck - can't vectorize easily)
        offspring_fitness = [evaluate(problem, individual) for individual in offspring]
        
        # Combine populations - use numpy operations where possible
        combined_pop = np.vstack([population, offspring])
        combined_sigma = np.vstack([sigma_pop, new_sigmas])
        combined_fitness = fitness_pop + offspring_fitness
        
        # Environmental selection
        selected_indices = environmental_selection(combined_fitness, mu)
        
        # Update population using advanced indexing
        population = combined_pop[selected_indices]
        sigma_pop = combined_sigma[selected_indices]
        fitness_pop = [combined_fitness[i] for i in selected_indices]
        
        # Print progress (reduced frequency for speed)
        if verbose and (g + 1) % 100 == 0:
            avg_sigma = np.mean(sigma_pop)
            extra_info = f"avg σ: {avg_sigma:.4f}"
            print_progress(g + 1, mu, fitness_pop, "ES", extra_info)

    return population, fitness_pop


def evolutionary_strategies_fast(problem, pop_size=50, generations=500, verbose=False):
    """
    Fast ES variant with simplified operations and (μ,λ) selection
    
    Args:
        problem: Problem instance from pymoo
        pop_size: Population size (μ)
        generations: Number of generations
        verbose: Print progress
        
    Returns:
        population: Final population
        fitness_pop: Final fitness values
    """
    
    mu = pop_size  # Parent population size
    lambda_size = pop_size * 2  # More offspring for better exploration
    
    # Vectorized initialization
    population = np.random.uniform(problem.xl, problem.xu, (mu, problem.n_var))
    
    # Single global sigma for all variables (faster)
    sigma = 0.1 * np.mean(problem.xu - problem.xl)
    sigma_min = 1e-6 * np.mean(problem.xu - problem.xl)
    
    # Evaluate initial population
    fitness_pop = [evaluate(problem, individual) for individual in population]
    
    # Pre-compute constants
    search_center = (problem.xl + problem.xu) / 2
    center_bias = 0.05  # Reduced bias for speed
    
    # Evolution loop
    for g in range(generations):
        # Generate all offspring at once - vectorized
        parent_indices = np.random.randint(mu, size=lambda_size)
        parents = population[parent_indices]
        
        # Simple sigma adaptation (1/5th rule approximation)
        if g > 0 and g % 20 == 0:
            success_rate = len([f for f in fitness_pop if f is not None]) / mu
            if success_rate > 0.2:
                sigma *= 1.1
            else:
                sigma *= 0.9
            sigma = max(sigma, sigma_min)
        
        # Vectorized mutation with bias
        bias_vectors = center_bias * (search_center - parents) / (problem.xu - problem.xl)
        mutations = np.random.normal(0, sigma, (lambda_size, problem.n_var))
        offspring = np.clip(parents + mutations + bias_vectors, problem.xl, problem.xu)
        
        # Evaluate offspring
        offspring_fitness = [evaluate(problem, individual) for individual in offspring]
        
        # (μ,λ) selection - only from offspring (faster than μ+λ)
        feasible_indices = [i for i, f in enumerate(offspring_fitness) if f is not None]
        
        if len(feasible_indices) >= mu:
            # Enough feasible solutions - select best
            feasible_fitness = [offspring_fitness[i] for i in feasible_indices]
            feasible_offspring = offspring[feasible_indices]
            
            # Simple selection by first objective (faster than full Pareto ranking)
            if len(feasible_fitness[0]) == 1:
                # Single objective
                sorted_pairs = sorted(zip(feasible_offspring, feasible_fitness), 
                                    key=lambda x: x[1][0])
            else:
                # Multi-objective - use sum of objectives as approximation
                sorted_pairs = sorted(zip(feasible_offspring, feasible_fitness), 
                                    key=lambda x: np.sum(x[1]))
            
            population = np.array([pair[0] for pair in sorted_pairs[:mu]])
            fitness_pop = [pair[1] for pair in sorted_pairs[:mu]]
        else:
            # Not enough feasible - use environmental selection from all
            combined_fitness = offspring_fitness
            selected_indices = environmental_selection(combined_fitness, mu)
            population = offspring[selected_indices]
            fitness_pop = [combined_fitness[i] for i in selected_indices]
        
        # Print progress (less frequent)
        if verbose and (g + 1) % 100 == 0:
            extra_info = f"σ: {sigma:.4f}"
            print_progress(g + 1, mu, fitness_pop, "ES-Fast", extra_info)
    
    return population, fitness_pop


# Remove duplicate functions - they are now in eva_core.py
