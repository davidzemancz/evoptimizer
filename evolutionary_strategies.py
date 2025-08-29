import numpy as np
from eva_core import evaluate, dominates, print_progress, filter_feasible_solutions, environmental_selection, clip_to_bounds


def evolutionary_strategies(problem, pop_size=100, generations=300, verbose=False):
    """
    Simple Evolutionary Strategies (μ+λ) with adaptive operators
    
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
    
    # Initialize population with strategy parameters (sigma for each variable)
    population = []
    sigma_pop = []
    
    for n in range(mu):
        # Individual
        individual = np.random.uniform(problem.xl, problem.xu, problem.n_var)
        population.append(individual)
        
        # Strategy parameters (mutation strengths) - one per variable
        sigma = np.full(problem.n_var, 0.1 * (problem.xu - problem.xl))
        sigma_pop.append(sigma)
    
    # Evaluate initial population
    fitness_pop = []
    for i in range(mu):
        fit = evaluate(problem, population[i])
        fitness_pop.append(fit)
    
    # Adaptive parameters
    tau = 1.0 / np.sqrt(2 * np.sqrt(problem.n_var))  # Global learning rate
    tau_prime = 1.0 / np.sqrt(2 * problem.n_var)     # Local learning rate
    
    # Evolution loop
    for g in range(generations):
        offspring = []
        offspring_sigma = []
        offspring_fitness = []
        
        # Generate offspring
        for _ in range(lambda_size):
            # Select random parent
            parent_idx = np.random.randint(mu)
            parent = population[parent_idx].copy()
            parent_sigma = sigma_pop[parent_idx].copy()
            
            # Adapt strategy parameters first (self-adaptation)
            global_factor = np.exp(tau * np.random.normal())
            local_factors = np.exp(tau_prime * np.random.normal(size=problem.n_var))
            
            # Update sigma with minimum threshold
            new_sigma = parent_sigma * global_factor * local_factors
            new_sigma = np.maximum(new_sigma, 1e-6 * (problem.xu - problem.xl))
            
            # Biased mutation - bias towards center of search space
            center_bias = 0.1  # Bias strength
            search_center = (problem.xl + problem.xu) / 2
            bias_vector = center_bias * (search_center - parent) / (problem.xu - problem.xl)
            
            # Apply mutation with bias
            mutation = np.random.normal(0, 1, problem.n_var) * new_sigma
            offspring_individual = parent + mutation + bias_vector
            
            # Ensure bounds
            offspring_individual = clip_to_bounds(offspring_individual, problem)
            
            # Evaluate offspring
            offspring_fit = evaluate(problem, offspring_individual)
            
            offspring.append(offspring_individual)
            offspring_sigma.append(new_sigma)
            offspring_fitness.append(offspring_fit)
        
        # (μ+λ) selection: combine parents and offspring
        combined_pop = population + offspring
        combined_sigma = sigma_pop + offspring_sigma
        combined_fitness = fitness_pop + offspring_fitness
        
        # Select best μ individuals using environmental selection
        selected_indices = environmental_selection(combined_fitness, mu)
        
        # Update population
        population = [combined_pop[i] for i in selected_indices]
        sigma_pop = [combined_sigma[i] for i in selected_indices]
        fitness_pop = [combined_fitness[i] for i in selected_indices]
        
        # Print progress
        if verbose and (g + 1) % 100 == 0:
            avg_sigma = np.mean([np.mean(s) for s in sigma_pop])
            extra_info = f"avg σ: {avg_sigma:.4f}"
            print_progress(g + 1, mu, fitness_pop, "ES", extra_info)

    return population, fitness_pop


# Remove duplicate functions - they are now in eva_core.py
