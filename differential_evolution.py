import numpy as np
from eva_core import evaluate, dominates, print_progress, filter_feasible_solutions, initialize_population, clip_to_bounds


def differential_evolution(problem, pop_size=100, generations=500, F=0.8, CR=0.9, verbose=False):
    """
    Simple Differential Evolution algorithm
    
    Args:
        problem: Problem instance from pymoo
        pop_size: Population size
        generations: Number of generations
        F: Scaling factor for mutation
        CR: Crossover rate
        verbose: Print progress
        
    Returns:
        population: Final population
        fitness_pop: Final fitness values
    """
    
    # Initialize population
    population, fitness_pop = initialize_population(problem, pop_size)
    
    # Evoluce
    for g in range(generations):
        new_population = []
        new_fitness = []
        
        for i in range(pop_size):
            # Vezmu jednice a modifikuji
            target = population[i]
            
            # Vyberu tri nahodne jedince
            candidates = list(range(pop_size))
            candidates.remove(i)
            a, b, c = np.random.choice(candidates, 3, replace=False)
            
            # Mutation: V = Xa + F * (Xb - Xc)
            mutant = population[a] + F * (population[b] - population[c])
            
            # Ensure bounds
            mutant = clip_to_bounds(mutant, problem)
            
            # Krizeni
            trial = target.copy()
            cross_points = np.random.rand(problem.n_var) < CR
            
            # Alespon jedna nova hodnota
            if not np.any(cross_points):
                cross_points[np.random.randint(problem.n_var)] = True

            # Aplikace krizeni
            trial[cross_points] = mutant[cross_points]
            
            # Necham si lepsiho
            trial_fitness = evaluate(problem, trial)
            target_fitness = fitness_pop[i]
            
            # Kontrola infeasible reseni
            if trial_fitness is None and target_fitness is None: # Oba infeasible, beru 50/50
                if np.random.rand() < 0.5:
                    new_population.append(trial)
                    new_fitness.append(trial_fitness)
                else:
                    new_population.append(target)
                    new_fitness.append(target_fitness)
            elif trial_fitness is None: # Novy infeasible, beru target
                new_population.append(target)
                new_fitness.append(target_fitness)
            elif target_fitness is None: # Target infeasible, beru trial
                new_population.append(trial)
                new_fitness.append(trial_fitness)
            else:
                # Pokud jsou oba feasible, beru lepsiho
                if dominates(trial_fitness, target_fitness):
                    new_population.append(trial)
                    new_fitness.append(trial_fitness)
                else:
                    new_population.append(target)
                    new_fitness.append(target_fitness)
        
        # Nova populace
        population = new_population
        fitness_pop = new_fitness
        
        # Printim 
        if verbose and (g + 1) % 50 == 0:
            # unique solutions
            unique_count = len(set(tuple(sol) for sol in population))
            extra_info = f"{unique_count}/{pop_size} unique solutions"
            print_progress(g + 1, pop_size, fitness_pop, "DE", extra_info)

    return population, fitness_pop


# Remove duplicate functions - they are now in eva_core.py
