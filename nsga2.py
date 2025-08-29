import numpy as np
from eva_core import evaluate, print_progress, filter_feasible_solutions, clip_to_bounds


def nsga2(problem, pop_size=100, generations=300, verbose=False):
    """
    Non-dominated Sorting Genetic Algorithm II
    
    Args:
        problem: Problem instance from pymoo
        pop_size: Population size
        generations: Number of generations
        verbose: Print progress
        
    Returns:
        population: Final population
        fitness_pop: Final fitness values
    """
    
    # Vytovrim populaci a spocitam fitness
    population = np.random.uniform(problem.xl, problem.xu, (pop_size, problem.n_var))
    fitness_pop = [evaluate(problem, ind) for ind in population]
    
    # Parametry nsga
    crossover_prob = 0.8
    mutation_prob = 1.0 / problem.n_var
    
    # Evoluce
    for g in range(generations):
        # Generate offspring through crossover and mutation
        offspring = []
        offspring_fitness = []
        
        for _ in range(pop_size // 2):
            # Vyberu roduce
            parent1_idx = tournament_selection(population, fitness_pop)
            parent2_idx = tournament_selection(population, fitness_pop)
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Krizeni
            if np.random.rand() < crossover_prob:
                child1, child2 = crossover(parent1, parent2, problem)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutace
            child1 = mutate(child1, problem, mutation_prob)
            child2 = mutate(child2, problem, mutation_prob)
            
            # Pridam decka do populace
            fit1 = evaluate(problem, child1)
            fit2 = evaluate(problem, child2)
            offspring.extend([child1, child2])
            offspring_fitness.extend([fit1, fit2])
        
        # Sloucim populace a udelam selekci
        combined_pop = np.vstack([population, np.array(offspring)])
        combined_fitness = fitness_pop + offspring_fitness
        selected_indices = selection(combined_fitness, pop_size)
        
        population = combined_pop[selected_indices]
        fitness_pop = [combined_fitness[i] for i in selected_indices]
        
        # Progres
        if verbose and (g + 1) % 50 == 0:
            fronts = nondominated_sort([f for f in fitness_pop if f is not None])
            front_sizes = [len(front) for front in fronts]
            extra_info = f"Fronts: {front_sizes[:3]}"
            print_progress(g + 1, pop_size, fitness_pop, "NSGA2", extra_info)
    
    return population, fitness_pop


def selection(fitness_pop, pop_size):
    """
    NSGA-II environmental selection using non-dominated sorting and crowding dist
    """
    # Rozdelim na feasible a infeasible
    feasible_indices = [i for i, f in enumerate(fitness_pop) if f is not None]
    infeasible_indices = [i for i, f in enumerate(fitness_pop) if f is None]
    
    selected = []
   
    if len(feasible_indices) >= pop_size:  # Pokud mam dostatek feasible
        feasible_fitness = [fitness_pop[i] for i in feasible_indices]
        
        # Setridim
        fronts = nondominated_sort(feasible_fitness)
        
        # Vytvorim fronty
        for front in fronts:
            if len(selected) + len(front) <= pop_size:
                # Pokud se vejde, beru vse
                selected.extend([feasible_indices[i] for i in front])
            else:
                # Jinak pouziji crowding distance
                remaining = pop_size - len(selected)
                if remaining > 0:
                    front_fitness = [feasible_fitness[i] for i in front]
                    
                    distances = crowding_distance(front_fitness)
                    
                    # Setrdim sestipne a beru nejvzdalenejsi
                    sorted_front = sorted(zip(front, distances), key=lambda x: x[1], reverse=True)
                    for i in range(min(remaining, len(sorted_front))):
                        selected.append(feasible_indices[sorted_front[i][0]])
                break
    else: # Nemam dostatek feasible
        selected.extend(feasible_indices)
        remaining = pop_size - len(feasible_indices)
        
        if remaining > 0 and infeasible_indices: # Pridam nahodne infeasible
            random_infeasible = np.random.choice(infeasible_indices, 
                                               min(remaining, len(infeasible_indices)), 
                                               replace=False)
            selected.extend(random_infeasible)

    # Pokud je vybrano prilis malo jedincu, doplnim nahodnymi
    while len(selected) < pop_size:
        if selected:
            selected.append(np.random.choice(selected))
        else:
            selected.append(0)
    
    return selected[:pop_size]


def nondominated_sort(fitness_pop):
    n = len(fitness_pop)
    
    domination_count = [0] * n 
    dominated_solutions = [[] for i in range(n)]
    fronts = [[]]
    
    # Spocitam dominantni reseni
    for i in range(n):
        for j in range(n):
            if i != j:
                if dominates(fitness_pop[i], fitness_pop[j]):
                    dominated_solutions[i].append(j)
                elif dominates(fitness_pop[j], fitness_pop[i]):
                    domination_count[i] += 1
        
        # Pokud neni dominovano, pridam do prvni fronty
        if domination_count[i] == 0:
            fronts[0].append(i)

    # Vytvorim dalsi fronty
    current_front = 0
    while current_front < len(fronts) and len(fronts[current_front]) > 0:
        next_front = []
        
        for i in fronts[current_front]:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        
        if next_front:
            fronts.append(next_front)
        current_front += 1
    
    # Vratim neprazdne fronty (kdyby nahodou)
    return [front for front in fronts if len(front) > 0]


def crowding_distance(fitness_pop):
    """
    Calculate crowding distance for diversity preservation
    """
    n = len(fitness_pop)
    if n <= 2: # Pokud malo jedincu, vratim inf
        return [float('inf')] * n 
    
    distances = [0.0] * n
    n_objectives = len(fitness_pop[0])
    
    # Pro kazdou objective urcim vydalenosti
    for obj in range(n_objectives):

        # Setridim podle aktualni objective
        sorted_indices = sorted(range(n), key=lambda i: fitness_pop[i][obj])
        
        # Boudny
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')
        
        # Spocitam vzdalenost od stredu
        obj_min = fitness_pop[sorted_indices[0]][obj]
        obj_max = fitness_pop[sorted_indices[-1]][obj]
        obj_range = obj_max - obj_min
        
        # Pokud mam range, tj. neni 0, tak nastavim vzdalenost
        if obj_range > 1e-10:
            for i in range(1, n - 1):
                distances[sorted_indices[i]] += (
                    fitness_pop[sorted_indices[i + 1]][obj] - 
                    fitness_pop[sorted_indices[i - 1]][obj]
                ) / obj_range
    
    return distances


def dominates(f1, f2):
    """
    Check if f1 dominates f2 (Pareto dominance)
    """
    if f1 is None or f2 is None:
        return False
    
    better_or_equal = all(f1[i] <= f2[i] for i in range(len(f1)))
    strictly_better = any(f1[i] < f2[i] for i in range(len(f1)))
    
    return better_or_equal and strictly_better


def tournament_selection(population, fitness_pop, tournament_size=2):
   
    candidates = np.random.choice(len(population), tournament_size, replace=False)
    
    # Feasible reseni
    feasible_candidates = [i for i in candidates if fitness_pop[i] is not None]
    
    if len(feasible_candidates) >= 2:
        # Beru lepsiho (dominantniho)
        best = feasible_candidates[0]
        for candidate in feasible_candidates[1:]:
            if dominates(fitness_pop[candidate], fitness_pop[best]):
                best = candidate
        return best
    elif feasible_candidates:
        return feasible_candidates[0]
    
    # Pokud nemam feasible, vracim nahodne
    return np.random.choice(candidates)


def crossover(parent1, parent2, problem, crossover_rate=0.5):
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    for i in range(len(parent1)):
        if np.random.rand() < crossover_rate:
            child1[i], child2[i] = child2[i], child1[i]
    
    # Kontroluji boudny
    child1 = clip_to_bounds(child1, problem)
    child2 = clip_to_bounds(child2, problem)
    
    return child1, child2


def mutate(individual, problem, mutation_prob):
    mutated = individual.copy()
    
    for i in range(len(individual)):
        if np.random.rand() < mutation_prob:
            sigma = 0.1 * (problem.xu[i] - problem.xl[i])
            mutated[i] += np.random.normal(0, sigma) # Gaussovska mutace
    
    return clip_to_bounds(mutated, problem)
