from pymoo.problems import get_problem
from pymoo.util.plotting import plot
import numpy as np
import sys
import matplotlib.pyplot as plt

# py .\main.py problem_name verbose
# py .\main.py bnh true         (https://pymoo.org/problems/multi/bnh.html)
# py .\main.py osy true

def main(problem_name, verbose):
    problem = get_problem(problem_name, 1)

    if verbose: print(f"Problem: {problem_name}")

    POP_SIZE = 100
    GENS = 1_000

    # Vytvorim populaci
    population = []
    for n in range(POP_SIZE):
        population.append(np.random.uniform(problem.xl, problem.xu, problem.n_var))

    # Parametry DE
    F = 0.8 
    CR = 0.9
    
    # Zjistim fitness
    fitness_pop = []
    for i in range(POP_SIZE):
        fit = evaluate(problem, population[i])
        fitness_pop.append(fit)
    
    # Evoluce
    for g in range(GENS):
        new_population = []
        new_fitness = []
        
        for i in range(POP_SIZE):
            # Vezmu jednice a modifikuji
            target = population[i]
            
            # Vyberu tri nahodne jedince
            candidates = list(range(POP_SIZE))
            candidates.remove(i)
            a, b, c = np.random.choice(candidates, 3, replace=False)
            
            # Mutace
            mutant = population[a] + F * (population[b] - population[c])
            mutant = np.clip(mutant, problem.xl, problem.xu)
            
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
        if verbose and (g + 1) % 100 == 0:
            feasible_count = sum(1 for f in fitness_pop if f is not None)
            print(f"Generation {g + 1}: {feasible_count}/{POP_SIZE} feasible solutions")
    
    # Filter out infeasible solutions for plotting
    feasible_solutions = []
    feasible_objectives = []
    for i, fit in enumerate(fitness_pop):
        if fit is not None:
            feasible_solutions.append(population[i])
            feasible_objectives.append(fit)
    
    if verbose:
        print(f"Final: {len(feasible_objectives)} feasible solutions found")
    
    # Plot results
    if feasible_objectives:
        feasible_objectives = np.array(feasible_objectives)
        plot_solutions_with_pareto_front(problem, feasible_objectives)
    
    return population

def evaluate(problem, x):
    """
    Vyhodnoti fitness funkci (vzdy minimalizuji)
    Pokud neni feasible, vrati None
    """

    # Tohle nekontroluji a hlidam, ze se to nestane   
    # Promenne jsou v bounds
    # if np.any(x < problem.xl) or np.any(x > problem.xu):
    #     return None

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

def plot_solutions_with_pareto_front(problem, feasible_objectives):
    """
    Plot the feasible solutions along with the Pareto front
    """
    try:
        # Get the true Pareto front
        pf = problem.pareto_front()
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        
        # Plot feasible solutions
        if len(feasible_objectives) > 0:
            plt.scatter(feasible_objectives[:, 0], feasible_objectives[:, 1], 
                       c='blue', alpha=0.6, s=50, label='DE Solutions')
        
        # Plot Pareto front if available
        if pf is not None and len(pf) > 0:
            plt.plot(pf[:, 0], pf[:, 1], 'r-', linewidth=2, label='True Pareto Front')
            plt.scatter(pf[:, 0], pf[:, 1], c='red', s=30, zorder=5)
        
        plt.xlabel('Objective 1 (f1)')
        plt.ylabel('Objective 2 (f2)')
        plt.title(f'Differential Evolution Solutions vs Pareto Front')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save and show plot
        plt.savefig('de_solutions_pareto_front.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Plot saved as 'de_solutions_pareto_front.png'")
        
    except Exception as e:
        print(f"Error creating plot: {e}")
        print("Plotting not available")

if __name__ == "__main__":
    problem_name = sys.argv[1]
    verbose = sys.argv[2].lower() == "true" if len(sys.argv) > 2 else False
    main(problem_name, verbose)