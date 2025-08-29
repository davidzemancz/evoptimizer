from pymoo.problems import get_problem
from pymoo.util.plotting import plot
import numpy as np
import sys
import matplotlib.pyplot as plt
from differential_evolution import differential_evolution
from evolutionary_strategies import evolutionary_strategies, evolutionary_strategies_fast
from eva_core import filter_feasible_solutions

# (https://pymoo.org/problems/multi/bnh.html)
# --- py .\main.py problem_name algorithm verbose ---
# py .\main.py bnh de true         
# py .\main.py osy de true
# py .\main.py zdt1 de true
# py .\main.py dascmop1 de true
# py .\main.py bnh es true         (Evolutionary Strategies)
# py .\main.py dascmop1 es true
# py .\main.py bnh esf true        (Fast Evolutionary Strategies)
# py .\main.py dascmop1 esf true

def main(problem_name, algorithm, verbose):
    try:
        problem = get_problem(problem_name)
    except:
        problem = get_problem(problem_name, 1) # difficulty 1

    if verbose: 
        print(f"Problem: {problem_name}")
        print(f"Algorithm: {algorithm}")

    # Choose algorithm
    if algorithm.lower() == 'de':
        population, fitness_pop = differential_evolution(problem, verbose=verbose)
    elif algorithm.lower() == 'es':
        population, fitness_pop = evolutionary_strategies(problem, verbose=verbose)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Available: de, es")

    # Filter out infeasible solutions for plotting
    feasible_solutions, feasible_objectives = filter_feasible_solutions(population, fitness_pop)
    
    if verbose:
        print(f"Final: {len(feasible_objectives)} feasible solutions found")
    
    # Plot results
    if feasible_objectives:
        feasible_objectives = np.array(feasible_objectives)
        plot_solutions_with_pareto_front(problem, feasible_objectives, algorithm)
    
    return population


# Remove duplicate functions - they are now in eva_core.py

def plot_solutions_with_pareto_front(problem, feasible_objectives, algorithm_name="Algorithm"):
    """
    Plot the feasible solutions along with the Pareto front
    """
    try:
        # Get the true Pareto front
        pf = problem.pareto_front()
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        
        # Plot feasible solutions first (in front)
        if len(feasible_objectives) > 0:
            plt.scatter(feasible_objectives[:, 0], feasible_objectives[:, 1], 
                       c='blue', alpha=0.6, s=50, zorder=5, label=f'{algorithm_name.upper()} Solutions')
        
        # Plot Pareto front behind (as background)
        if pf is not None and len(pf) > 0:
            plt.plot(pf[:, 0], pf[:, 1], 'r-', linewidth=2, label='True Pareto Front')
            plt.scatter(pf[:, 0], pf[:, 1], c='red', s=30, zorder=3)
        
        plt.xlabel('Objective 1 (f1)')
        plt.ylabel('Objective 2 (f2)')
        plt.title(f'{algorithm_name.upper()} Solutions vs Pareto Front')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save and show plot
        filename = f'{algorithm_name.lower()}_solutions_pareto_front.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Plot saved as '{filename}'")
        
    except Exception as e:
        print(f"Error creating plot: {e}")
        print("Plotting not available")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python main.py <problem_name> <algorithm> [verbose]")
        print("Example: python main.py bnh de true")
        print("         python main.py dascmop1 es true")
        print("         python main.py bnh esf true")
        print("Available algorithms: de, es, esf")
        sys.exit(1)
        
    problem_name = sys.argv[1]
    algorithm = sys.argv[2]
    verbose = sys.argv[3].lower() == "true" if len(sys.argv) > 3 else False
    main(problem_name, algorithm, verbose)