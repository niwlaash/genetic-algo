import numpy as np
import matplotlib.pyplot as plt

# List of product names
products = ['milk', 'sugar', 'coffee', 'chocolate', 'rice', 'salt', 'pepper', 'paprika', 'garlic_powder',
            'onion_powder']


# Function to initialize the population with product names
def initialize_population(population_size, inventory_size):
    return np.random.randint(0, 100, size=(population_size, inventory_size))


# Function to calculate fitness based on holding and stockout costs
def calculate_fitness(population, demand, holding_cost, stockout_cost):
    holding_costs = np.sum(population * holding_cost, axis=1)
    stockout_costs = np.maximum(0, demand - population.sum(axis=1)) * stockout_cost
    total_costs = holding_costs + stockout_costs
    return total_costs


# Function for tournament selection
def select_parents(population, fitness, tournament_size):
    selected_parents = []
    for _ in range(len(population)):
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = fitness[tournament_indices]
        selected_parents.append(population[tournament_indices[np.argmin(tournament_fitness)]])
    return np.array(selected_parents)


# Function for one-point crossover
def crossover(parents, crossover_rate):
    children = parents.copy()
    for i in range(0, len(parents), 2):
        if np.random.rand() < crossover_rate:
            crossover_point = np.random.randint(1, len(parents[i]))
            children[i, crossover_point:] = parents[i + 1, crossover_point:]
            children[i + 1, crossover_point:] = parents[i, crossover_point:]
    return children


# Function for mutation
def mutate(children, mutation_rate):
    mutation_mask = np.random.rand(*children.shape) < mutation_rate
    children[mutation_mask] = np.random.randint(0, 100, size=np.sum(mutation_mask))
    return children


# Function to evolve the population
def evolve(population, fitness, crossover_rate, mutation_rate, tournament_size):
    parents = select_parents(population, fitness, tournament_size)
    children = crossover(parents, crossover_rate)
    mutated_children = mutate(children, mutation_rate)
    return mutated_children


# Function to visualize the optimization process
def visualize_progress(costs):
    generations = len(costs)
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, generations + 1), costs, marker='o' )
    plt.title('Genetic Algorithm Progress')
    plt.xlabel('Generation')
    plt.ylabel('Cost')
    plt.show()


# Main genetic algorithm function with visualization
def genetic_algorithm_with_visualization(population_size, inventory_size, generations, crossover_rate, mutation_rate,
                                         tournament_size):
    # Real-world parameters (e.g., demand, holding cost, stockout cost)
    demand = 50  # Average demand for the product
    holding_cost = 0.5  # Holding cost per unit per time period
    stockout_cost = 5  # Cost per unit for stockout

    # Initialize population with product names
    population = initialize_population(population_size, inventory_size)
    best_costs = []

    for generation in range(generations):
        fitness = calculate_fitness(population, demand, holding_cost, stockout_cost)
        population = evolve(population, fitness, crossover_rate, mutation_rate, tournament_size)

        # Track the best solution in each generation
        best_solution = population[np.argmin(fitness)]
        best_cost = np.min(fitness)
        best_costs.append(best_cost)

        print(f"Generation {generation + 1}, Best Solution: {best_solution}, Cost: {best_cost}")

    # Visualize the optimization progress
    visualize_progress(best_costs)

    # Return the best solution and its cost
    best_index = np.argmin(fitness)
    best_solution_with_names = list(zip(products, best_solution))
    return best_solution_with_names, fitness[best_index]


# GA parameters
population_size = 50
inventory_size = 10
generations = 100
crossover_rate = 0.8
mutation_rate = 0.03
tournament_size = 5

best_solution, best_cost = genetic_algorithm_with_visualization(
    population_size, inventory_size, generations, crossover_rate, mutation_rate, tournament_size
)

print(f"\nFinal Best Solution: {best_solution}, Final Cost: {best_cost}")
