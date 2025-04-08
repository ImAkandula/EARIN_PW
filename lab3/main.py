import matplotlib.pyplot as plt
from genetic_algorithm import GeneticAlgorithm

if __name__ == "__main__":
    # --- GA Configuration ---
    ga = GeneticAlgorithm(
        population_size=50,
        mutation_rate=0.2,
        mutation_strength=0.1,
        crossover_rate=0.9,
        num_generations=100,
    )

    # --- Run the GA with a fixed seed ---
    best_solutions, best_fitness_values, average_fitness_values = ga.evolve(seed=42)

    # --- Output best result ---
    print("Best solution found:", best_solutions[-1])
    print("Best fitness value:", best_fitness_values[-1])

    # --- Plot results ---
    generations = list(range(1, len(best_fitness_values) + 1))
    plt.figure(figsize=(10, 5))

    plt.plot(generations, best_fitness_values, label="Best Fitness")
    plt.plot(generations, average_fitness_values, label="Average Fitness")
    plt.yscale("log")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (log scale)")
    plt.title("Booth Function Optimization via Genetic Algorithm")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
