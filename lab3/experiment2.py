import numpy as np
from genetic_algorithm import GeneticAlgorithm

# === Best parameters from Experiment 1 ===
best_params = {
    "population_size": 50,
    "mutation_rate": 0.2,
    "mutation_strength": 0.1,
    "crossover_rate": 0.9,
    "num_generations": 100
}

# === Step 1: 5 different seeds ===
seeds = [1, 42, 100, 2024, 999]
results_seed_run = []

for seed in seeds:
    ga = GeneticAlgorithm(
        population_size=best_params["population_size"],
        mutation_rate=best_params["mutation_rate"],
        mutation_strength=best_params["mutation_strength"],
        crossover_rate=best_params["crossover_rate"],
        num_generations=best_params["num_generations"]
    )
    best_solutions, best_fitness_values, _ = ga.evolve(seed=seed)
    best_fitness = best_fitness_values[-1]
    best_solution = best_solutions[-1]
    results_seed_run.append((seed, best_solution, best_fitness))

# === Calculate stats ===
fitness_values = [r[2] for r in results_seed_run]
mean_fitness = np.mean(fitness_values)
std_fitness = np.std(fitness_values)

# === Step 2: Reduced population sizes ===
fractions = [0.5, 0.25, 0.1]
reduced_results = []

for frac in fractions:
    reduced_pop = max(2, int(best_params["population_size"] * frac))
    fitnesses = []

    for seed in seeds:
        ga = GeneticAlgorithm(
            population_size=reduced_pop,
            mutation_rate=best_params["mutation_rate"],
            mutation_strength=best_params["mutation_strength"],
            crossover_rate=best_params["crossover_rate"],
            num_generations=best_params["num_generations"]
        )
        _, best_fitness_values, _ = ga.evolve(seed=seed)
        fitnesses.append(best_fitness_values[-1])

    reduced_mean = np.mean(fitnesses)
    reduced_std = np.std(fitnesses)
    reduced_results.append((reduced_pop, reduced_mean, reduced_std))

# === Save results to TXT ===
with open("experiment_2_randomness.txt", "w") as file:
    file.write("=== Experiment 2: Randomness in Genetic Algorithm ===\n\n")

    file.write("1. Using best parameters, 5 seeds:\n")
    for seed, sol, fit in results_seed_run:
        file.write(f"  Seed {seed} -> Fitness: {fit:.6e}, Solution: ({sol[0]:.4f}, {sol[1]:.4f})\n")
    file.write(f"\nMean Fitness: {mean_fitness:.6e}\n")
    file.write(f"Std Dev Fitness: {std_fitness:.6e}\n\n")

    file.write("2. Reduced Population Sizes:\n")
    for pop, mean, std in reduced_results:
        file.write(f"  Population: {pop} -> Mean: {mean:.6e}, Std Dev: {std:.6e}\n")

print("✅ Experiment 2 complete. Results saved to 'experiment_2_randomness.txt'")
