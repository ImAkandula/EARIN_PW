import numpy as np
import matplotlib.pyplot as plt
from genetic_algorithm import GeneticAlgorithm

# === Fixed good parameters (except crossover_rate) ===
base_params = {
    "population_size": 50,
    "mutation_rate": 0.2,
    "mutation_strength": 0.1,
    "num_generations": 100
}

# === Values to test ===
crossover_rates = [0.5, 0.7, 0.9, 1.0]
seeds = [1, 2, 3]  # Use more seeds for averaging

# === For each crossover rate, run and average the results ===
for rate in crossover_rates:
    all_best = []
    all_avg = []

    for seed in seeds:
        ga = GeneticAlgorithm(
            population_size=base_params["population_size"],
            mutation_rate=base_params["mutation_rate"],
            mutation_strength=base_params["mutation_strength"],
            crossover_rate=rate,
            num_generations=base_params["num_generations"]
        )

        _, best_fitness_values, avg_fitness_values = ga.evolve(seed=seed)
        all_best.append(best_fitness_values)
        all_avg.append(avg_fitness_values)

    # Average across seeds
    mean_best = np.mean(all_best, axis=0)
    mean_avg = np.mean(all_avg, axis=0)

    # === Plotting ===
    generations = list(range(1, base_params["num_generations"] + 1))
    plt.figure(figsize=(10, 5))
    plt.plot(generations, mean_best, label="Best Fitness")
    plt.plot(generations, mean_avg, label="Average Fitness")
    plt.yscale("log")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (log scale)")
    plt.title(f"Crossover Rate = {rate} (Averaged over {len(seeds)} seeds)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # === Save to file ===
    filename = f"crossover_impact_{int(rate * 100)}.png"
    plt.savefig(filename)
    plt.close()

    print(f"✅ Saved: {filename}")
