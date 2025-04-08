import numpy as np
import matplotlib.pyplot as plt
from genetic_algorithm import GeneticAlgorithm

# === Base GA parameters ===
base_params = {
    "population_size": 50,
    "crossover_rate": 0.9,
    "num_generations": 100
}

# === Mutation settings to test ===
mutation_rates = [0.1, 0.3, 0.5]
mutation_strengths = [0.05, 0.1, 0.2]
seeds = [1, 2, 3]  # Multiple seeds for averaging

# === Run GA for each combination of mutation settings ===
for m_rate in mutation_rates:
    for m_strength in mutation_strengths:
        all_best = []
        all_avg = []

        for seed in seeds:
            ga = GeneticAlgorithm(
                population_size=base_params["population_size"],
                mutation_rate=m_rate,
                mutation_strength=m_strength,
                crossover_rate=base_params["crossover_rate"],
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
        plt.title(f"Mutation Rate = {m_rate}, Strength = {m_strength}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # === Save plot ===
        filename = f"mutation_effect_r{int(m_rate*100)}_s{int(m_strength*100)}.png"
        plt.savefig(filename)
        plt.close()

        print(f"✅ Saved plot: {filename}")
