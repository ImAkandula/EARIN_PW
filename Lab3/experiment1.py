import itertools
from genetic_algorithm import GeneticAlgorithm

# Define parameter ranges
population_sizes = [50, 25, 10]
mutation_rates = [0.1, 0.3]
mutation_strengths = [0.05, 0.1]
crossover_rates = [0.7, 0.9]
seeds = [42, 1, 123]

# Generate all combinations of parameters
combinations = list(itertools.product(
    population_sizes,
    mutation_rates,
    mutation_strengths,
    crossover_rates,
    seeds
))

# Prepare results
results = []

for i, (pop_size, mut_rate, mut_strength, cross_rate, seed) in enumerate(combinations):
    ga = GeneticAlgorithm(
        population_size=pop_size,
        mutation_rate=mut_rate,
        mutation_strength=mut_strength,
        crossover_rate=cross_rate,
        num_generations=100
    )

    best_solutions, best_fitness_values, avg_fitness_values = ga.evolve(seed=seed)

    best_solution = best_solutions[-1]
    best_fitness = best_fitness_values[-1]

    result_line = (
        f"Test #{i+1}\n"
        f"  Population size: {pop_size}\n"
        f"  Mutation rate: {mut_rate}\n"
        f"  Mutation strength: {mut_strength}\n"
        f"  Crossover rate: {cross_rate}\n"
        f"  Seed: {seed}\n"
        f"  Best solution: ({best_solution[0]:.4f}, {best_solution[1]:.4f})\n"
        f"  Best fitness: {best_fitness:.6e}\n"
        f"{'-'*40}\n"
    )

    results.append(result_line)

# Save results to a text file
with open("experiment_results.txt", "w") as file:
    file.writelines(results)

print("✅ Experiments complete. Results saved to 'experiment_results.txt'")
