import random
import numpy as np
from functions import booth_2d, init_ranges


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


class GeneticAlgorithm:
    def __init__(
        self,
        population_size: int,
        mutation_rate: float,
        mutation_strength: float,
        crossover_rate: float,
        num_generations: int,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.num_generations = num_generations

        self.function = booth_2d
        self.init_x_range, self.init_y_range = init_ranges[self.function]

    def initialize_population(self):
        x_vals = np.random.uniform(
            self.init_x_range[0], self.init_x_range[1], self.population_size
        )
        y_vals = np.random.uniform(
            self.init_y_range[0], self.init_y_range[1], self.population_size
        )
        return np.column_stack((x_vals, y_vals))

    def evaluate_population(self, population):
        return np.array([self.function(ind[0], ind[1]) for ind in population])

    def selection(self, population, fitness_values):
        # Lower fitness = better, so invert for probability
        fitness_scores = 1 / (1 + fitness_values)
        probabilities = fitness_scores / np.sum(fitness_scores)
        selected_indices = np.random.choice(
            np.arange(len(population)),
            size=len(population),
            replace=True,
            p=probabilities
        )
        return population[selected_indices]

    def crossover(self, parents):
        np.random.shuffle(parents)
        offspring = []

        for i in range(0, len(parents), 2):
            p1 = parents[i]
            if i + 1 >= len(parents):
                offspring.append(p1)
                break

            p2 = parents[i + 1]

            if np.random.rand() < self.crossover_rate:
                alpha = np.random.rand()
                child1 = alpha * p1 + (1 - alpha) * p2
                child2 = alpha * p2 + (1 - alpha) * p1
                offspring.extend([child1, child2])
            else:
                offspring.extend([p1, p2])

        return np.array(offspring)

    def mutate(self, individuals):
        for i in range(len(individuals)):
            if np.random.rand() < self.mutation_rate:
                mutation = np.random.normal(0, self.mutation_strength, 2)
                individuals[i] += mutation
                # Keep within bounds
                individuals[i][0] = np.clip(individuals[i][0], *self.init_x_range)
                individuals[i][1] = np.clip(individuals[i][1], *self.init_y_range)
        return individuals

    def evolve(self, seed: int):
        set_seed(seed)

        population = self.initialize_population()
        best_solutions = []
        best_fitness_values = []
        average_fitness_values = []

        for generation in range(self.num_generations):
            fitness = self.evaluate_population(population)

            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]
            best_fitness = fitness[best_idx]
            avg_fitness = np.mean(fitness)

            best_solutions.append(best_solution)
            best_fitness_values.append(best_fitness)
            average_fitness_values.append(avg_fitness)

            selected = self.selection(population, fitness)
            offspring = self.crossover(selected)
            population = self.mutate(offspring)

        return best_solutions, best_fitness_values, average_fitness_values
