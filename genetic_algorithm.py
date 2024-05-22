import numpy as np
import matplotlib.pyplot as plt


# Функція пристосованості (значення функції Y(x))
def fitness_function(x):
    return (x ** 3) + np.cos(15 * x)


# Конвертація з двійкової форми до десяткової
def binary_to_decimal(binary):
    return -2 + (2 - (-2)) * int(binary, 2) / (2 ** len(binary) - 1)


# Створення хромосоми (генерація випадкового значення x у двійковій формі)
def generate_chromosome():
    binary_length = 20
    return ''.join(np.random.choice(['0', '1'], size=binary_length))


# Селекція (ранжирування)
def selection(population, fitness_values):
    # Ранжування особин за їхньою пристосованістю
    ranked_indices = np.argsort(fitness_values)
    ranked_population = population[ranked_indices]
    # Вибір пар за рангом
    num_pairs = len(population) // 2
    selected_pairs = []
    for i in range(num_pairs):
        selected_pairs.append((ranked_population[i], ranked_population[-(i+1)]))
    return selected_pairs


# Рекомбінація (кросовер)
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1))
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


# Мутація (зміна випадкового біта хромосоми)
def mutation(chromosome):
    mutation_point = np.random.randint(0, len(chromosome))
    mutated_chromosome = list(chromosome)
    mutated_chromosome[mutation_point] = '0' if mutated_chromosome[mutation_point] == '1' else '1'
    return ''.join(mutated_chromosome)


def genetic_algorithm(population_size, generations):
    population = np.array([generate_chromosome() for _ in range(population_size)])

    minimum = float('inf')
    maximum = float('-inf')
    min_chromosome = None
    max_chromosome = None

    for _ in range(generations):
        fitness_values = [fitness_function(binary_to_decimal(chromosome)) for chromosome in population]

        current_min = np.min(fitness_values)
        current_max = np.max(fitness_values)
        if current_min < minimum:
            minimum = current_min
            min_chromosome = population[np.argmin(fitness_values)]
        if current_max > maximum:
            maximum = current_max
            max_chromosome = population[np.argmax(fitness_values)]

        # Селекція найкращих особин
        selected_pairs = selection(population, fitness_values)

        # Рекомбінація (кросовер)
        children = []
        for parent1, parent2 in selected_pairs:
            child1, child2 = crossover(parent1, parent2)

            # Мутація
            child1 = mutation(child1)
            child2 = mutation(child2)
            children.extend([child1, child2])

        # Оновлення популяції
        population = np.array(children)

    return minimum, maximum, min_chromosome, max_chromosome


minimum, maximum, min_chromosome, max_chromosome = genetic_algorithm(population_size=4, generations=10000)

print("Minimum value:", minimum)
print("Maximum value:", maximum)
print("Chromosome for minimum value:", min_chromosome)
print("Chromosome for maximum value:", max_chromosome)
print("Chromosome for minimum value (decimal):", binary_to_decimal(min_chromosome))
print("Chromosome for maximum value (decimal):", binary_to_decimal(max_chromosome))

x_values = np.linspace(-2, 2, 1000)
y_values = fitness_function(x_values)
plt.plot(x_values, y_values, label='Y(x)')
plt.scatter(binary_to_decimal(min_chromosome), minimum, color='green', label='Minimum Value')
plt.scatter(binary_to_decimal(max_chromosome), maximum, color='red', label='Maximum Value')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Minimum and Maximum')
plt.legend()
plt.grid(True)
plt.show()
