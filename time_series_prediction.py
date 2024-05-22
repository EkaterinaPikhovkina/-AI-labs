import numpy as np


learning_rate = 0.1
normalization_constant = 10


# Тренувальні дані
training_inputs = np.array([
    [0.48, 4.30, 0.91],
    [4.30, 0.91, 4.85],
    [0.91, 4.85, 0.53],
    [4.85, 0.53, 4.51],
    [0.53, 4.51, 1.95],
    [4.51, 1.95, 5.88],
    [1.95, 5.88, 0.63],
    [5.88, 0.63, 5.79],
    [0.63, 5.79, 0.92],
    [5.79, 0.92, 5.18],
])

training_inputs /= normalization_constant

training_outputs = np.array([[4.85],
                             [0.53],
                             [4.51],
                             [1.95],
                             [5.88],
                             [0.63],
                             [5.79],
                             [0.92],
                             [5.18],
                             [1.88],
                             ])

training_outputs /= normalization_constant


# Тестові дані
testing_inputs = np.array([
    [0.92, 5.18, 1.88],
    [5.18, 1.88, 4.84],
])

testing_inputs /= normalization_constant

testing_outputs = np.array([
    [4.84],
    [1.63],
])


# Функція активації - сигмоїдна
def sigmoid(x):
    scaled_x = np.clip(x, -5, 5)  # Зменшуємо значення x, щоб уникнути переповнення
    return 1 / (1 + np.exp(-scaled_x))


# Похідна функції помилки по ваговим коефіцієнтам
def derivative(x):
    der = (np.exp(-x) / (1 + np.exp(-x)) ** 2)
    return der


# Навчання нейронної мережі методом зворотнього поширення

# Ініціалізація вагових коефіцієнтів
w_1 = np.random.normal(scale=1, size=(3, 2))
w_2 = np.random.normal(scale=1, size=(2, 1))

iterations = 0

while iterations <= 100000:

    # Прямий прохід
    si_1 = np.dot(training_inputs, w_1)
    si_1_sigmoid = sigmoid(si_1)
    si_2 = np.dot(si_1_sigmoid, w_2)
    output = sigmoid(si_2)

    # Обчислення помилки
    error = np.mean((training_outputs - output) ** 2)

    # Зворотній прохід (back propagation)
    error_1 = training_outputs - output
    derivative_1 = derivative(output) * error_1
    error_2 = np.dot(derivative_1, w_2.T)
    derivative_2 = derivative(si_1_sigmoid) * error_2

    # Оновлення вагових коефіцієнтів
    w_1_update = np.dot(training_inputs.T, derivative_2)
    w_2_update = np.dot(si_1_sigmoid.T, derivative_1)
    w_1 = w_1 + learning_rate * w_1_update
    w_2 = w_2 + learning_rate * w_2_update

    if iterations == 0 or iterations == 100000:
        print(f"Iteration: {iterations}")
        for row in range(len(output)):
            print(f"True: {training_outputs[row] * normalization_constant}, "
                  f"Predicted: {output[row] * normalization_constant}, "
                  f"Error: {abs(error_1[row])}")
        print("\n")

    iterations += 1


# Прогнозування наступного значення
print("Test")
for row in range(len(testing_outputs)):
    predicted_value = sigmoid(np.dot(sigmoid(np.dot(testing_inputs[row], w_1)), w_2)) * normalization_constant
    print(f"True: {testing_outputs[row]}, "
          f"Predicted value: {predicted_value}, "
          f"Error: {abs(testing_outputs[row] - predicted_value)}")
