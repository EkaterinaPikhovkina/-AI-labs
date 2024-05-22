# train.py


import numpy as np


# Параметри навчання
learning_rate = 0.1
normalization_constant = 10
noise_scale = 0.01  # Розмір шуму


# Функція активації - сигмоїдна
def sigmoid(x):
    scaled_x = np.clip(x, -5, 5)  # Зменшуємо значення x, щоб уникнути переповнення
    return 1 / (1 + np.exp(-scaled_x))


# Похідна функції помилки по ваговим коефіцієнтам
def derivative(x):
    der = (np.exp(-x) / (1 + np.exp(-x)) ** 2)
    return der


# Навчання нейронної мережі методом зворотнього поширення
def train_neural_network(X_train, y_train):
    # Ініціалізація вагових коефіцієнтів нулями
    w_1 = np.zeros((36, 36)) + np.random.normal(scale=noise_scale, size=(36, 36))
    w_2 = np.zeros((36, 2)) + np.random.normal(scale=noise_scale, size=(36, 2))

    iterations = 0
    error = 1

    while error > 0.2:

        # Прямий прохід
        si_1 = np.dot(X_train, w_1)
        si_1_sigmoid = sigmoid(si_1)
        si_2 = np.dot(si_1_sigmoid, w_2)
        output = sigmoid(si_2)

        # Обчислення помилки
        error = np.mean((y_train - output) ** 2)

        # Зворотній прохід (back propagation)
        error_1 = y_train - output
        derivative_1 = derivative(output) * error_1
        error_2 = np.dot(derivative_1, w_2.T)
        derivative_2 = derivative(si_1_sigmoid) * error_2

        # Оновлення вагових коефіцієнтів
        w_1_update = np.dot(X_train.T, derivative_2)
        w_2_update = np.dot(si_1_sigmoid.T, derivative_1)
        w_1 += learning_rate * w_1_update
        w_2 += learning_rate * w_2_update

        if iterations % 100 == 0:
            print(f"Iteration: {iterations}")
            for row in range(len(output)):
                print(f"True: {y_train[row]}, Predicted: {np.round(output[row]).astype(int)}")

        iterations += 1

    # Збереження навчених ваг
    np.savez('trained_weights.npz', w_1=w_1, w_2=w_2)


# Training data
X_train = np.array([
    [1, 1, 0, 0, 1, 1,
     0, 1, 0, 0, 1, 1,
     0, 1, 1, 1, 1, 1,
     0, 0, 1, 1, 1, 0,
     0, 1, 1, 1, 0, 0,
     1, 1, 1, 0, 0, 0],

    [1, 0, 0, 0, 1, 1,
     1, 1, 0, 1, 1, 0,
     0, 1, 1, 1, 0, 0,
     0, 0, 1, 1, 0, 0,
     0, 1, 0, 1, 1, 0,
     1, 1, 0, 0, 1, 1],

    [1, 1, 1, 1, 1, 1,
     1, 0, 0, 0, 1, 1,
     0, 0, 0, 1, 1, 0,
     0, 0, 1, 1, 0, 0,
     0, 1, 1, 0, 0, 1,
     1, 1, 1, 1, 1, 1],

    [1, 1, 0, 0, 1, 1,
     1, 1, 0, 0, 1, 1,
     1, 1, 0, 0, 1, 1,
     1, 1, 0, 0, 1, 1,
     1, 1, 1, 1, 1, 1,
     0, 1, 1, 1, 1, 0]
])

# Labels for training
y_train = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])


# Виклик функції навчання
train_neural_network(X_train, y_train)
