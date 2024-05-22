# test.py


import numpy as np


# Функція активації - сигмоїдна
def sigmoid(x):
    scaled_x = np.clip(x, -5, 5)  # Зменшуємо значення x, щоб уникнути переповнення
    return 1 / (1 + np.exp(-scaled_x))


# Завантаження навчених ваг
loaded_weights = np.load('trained_weights.npz')
w_1 = loaded_weights['w_1']
w_2 = loaded_weights['w_2']


# Testing data
X_test = np.array([
    [0, 1, 0, 0, 1, 1,  # Змінено символ [0][0]
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

    [1, 0, 0, 0, 1, 1,  # Змінено символ [3][1]
     1, 1, 0, 0, 1, 1,
     1, 1, 0, 0, 1, 1,
     1, 1, 0, 0, 1, 1,
     1, 1, 1, 1, 1, 1,
     0, 1, 1, 1, 1, 0]
])


# Виведення очікуваних результатів після тестування
print("Test")
for i in range(len(X_test)):
    predicted_value = np.round(sigmoid(np.dot(sigmoid(np.dot(X_test[i], w_1)), w_2))).astype(int)
    print(f"Predicted: {predicted_value}")
