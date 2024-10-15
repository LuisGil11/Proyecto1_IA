import numpy as np
import pandas as pd

def generate_series(x0 = 0, y0 = 0, n = 1000):
    x = np.zeros(n)
    y = np.zeros(n)
    x[0], y[0] = x0, y0
    for i in range(1, n):
        x[i] = 1 - a * x[i - 1] ** 2 + y[i - 1]
        y[i] = b * x[i - 1]
    return x, y

# Inicializamos los parámetros
a = 1.4
b = 0.3
w = np.random.rand(2)
alpha = 0.001

# Datos de entrenamiento
x, y = generate_series(0,0, 1000)
X = np.column_stack((x[:-1], y[:-1]))
Y = x[1:]

# Dividir los datos en conjuntos de entrenamiento y prueba
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
Y_train, Y_test = Y[:split_index], Y[split_index:]

def forward_prop(features):
    y_hat = np.dot(features, w) + b
    if y_hat > 0:
        return 1
    else:
        return 0

def backward_prop(y_hat, row):
    global w, b
    error = Y_train[row] - y_hat
    w[0] = w[0] + alpha * (error) * x[row]
    w[1] = w[1] + alpha * (error) * x[row]
    b = b + alpha * (error)


# Retorna la predicción de y_hat, para el conjunto de prueba.
def predict(X):
    # El usuario podr´ıa ingresar varias filas.
    # Calcule y_hat para cada una de las filas del conjunto de
    # datos de prueba:

    y_pred = []
    for row in X:
        #Suma ponderada
        y_hat = np.dot(row, w) + b
        y_pred.append(y_hat)
    # Devuelve la matriz de valores y_hat predichos para los datos de prueba
    # correspondientes (x)
    return y_pred

# * Entrenamiento

# Número de épocas
for epoch in range(1000):

    # Para cada fila en x (ciclo a través del conjunto de datos)
    for row in range(X_train.shape[0]):

        # Para cada fila en x, predice y_hat
        features = X_train[row]
        y_hat = forward_prop(features)

        # Para cada fila en x, actualiza los pesos y el sesgo
        backward_prop(y_hat, row)
print(w, b)

# * Prueba

# Predecimos los valores de y_hat para el conjunto de datos de prueba
y_pred = predict(X_test)

# Calcular el error
error = Y_test - y_pred

# Imprimimos una tabla comparativa con los valores reales y los predichos
df = pd.DataFrame({'y': Y_test, 'y_pred': y_pred, 'error': error})

print(df)