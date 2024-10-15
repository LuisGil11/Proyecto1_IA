import numpy as np

def generate_series(x0 = 0, y0 = 0, n = 1000):
    x = np.zeros(n)
    y = np.zeros(n)
    x[0] = x0
    y[0] = y0
    for i in range(1, n):
        x[i] = 1 - a * x[i - 1] ** 2 + y[i - 1]
        y[i] = b * y[i - 1]
    return x, y

# Inicializamos los parámetros
a = 1.4
b = 0.3
w = np.random.rand(2)
alpha = 0.001

# Generamos las series
x, y = generate_series()

def forward_prop(row):
    y_hat = np.dot([x[row], y[row]], w) + b
    if y_hat > 0:
        return 1
    else:
        return 0

def backward_prop(y_hat, row):
    global w, b
    w[0] = w[0] + alpha * (y[row] - y_hat) * x[row]
    w[1] = w[1] + alpha * (y[row] - y_hat) * x[row]
    b = b + alpha * (y[row] - y_hat)


# Retorna la predicción de y_hat, para el conjunto de prueba.
def predict(x):
    y = []

    # El usuario podr´ıa ingresar varias filas.
    # Calcule y_hat para cada una de las filas del conjunto de
    # datos de prueba:

    #Suma ponderada
    y_pred = np.dot(row, w) + b

    # Paso de la suma ponderada por la función de activación
    if (y_pred > 0):
        y_pred = 1
    else:
        y_pred = 0
    
    y.append(y_pred)

    # Devuelve la matriz de valores y_hat predichos para los datos de prueba
    # correspondientes (x)
    return y

# * Entrenamiento

# Número de épocas
for epoch in range(1000):

    # Para cada fila en x (ciclo a través del conjunto de datos)
    for row in range(x.shape[0]):

        # Para cada fila en x, predice y_hat
        y_hat = forward_prop(row)

        # Para cada fila en x, actualiza los pesos y el sesgo
        backward_prop(y_hat, row)
print(w, b)