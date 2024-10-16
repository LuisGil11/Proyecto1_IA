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
x_serie, y_serie = generate_series(0, 0, 1000)
x = np.column_stack((x_serie[:-1], y_serie[:-1]))
y = x_serie[1:]

# Dividimos en conjunto de entrenamiento y prueba
x_train, x_test = x[:800], x[800:]
y_train, y_test = y[:800], y[800:]

# Asignamos al modelo la data de entrenamiento
x = x_train
y = y_train


def forward_prop(row): 
    y_hat = np.dot(x[row], w) + b
    return y_hat
    if y_hat > 0:
        return 1
    else:
        return 0

def backward_prop(y_hat, row):
    global w, b
    error = y[row] - y_hat
    w[0] = w[0] + alpha * (error) * x[row][0]
    w[1] = w[1] + alpha * (error) * x[row][1]
    b = b + alpha * (error)

# Retorna la prediccion de y_hat, para el conjunto de prueba.
def predict(x):
    y_pred = []
    # y = []
    
    # El usuario podría ingresar varias filas. Calcule y_hat para cada una de las filas
    # del conjunto de datos de prueba.

    for row in x:
        # Suma ponderada
        y_hat = np.dot(row, w) + b
        y_pred.append(y_hat)
        # Paso de la suma ponderada por la función de activación
        # if y_pred > 0:
        #     y.append(1)
        # else:
        #     y.append(0)
    
    # return y
    return y_pred

# * Entrenamiento
# Número de épocas
for epoch in range(1000):

    # Para cada fila en x (ciclo a través del conjunto de datos)
    for row in range(x.shape[0]):

        # Para cada fila de x, predice y_hat
        y_hat = forward_prop(row)

        # Para cada fila se actualiza los pesos
        backward_prop(y_hat, row)

print(w, b)

# * Prueba
y_pred = predict(x_test)
# print(y_pred)

# Calculamos el error
error = np.mean((y_test - y_pred) ** 2)
print(error)