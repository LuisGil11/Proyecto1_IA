import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# def forward_prop(row):
#     return np.dot(row, w) + b

def forward_prop(x):
    z1 = np.dot(x, W1) + b1
    a1 = sigmoide(z1)
    z2 = np.dot(a1, W2) + b2
    z2 = z2.item()
    a2 = sigmoide(z2)
    return a1, a2, z1, z2

def sigmoide(x):
  return 1 / (1 + np.exp(-x))
  
def dsigmoide(x):
  return sigmoide(x) * (1 - sigmoide(x))

def generate_series(x0 = 0, y0 = 0, n = 1000):
    x = np.zeros(n)
    y = np.zeros(n)
    x[0] = x0
    y[0] = y0
    for i in range(1, n):
        x[i] = 1 - a * x[i - 1] ** 2 + y[i - 1]
        y[i] = b * x[i - 1]
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

input_size = 2
hidden_size = 2
output_size = 1

# Inicializamos los parámetros
alpha = 0.01 # Coeficiente de aprendizaje

# Pesos para las neuronas de la capa escondida y la salida
W1 = np.random.rand(input_size, hidden_size)
W2 = np.random.rand(hidden_size)

# Sesgos para las neuronas de la capa escondida y la salida
b1 = np.random.rand(hidden_size)
b2 = np.random.rand(output_size)

# Se toma row = 0, para partir del primer valor de la serie
# w = W1 # Definimos w como W1 para que se pueda usar en forward_prop
# a1 = forward_prop(x[0])
# print(a1) # a1 almacena ambas salidas de las neuronas de la capa escondida
# w = W2 # Definimos w como W2 para que se pueda usar en forward_prop
# a2 = forward_prop(a1)
# print(a2) # a2 almacena la salida de la red neuronal

def backward_prop(y_hat, row, x, y, a1, z1):
  global b1, b2, W1, W2
  error = y[row] - y_hat

  # Construimos un vector para las derivadas parciales 
  # conrespecto a los sesgos de la capa escondida
  db1 = np.array([
      -error * y_hat * (1 - y_hat) * W2[0] * dsigmoide(z1[0]),
      -error * y_hat * (1 - y_hat) * W2[1] * dsigmoide(z1[1])
  ])

  # Construimos un vector para la derivada parcial
  # conrespecto al sesgo de la capa de salida
  db2 = -error * y_hat * (1 - y_hat)

  # Actualizamos los sesgos
  b1 = b1 - alpha * db1
  b2 = b2 - alpha * db2
  
  # Construimos una matriz para las derivadas parciales
  # conrespecto a los pesos de la capa escondida
  dW1 = np.array([
      [-error * y_hat * (1 - y_hat) * W2[0] * dsigmoide(z1[0]) * x[row][0],
       -error * y_hat * (1 - y_hat) * W2[1] * dsigmoide(z1[1]) * x[row][0]],
      [-error * y_hat * (1 - y_hat) * W2[0] * dsigmoide(z1[0]) * x[row][1],
       -error * y_hat * (1 - y_hat) * W2[1] * dsigmoide(z1[1]) * x[row][1]]
  ])


  # Construimos un vector para las derivadas parciales
  # conrespecto a los pesos de la capa de salida
  dW2 = np.array([
      -error * y_hat * (1 - y_hat) * a1[0],
      -error * y_hat * (1 - y_hat) * a1[1]
  ])
  
  # Actualizamos los pesos
  W1 = W1 - alpha * dW1
  W2 = W2 - alpha * dW2

# Entrenamos la red neuronal
epochs = 1000
errors = []
for epoch in range(epochs):
  y_preds = []
  for row in range(len(x)):
    a1, a2, z1, z2 = forward_prop(x[row])
    y_preds.append(a2)
    backward_prop(a2, row, x, y, a1, z1)
  # Calculamos el error cuadrático medio y comparamos la calidad de las respuestas
  mse = np.mean((np.array(y_preds) - y_train) ** 2)
  errors.append(mse)

print("Error cuadrático medio final en entrenamiento:", mse)

# for row in range(len(x)):
#   a1, a2, z1, z2 = forward_prop(x[row])
#   backward_prop(a2, row, x, y, a1, z1)

# Probamos la red neuronal
x = x_test
y = y_test

y_preds = []
for row in range(len(x)):
  a1, a2, z1, z2 = forward_prop(x[row])
  y_preds.append(a2)

# Calculamos el error cuadrático medio en el conjunto de prueba
mse = np.mean((np.array(y_preds) - y_test) ** 2)
print("Error cuadrático medio en prueba:", mse)

# Graficamos las predicciones y los puntos reales
plt.figure(figsize=(10, 5))
plt.plot(y_test, label='Puntos Reales')
plt.plot(y_preds, label='Predicciones')
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.title('Predicciones vs Puntos Reales')
plt.legend()
plt.show()

# Graficamos el error cuadrático medio en función de las épocas
plt.figure(figsize=(10, 5))
plt.plot(errors, label='Error Cuadrático Medio')
plt.xlabel('Épocas')
plt.ylabel('Error Cuadrático Medio')
plt.title('Error Cuadrático Medio vs Épocas')
plt.legend()
plt.show()