import numpy as np

# Inicializamos los parámetros
input_size = 2
hidden_size = 2
output_size = 1

# Pesos y sesgos para la capa escondida
W1 = np.random.rand(input_size, hidden_size)
b1 = np.random.rand(hidden_size)

# Pesos y sesgos para la capa de salida
W2 = np.random.rand(hidden_size, output_size)
b2 = np.random.rand(output_size)

# Tasa de aprendizaje
alpha = 0.001

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)
    

def forward_prop(X):
    # Capa escondida
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    
    # Capa de salida
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    
    return a1, a2

def backward_prop(X, y, a1, a2):
    global W1, b1, W2, b2

    n = X.shape[0]
    
    # Error en la capa de salida
    error_output = y - a2
    delta_output = error_output * sigmoid_derivative(a2)

    # Gradientes para la capa de salida
    dW2 = np.dot(a1.T, delta_output) / n
    db2 = np.sum(delta_output, axis=0) / n
    
    # Error en la capa escondida
    error_hidden = delta_output.dot(W2.T)
    delta_hidden = error_hidden * sigmoid_derivative(a1)
    
    # Gradientes para la capa escondida
    dW1 = np.dot(X.T, delta_hidden) / n
    db1 = np.sum(delta_hidden, axis=0) / n
    
    # Actualizar los pesos y sesgos
    W2 += alpha * a1.T.dot(delta_output)
    b2 += alpha * np.sum(delta_output, axis=0)
    W1 += alpha * X.T.dot(delta_hidden)
    b1 += alpha * np.sum(delta_hidden, axis=0)


    # Datos de entrenamiento (ejemplo)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])  # XOR

# Número de épocas
epochs = 10000

for epoch in range(epochs):
    # Propagación hacia adelante
    a1, a2 = forward_prop(X)
    
    # Retropropagación
    backward_prop(X, y, a1, a2)
    
    # Imprimir el error cada 1000 épocas
    if (epoch + 1) % 1000 == 0:
        loss = np.mean(np.square(y - a2))
        print(f"Epoch {epoch + 1}, Loss: {loss}")

# Predicciones finales
_, predictions = forward_prop(X)
print("Predicciones:")
print(predictions)