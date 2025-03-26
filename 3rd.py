import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Data
data = pd.read_csv('train.csv')
data = np.array(data)
np.random.shuffle(data)

# Split Data
m, n = data.shape
data_dev = data[:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:] / 255.0

def initate_paras():
    w1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return w1, b1, w2, b2


def Relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))

def forward(w1, b1, w2, b2, X):
    Z1 = w1.dot(X) + b1   # First layer weighted sum
    A1 = Relu(Z1)         # Activation function (ReLU)
    Z2 = w2.dot(A1) + b2  # Second layer weighted sum
    A2 = softmax(Z2)      # Activation function (Softmax)
    return Z1,A1,Z2,A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.max() + 1, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (Z1 > 0)
    dW1 = 1 / m * np.dot(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def updateparams(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2
def get_predictions(A2):
    return np.argmax(A2, axis=0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = initate_paras()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = updateparams(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 50 == 0:
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, Y)
            print(f"Iteration {i}: Accuracy = {accuracy}")
    return W1, b1, W2, b2

data_train = data[1000:].T
Y_train = data_train[0]
X_train = data_train[1:] / 255.0  # Normalize pixel values


W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 200, 0.1)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward(W1, b1, W2, b2, X)
    return get_predictions(A2)

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index].reshape((28, 28)) * 255
    prediction = make_predictions(X_train[:, [index]], W1, b1, W2, b2)
    label = Y_train[index]
    print(f"Prediction: {prediction}, Label: {label}")
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

test_prediction(3, W1, b1, W2, b2)
test_prediction(8, W1, b1, W2, b2)
