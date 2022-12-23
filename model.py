import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import json

# import dataset and store in training set and testing set
data = pd.read_csv('train.csv')

"""Import data using kaggle"""
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
#normalize values
X_dev = X_dev / 255

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
# normalize values
X_train = X_train / 255

"""Import data usng keras"""
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# create numpy array of x_train
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2]))

# create numpy array of x_test
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]*x_test.shape[2]))

# transpose all inputs
# normalize values
x_train = x_train.T / 255
y_train = y_train.T
# normalize values
x_test = x_test.T / 255
y_test = y_test.T


# initilalize weights and biases
def init_weights_and_biases():
    w1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5 
    return w1, b1, w2, b2

# ReLu activation function
def ReLu(Z):
    return np.maximum(Z, 0)

# SoftMax activation function
def SoftMax(Z):
     A = np.exp(Z) / sum(np.exp(Z))
     return A

# forward propagation
def forward_prop(w1, b1, w2, b2, X):
    # first layer
    # calculate unactivated first layer
    Z1 = w1.dot(X) + b1
    # calculate activated first layer
    A1 = ReLu(Z1)

    # second layer
    # calculate unactivated second layer
    Z2 = w2.dot(A1) + b2
    # calculate activated second layer
    A2 = SoftMax(Z2)

    return Z1, A1, Z2, A2

# encode Y to correspond to individual outputs
def one_hot_encode_Y(y):
    # create new matrix filled with zeros (with tuple of its size)
    one_hot_Y = np.zeros((y.size, y.max() + 1))
    # for each row, go to the column specified by y and set it to 1
    one_hot_Y[np.arange(y.size), y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

# derivative of ReLu activation function
def deriv_ReLu(Z):
    return Z > 0

# back propagation
def back_prop(Z1, A1, Z2,A2, w2, X, Y):
    m = Y.size

    # second layer
    one_hot = one_hot_encode_Y(Y)
    dZ2 = A2 - one_hot
    dw2 = 1/m * dZ2.dot(A1.T)
    db2 = 1/m * np.sum(dZ2)

    # first layer
    dZ1 = w2.T.dot(dZ2) * deriv_ReLu(Z1)
    dw1 = 1/m * dZ1.dot(X.T)
    db1 = 1/m * np.sum(dZ1)
    return dw1, db1, dw2, db2

# adjust the weights and biases for each neuron
def adjust_weights_and_biases(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(p, Y):
    print(p, Y)
    return np.sum(p == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
    w1, b1, w2, b2 = init_weights_and_biases()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(w1, b1, w2, b2, X)
        dw1, db1, dw2, db2 = back_prop(Z1, A1, Z2, A2, w2, X, Y)
        w1, b1, w2, b2 = adjust_weights_and_biases(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        if i % 100 == 0:
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(A2), Y))
    return w1, b1, w2, b2

w1, b1, w2, b2 = gradient_descent(x_test, y_test, 10, 0.5)

def make_prediction(x, w1, b1, w2, b2):
    _, _, _, A2 = forward_prop(w1, b1, w2, b2, x)
    return get_predictions(A2)

def test_prediction(index, w1, b1, w2, b2):
    current = x_test[:, index, None]
    prediction = make_prediction(current, w1, b1, w2, b2)
    label = y_test[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current = current.reshape((28, 28)) * 255
    plt.imshow(current, cmap='gray')
    plt.show()

for i in range(25):
    test_prediction(i, w1, b1, w2, b2)
