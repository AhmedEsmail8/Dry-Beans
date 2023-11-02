import numpy as np


class Adaline:
    def __init__(self, learning_rate=0.001, epochs=200):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def train(self, X, y):
        num_features = X.shape[1]
        np.random.seed(42)
        self.weights = np.random.randn(num_features)
        self.bias = np.random.randn()

        for _ in range(self.epochs):
            output = self.activation(self.net_input(X))
            error = y - output
            self.weights += self.learning_rate * np.dot(X.T, error)
            self.bias += self.learning_rate * np.sum(error)

    def net_input(self, X):
        return np.dot(X, self.weights) + self.bias

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)