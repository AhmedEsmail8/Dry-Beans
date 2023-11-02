import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.bias = None
        self.weights = None

    def initialize_weights(self, no_features):
        np.random.seed(42)
        self.weights = np.random.randn(no_features)
        self.bias = np.random.rand()

    def signum_activation(self, X):
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        return np.where(self.signum_activation(X) >= 0.0, 1, -1)

    def train_perceptron(self, X, Y):
        self.initialize_weights(X.shape[1])
        np.random.seed(42)
        for _ in range(self.n_iterations):
            for i in range(len(X)):
                prediction = self.predict(X[i])
                if prediction != Y[i]:
                    error = Y[i] - prediction
                    self.weights += self.learning_rate * X[i] * error
                    self.bias += self.learning_rate * error