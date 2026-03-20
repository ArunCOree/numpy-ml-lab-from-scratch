import numpy as np
from utils import shuffle_data, add_bias_term

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None

    def fit(self, x, y):
        x = add_bias_term(x)
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)

        for _ in range(self.n_iterations):
            y_predicted = np.dot(x, self.weights)
            dw = (1 / n_samples) * np.dot(x.T, (y_predicted - y))
            self.weights -= self.learning_rate * dw

    def predict(self, x):
        x = add_bias_term(x)
        return np.dot(x, self.weights)