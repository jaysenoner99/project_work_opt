import numpy as np
from numpy import linalg as la

# Prediciton threshold
threshold = 0
# Regularization strength
lambda_reg = 1


class Dataset:

    def __init__(self):
        self.data = None
        self.labels = None
        self.rng = np.random.default_rng()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def computeLogLoss(self, weights):
        loss = 0
        for x_i, y_i in zip(self.data, self.labels):
            loss = loss + np.log(1 + np.exp(-y_i * np.dot(weights, x_i.T)))
        loss = loss + lambda_reg * la.norm(weights) ** 2

        return loss

    def generate_dataset(self, num_observations, num_features):
        true_weights = self.generate_true_weights(num_features)
        self.data, self.labels = self.generate_examples(num_observations, num_features, true_weights)
        return self.data, true_weights, self.labels

    def generate_examples(self, num_observations, num_features, true_weights):
        X = self.rng.standard_normal(size=(num_observations, num_features))
        X = np.c_[X, np.ones((num_observations, 1))]
        labels = np.sign(np.dot(X, true_weights))
        return X, labels

    def generate_true_weights(self, num_features):
        return self.rng.standard_normal(num_features + 1)

    def gradient(self, weights):
        r = np.zeros(self.data.shape[0])
        i = 0
        for x_i, y_i in zip(self.data, self.labels):
            r[i] = -1 * y_i * self.sigmoid(-y_i * np.dot(weights, x_i.T))
            i += 1
        grad = np.dot(self.data.T, r) + 2 * lambda_reg * weights
        return grad

    def better_gradient(self, weights):
        r = np.multiply(-self.labels, self.sigmoid(np.multiply(-self.labels, np.dot(self.data,weights))))
        return np.matmul(self.data.T, r) + 2 * lambda_reg * weights

    def hessian(self, weights):
        d = np.zeros(self.data.shape[0])
        i = 0
        for x_i, y_i in zip(self.data, self.labels):
            d[i] = self.sigmoid(y_i * np.dot(weights, x_i.T)) * self.sigmoid(-y_i * np.dot(weights, x_i))
        D = np.diag(d)
        return np.dot(np.dot(self.data.T, D), self.data) + 2 * lambda_reg * np.identity(weights.shape)

    def predict(self, weights, data):
        if self.sigmoid(np.dot(weights, data.T)) >= threshold:
            return 1
        else:
            return -1

    def test_prediction(self, num_examples, true_weights, final_weights):

        X, y = self.generate_examples(num_examples, self.data.shape[1] - 1, true_weights)
        good = 0
        for data, label in zip(X, y):
            prediction = self.predict(final_weights, data)
            if prediction == label:
                good += 1

        return good / num_examples
