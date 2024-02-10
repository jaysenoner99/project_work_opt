import numpy as np
from numpy import linalg as la

# Prediciton threshold
threshold = 0


class Dataset:

    def __init__(self):
        self.data = None
        self.labels = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def computeLogLoss(self, weights, lambda_reg=1):
        loss = 0
        for x_i, y_i in zip(self.data, self.labels):
            loss = loss + np.log(1 + np.exp(-y_i * np.dot(weights, x_i.T)))
        loss = loss + lambda_reg * la.norm(weights) ** 2

        return loss

    def generateData(self, num_observations, num_features):
        rng = np.random.default_rng()
        X = rng.standard_normal(size=(num_observations, num_features))
        X = np.c_[X, np.ones((num_observations, 1))]
        weights = rng.standard_normal(num_features + 1)
        labels = np.sign(np.dot(X, weights))
        self.data = X
        self.labels = labels
        return X, weights, labels

    def gradient(self, weights):
        r = np.zeros(self.data.shape[0])
        i = 0
        for x_i, y_i in zip(self.data, self.labels):
            r[i] = -y_i * self.sigmoid(-y_i * np.dot(weights, x_i.T))
            i += 1
        grad = np.dot(self.data.T, r)
        return grad

    def hessian(self, weights):
        d = np.zeros(self.data.shape[0])
        i = 0
        for x_i, y_i in zip(self.data, self.labels):
            d[i] = self.sigmoid(y_i * np.dot(weights, x_i.T)) * self.sigmoid(-y_i * np.dot(weights, x_i))
        D = np.diag(d)
        return np.dot(np.dot(self.data.T, D), self.data)

    def predict(self, weights, data):
        if self.sigmoid(np.dot(weights, data.T)) >= threshold:
            return 1
        else:
            return -1

    def test(self, final_weights):
        num_examples = self.data.shape[0]
        self.generateData(num_examples,self.data.shape[1]-1)
        good = 0
        for data, label in zip(self.data, self.labels):
            prediction = self.predict(final_weights, data)
            if prediction == label:
                good += 1

        return good/num_examples
