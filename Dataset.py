import numpy as np
from numpy import linalg as la
import scipy.optimize as sc
import matplotlib.pyplot as plt

# Prediciton threshold
threshold = 0.5
# Regularization strength
lambda_reg = 1


class Dataset:

    def __init__(self):
        self.data = None
        self.labels = None
        self.rng = np.random.default_rng()
        self.optimal_point = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def compute_log_loss(self, weights):
        return np.sum(np.log(1 + np.exp(-self.labels * np.dot(self.data, weights)))) + lambda_reg * la.norm(
            weights) ** 2

    def generate_dataset(self, num_observations, num_features):
        true_weights = self.generate_true_weights(num_features)
        self.data, self.labels = self.generate_examples(num_observations, num_features, true_weights)
        return self.data, true_weights, self.labels

    def set_optimal_point(self, initial_weights):
        self.optimal_point = sc.minimize(self.compute_log_loss, initial_weights).x

    def generate_examples(self, num_observations, num_features, true_weights):
        X = self.rng.standard_normal(size=(num_observations, num_features), dtype='float64')
        X = np.c_[X, np.ones((num_observations, 1))]
        labels = np.sign(np.dot(X, true_weights))
        return X, labels

    def generate_true_weights(self, num_features):
        return self.rng.standard_normal(num_features + 1)

    def gradient(self, weights):
        r = np.multiply(-self.labels, self.sigmoid(np.multiply(-self.labels, np.dot(self.data, weights))))
        return np.matmul(self.data.T, r) + 2 * lambda_reg * weights

    def hessian(self, weights, hess_trick=0):
        d = np.zeros(self.data.shape[0])
        i = 0
        for x_i, y_i in zip(self.data, self.labels):
            d[i] = self.sigmoid(y_i * np.dot(weights, x_i.T)) * self.sigmoid(-y_i * np.dot(weights, x_i))
        D = np.diag(d)
        return np.dot(np.dot(self.data.T, D), self.data) + 2 * lambda_reg * np.identity(
            weights.shape[0]) + hess_trick * 10 ** (-12) * np.identity(weights.shape[0])

    def compute_loss_step_derivative(self, alpha, weights, direction):
        return np.sum(
            -self.labels * self.sigmoid(-self.labels * np.dot(self.data, weights + alpha * direction)) * np.dot(
                self.data, direction))

    def step_log_loss(self,alpha,weights,direction):
        return self.compute_log_loss(weights + alpha*direction)

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

    def test_solver(self, initial_weights, solver_to_call):
        num_iter, solution,error_array = solver_to_call(self, initial_weights)
        solver_name = solver_to_call.__name__
        print("Solver:", solver_name)
        print("Number of iterations:", num_iter)
        print("Solution:", solution)
        print("Real optimal value", self.compute_log_loss(self.optimal_point))
        print("Optimal value found:", self.compute_log_loss(solution))
        print("Absolute error between the optimal value and solution:",
              np.abs(self.compute_log_loss(self.optimal_point) - self.compute_log_loss(solution)))
        print("True weights:", self.optimal_point)
        print("Percentage of good classifications:", self.test_prediction(200, self.optimal_point, solution))
        print("\n")
        #iter_array = np.array(range(len(error_array)))
        #self.plot(solver_name, iter_array, error_array)
        return error_array


    def plot(self, file_name,x,y):
        fig, ax = plt.subplots()
        plt.title(file_name)
        ax.plot(x,y)
        plt.grid(True)
        plt.savefig("Plot/" + file_name)



