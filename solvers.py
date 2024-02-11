import numpy as np
from numpy import linalg as la

# Parameters for Armijo line_search
gamma = 0.35
delta = 0.25
max_iter_armijo = 1000
initial_stepsize = 1

# Solver parameters
max_iter = 1000
tol = 0.00001
eps = 0.000001


class Solver:
    def __init__(self):
        pass

    # Standard Newton method (alpha = 1)
    def standardNewton(self, dataset, initial_weights):
        pass

    # Armijo line search with limited max number of iterations
    def armijoLineSearch(self, dataset, weights, direction):
        alpha = initial_stepsize
        old_loss = dataset.computeLogLoss(weights)
        grad = dataset.gradient(weights)
        for _ in range(max_iter_armijo):
            new_weights = weights + alpha * direction
            new_loss = dataset.computeLogLoss(new_weights)
            if new_loss <= old_loss + gamma * alpha * np.dot(grad.T, direction):
                return alpha
            else:
                alpha *= delta
        print("Armijo: number of iterations > max_iter_armijo")
        return alpha

    #def exactLineSearch(self, dataset, weights, direction):

    def gradientDescent(self, dataset, initial_weights):
        weights = initial_weights
        for i in range(max_iter):
            direction = -dataset.gradient(weights)
            gradient_norm = la.norm(direction)
            print("Gradient norm at iteration " + str(i) + ": " + str(gradient_norm))
            if gradient_norm < eps:
                return i,weights
            step_size = self.armijoLineSearch(dataset, weights, direction)
            weights = weights + step_size * direction

        print("Gradient Descent: number of iterations > max_iter")
        return i, weights
