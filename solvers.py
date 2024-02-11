import numpy as np
from numpy import linalg as la

# Parameters for Armijo line_search
gamma = 0.35
delta = 0.25
max_iter_armijo = 1000
initial_stepsize = 1

# Parameters for Bisection
max_iter_bisection = 1000
tol_bis = 0.001
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

    # Armijo line search with max number of iterations
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

    def bisection(self, dataset, weights, direction, alpha_a, alpha_b):
        if dataset.computeLogLoss(weights + alpha_a * direction) * dataset.computeLogLoss(
                weights + alpha_b * direction) > 0:
            raise ValueError("Bisection method failed: loss function has the same sign when eval. on alpha_a and "
                             "alpha_b")
        i = 0
        while (alpha_b - alpha_a) / 2 > tol_bis and i < max_iter_bisection:
            mid = (alpha_a + alpha_b) / 2
            loss_mid_point = dataset.computeLogLoss(weights + mid * direction)
            if loss_mid_point == 0:
                break
            elif loss_mid_point * dataset.computeLogLoss(weights + alpha_a * direction) < 0:
                alpha_b = mid
            else:
                alpha_a = mid
            i += 1
        if i == max_iter_bisection:
            print("Bisection reached max number of iterations")
        return (alpha_a + alpha_b)/2, i
    def secant(self, dataset, weights, direction):
        pass



    def gradientDescent(self, dataset, initial_weights):
        weights = initial_weights
        for i in range(max_iter):
            direction = -dataset.better_gradient(weights)
            gradient_norm = la.norm(direction)
            print("Gradient norm at iteration " + str(i) + ": " + str(gradient_norm))
            if gradient_norm < eps:
                return i, weights
            step_size = self.armijoLineSearch(dataset, weights, direction)
            weights = weights + step_size * direction


        print("Gradient Descent: number of iterations > max_iter")
        return i, weights
