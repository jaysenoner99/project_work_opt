import numpy as np
from numpy import linalg as la

# Parameters for Armijo line_search
gamma = 0.3
delta = 0.25
max_iter_armijo = 1000
initial_stepsize = 1

# Parameters for Bisection
max_iter_bisection = 1000
tol_bis = 0.001

# Solver parameters
max_iter = 1000
tol = 0.00001   #For the stopping criterion norm(new_weights - weights) < tol
eps = 0.000001  #For the stopping criterion on newton decrement and norm of the gradient


class Solver:
    def __init__(self):
        pass

    # Armijo line search with max number of iterations
    def armijo_line_search(self, dataset, weights, direction):
        alpha = initial_stepsize
        old_loss = dataset.compute_log_loss(weights)
        grad = dataset.gradient(weights)
        for _ in range(max_iter_armijo):
            new_weights = weights + alpha * direction
            new_loss = dataset.compute_log_loss(new_weights)
            if new_loss <= old_loss + gamma * alpha * np.dot(grad.T, direction):
                return alpha
            else:
                alpha *= delta
        print("Armijo: number of iterations > max_iter_armijo")
        return alpha

    # TODO: bisection not working
    def bisection(self, dataset, weights, direction, alpha_a, alpha_b):
        if dataset.compute_loss_step_derivative(weights, alpha_a, direction) * dataset.compute_loss_step_derivative(
                weights, alpha_b, direction) > 0:
            raise ValueError("Bisection method failed: loss function derivative has the same sign when eval. on "
                             "alpha_a and"
                             "alpha_b")
        i = 0
        while (alpha_b - alpha_a) / 2 > tol_bis and i < max_iter_bisection:
            mid = (alpha_a + alpha_b) / 2
            loss_mid_point = dataset.compute_loss_step_derivative(weights, mid, direction)
            if loss_mid_point == 0:
                break
            elif loss_mid_point * dataset.compute_loss_step_derivative(weights, alpha_a, direction) < 0:
                alpha_b = mid
            else:
                alpha_a = mid
            i += 1
        if i == max_iter_bisection:
            print("Bisection reached max number of iterations")
        return (alpha_a + alpha_b) / 2

    def secant(self, dataset, weights, direction):
        pass

    # Standard Newton method (alpha = 1)
    def standard_newton_cholesky(self, dataset, initial_weights):
        weights = initial_weights
        for i in range(max_iter):
            grad = dataset.gradient(weights)
            hess = dataset.hessian(weights, 1)
            L = la.cholesky(hess)
            w = la.solve(L, -grad)
            direction = la.solve(L.T, w)
            newton_decrement = la.norm(w) ** 2
            if newton_decrement / 2 <= eps:
                return i, weights

            new_weights = weights + direction
            if dataset.compute_log_loss(new_weights) >= dataset.compute_log_loss(weights):
                step_size = self.armijo_line_search(dataset, weights, direction)
                new_weights = weights + step_size * direction
            weights = new_weights
        print("Standard newton method with cholesky factorization exceeded max number of iterations")
        return i, weights

    def standard_newton(self, dataset, initial_weights):
        weights = initial_weights
        for i in range(max_iter):
            grad = dataset.gradient(weights)
            hess = dataset.hessian(weights, 1)
            direction = la.solve(hess, -grad)
            new_weights = weights + direction
            # If the objective function has increased, compute armijo step.
            if dataset.compute_log_loss(new_weights) >= dataset.compute_log_loss(weights):
                step_size = self.armijo_line_search(dataset, weights, direction)
                new_weights = weights + step_size * direction
            if la.norm(new_weights - weights) < tol:
                return i, new_weights
            weights = new_weights

        print("Standard newton method exceeded max iter")
        return i, weights

    #Gradient Descent algorithm
    def gradient_descent(self, dataset, initial_weights):
        weights = initial_weights
        for i in range(max_iter):
            direction = -dataset.gradient(weights)
            if la.norm(direction) < eps:
                return i, weights
            step_size = self.armijo_line_search(dataset, weights, direction)
            weights = weights + step_size * direction

        print("Gradient Descent: number of iterations > max_iter")
        return i, weights

    # TODO: not working
    def gradient_descent_exact(self, dataset, initial_weights):
        weights = initial_weights
        for i in range(max_iter):
            direction = -dataset.gradient(weights)
            gradient_norm = la.norm(direction)
            print("Gradient norm at iteration " + str(i) + ": " + str(gradient_norm))
            if gradient_norm < eps:
                return i, weights
            step_size = self.bisection(dataset, weights, direction, 0, 3)
            weights = weights + step_size * direction
        print("Gradient Descent with exact line search: number of iterations > max_iter")
        return i, weights

    #Newton method with armijo line search and cholesky decomposition
    def newton_armijo_cholesky(self, dataset, initial_weights):
        weights = initial_weights
        for i in range(max_iter):
            grad = dataset.gradient(weights)
            hess = dataset.hessian(weights)
            L = la.cholesky(hess)
            w = la.solve(L, -grad)
            direction = la.solve(L.T, w)
            newton_decrement = la.norm(w) ** 2
            if newton_decrement / 2 <= eps:
                return i, weights
            step_size = self.armijo_line_search(dataset, weights, direction)
            weights = weights + step_size * direction
        print("Newton method with armijo line search exceeded max number of iterations")
        return i, weights

    def newton_armijo(self,dataset,initial_weights):
        weights = initial_weights
        for i in range(max_iter):
            grad = dataset.gradient(weights)
            hess = dataset.hessian(weights)
            direction = la.solve(hess, -grad)
            step_size = self.armijo_line_search(dataset, weights, direction)
            new_weights = weights + step_size
            if la.norm(new_weights - weights) < tol:
                return i, new_weights
            weights = new_weights
        print("Standard newton with armijo line search exceeded max number of iterations")
        return i,weights
