import numpy as np
from numpy import linalg as la
from scipy import optimize as opt
from timeit import default_timer as timer
import Dataset

# Parameters for armijo line search
gamma = 0.3
delta = 0.25
max_iter_armijo = 1000
initial_stepsize = 3



# gamma = 0.3
# delta = 0.25
# max_iter_armijo = 1000
# initial_stepsize = 3.5

# Parameters for Approx. exact line search(AELS)
beta = 2 / (1 + np.sqrt(5))
max_iter_AELS = 1000
initial_stepsizeAELS = 1.7

# Solver parameters
max_iter = 1000
tol = 0.000001  # For the stopping criterion norm(new_weights - weights) < tol
eps = 0.000001  # For the stopping criterion on newton decrement and norm of the gradient


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




    # Gradient Descent algorithm
    def gradient_descent(self, dataset, initial_weights):
        weights = initial_weights
        error = np.array([], dtype='float64')
        optimal_loss = dataset.compute_log_loss(dataset.optimal_point)
        error = np.append(error, dataset.compute_log_loss(initial_weights) - optimal_loss)
        time_array = [0]
        time_index = 1
        for i in range(max_iter):
            start = timer()
            direction = -dataset.gradient(weights)
            if la.norm(direction) < eps:
                return i, weights, error, time_array
            step_size = self.armijo_line_search(dataset, weights, direction)
            weights = weights + step_size * direction
            end = timer()
            time_array.append(end - start + time_array[time_index - 1])
            time_index += 1
            error = np.append(error, dataset.compute_log_loss(weights) - optimal_loss)
        print("Gradient Descent: number of iterations > max_iter")
        return i, weights, error, time_array

    # Gradient descent with exact line search
    def gradient_descent_exact(self, dataset, initial_weights, aels=True):
        weights = initial_weights
        error = np.array([], dtype='float64')
        optimal_loss = dataset.compute_log_loss(dataset.optimal_point)
        error = np.append(error, dataset.compute_log_loss(initial_weights) - optimal_loss)
        time_array = [0]
        time_index = 1
        step_size = initial_stepsizeAELS
        for i in range(max_iter):
            start = timer()
            direction = -dataset.gradient(weights)
            if la.norm(direction) < eps:
                return i, weights, error, time_array
            if aels is True:
                step_size = self.AELS(dataset, weights, step_size, direction)
            else:
                step_size = opt.minimize_scalar(dataset.step_log_loss, args=(weights, direction)).x
            weights = weights + step_size * direction
            end = timer()
            time_array.append(end - start + time_array[time_index - 1])
            time_index += 1
            error = np.append(error, dataset.compute_log_loss(weights) - optimal_loss)
        print("Gradient Descent with exact line search exceeded max number of iterations")
        return i, weights, error, time_array

    # Newton method with armijo line search and cholesky decomposition
    # def newton_armijo_cholesky(self, dataset, initial_weights):
    #     if len(initial_weights) - 1 == 2000 or dataset.repeated_features is True:
    #         hess_trick = 1
    #     else:
    #         hess_trick = 0
    #     weights = initial_weights
    #     error = np.array([], dtype='float64')
    #     optimal_loss = dataset.compute_log_loss(dataset.optimal_point)
    #     error = np.append(error, dataset.compute_log_loss(initial_weights) - optimal_loss)
    #     for i in range(max_iter):
    #         grad = dataset.gradient(weights)
    #         if la.norm(grad) < eps:
    #             return i, weights, error
    #         hess = dataset.hessian(weights, hess_trick)
    #         L = la.cholesky(hess)
    #         w = la.solve(L, -grad)
    #         direction = la.solve(L.T, w)
    #         step_size = self.armijo_line_search(dataset, weights, direction)
    #         weights = weights + step_size * direction
    #         error = np.append(error, dataset.compute_log_loss(weights) - optimal_loss)
    #     print("Newton method with armijo line search exceeded max number of iterations")
    #     return i, weights, error

    # Newton method with armijo line search
    def newton_armijo(self, dataset, initial_weights):
        if len(initial_weights) - 1 == 2000 or dataset.repeated_features is True or Dataset.lambda_reg == 0:
            hess_trick = 1
        else:
            hess_trick = 0
        weights = initial_weights
        error = np.array([], dtype='float64')
        optimal_loss = dataset.compute_log_loss(dataset.optimal_point)
        error = np.append(error, dataset.compute_log_loss(initial_weights) - optimal_loss)
        time_array = [0]
        time_index = 1
        for i in range(max_iter):
            start = timer()
            grad = dataset.gradient(weights)
            if la.norm(grad) < eps:
                return i, weights, error, time_array
            hess = dataset.hessian(weights, hess_trick)
            direction = la.solve(hess, -grad)
            step_size = self.armijo_line_search(dataset, weights, direction)
            weights = weights + step_size * direction
            end = timer()
            time_array.append(end - start + time_array[time_index - 1])
            time_index += 1
            error = np.append(error, dataset.compute_log_loss(weights) - optimal_loss)
        print("Standard newton with armijo line search exceeded max number of iterations")
        return i, weights, error, time_array

    # Greedy Newton method(Newton method with exact line search)
    def greedy_newton(self, dataset, initial_weights, aels=True):
        if len(initial_weights) - 1 == 2000 or dataset.repeated_features is True or Dataset.lambda_reg == 0:
            hess_trick = 1
        else:
            hess_trick = 0
        weights = initial_weights
        error = np.array([], dtype='float64')
        optimal_loss = dataset.compute_log_loss(dataset.optimal_point)
        error = np.append(error, dataset.compute_log_loss(initial_weights) - optimal_loss)
        time_array = [0]
        time_index = 1
        step_size = initial_stepsizeAELS
        for i in range(max_iter):
            start = timer()
            grad = dataset.gradient(weights)
            if la.norm(grad) < eps:
                return i, weights, error, time_array
            hess = dataset.hessian(weights, hess_trick)
            direction = la.solve(hess, -grad)
            if aels is True:
                step_size = self.AELS(dataset, weights, step_size, direction)
            else:
                step_size = opt.minimize_scalar(dataset.step_log_loss, args=(weights, direction)).x
            weights = weights + step_size * direction
            end = timer()
            time_array.append(end - start + time_array[time_index - 1])
            time_index += 1
            error = np.append(error, dataset.compute_log_loss(weights) - optimal_loss)
        print("Newton with exact line search exceeded max number of iterations")
        return i, weights, error, time_array

    def hybrid_newton(self, dataset, initial_weights, aels=True):
        if len(initial_weights) - 1 == 2000 or dataset.repeated_features is True or Dataset.lambda_reg == 0:
            hess_trick = 1
        else:
            hess_trick = 0
        weights = initial_weights
        error = np.array([], dtype='float64')
        optimal_loss = dataset.compute_log_loss(dataset.optimal_point)
        error = np.append(error, dataset.compute_log_loss(initial_weights) - optimal_loss)
        time_array = [0]
        time_index = 1
        exact_stepsize = initial_stepsizeAELS
        for i in range(max_iter):
            start = timer()
            grad = dataset.gradient(weights)
            if la.norm(grad) < eps:
                return i, weights, error, time_array
            hess = dataset.hessian(weights, hess_trick)
            newton_direction = la.solve(hess, -grad)

            newton_pure_step = weights + newton_direction
            if aels is True:
                exact_stepsize = self.AELS(dataset, weights, exact_stepsize, -grad)
            else:
                exact_stepsize = opt.minimize_scalar(dataset.step_log_loss, args=(weights, -grad)).x
            exact_gradient_step = weights - exact_stepsize * grad
            if dataset.compute_log_loss(exact_gradient_step) < dataset.compute_log_loss(newton_pure_step):
                weights = exact_gradient_step
            else:
                weights = newton_pure_step
            end = timer()
            time_array.append(end - start + time_array[time_index - 1])
            time_index += 1
            error = np.append(error, dataset.compute_log_loss(weights) - optimal_loss)
        print("Hybrid Newton method exceeded max number of iterations")
        return i, weights, error, time_array

    #Approx. exact line search routine.
    def AELS(self, dataset, weights, last_stepsize, search_direction):
        step_size = last_stepsize
        old_loss = dataset.compute_log_loss(weights)
        new_temp_loss = dataset.step_log_loss(step_size, weights, search_direction)
        alpha = beta
        if new_temp_loss < old_loss:
            alpha = 1 / beta
        for _ in range(max_iter_AELS):
            step_size = alpha * step_size
            old_loss = new_temp_loss
            new_temp_loss = dataset.step_log_loss(step_size, weights, search_direction)
            if new_temp_loss >= old_loss:
                break
        if step_size == last_stepsize / beta:
            step_size = last_stepsize
            alpha = beta
            new_temp_loss, old_loss = old_loss, new_temp_loss
            for _ in range(max_iter_AELS):
                step_size = alpha * step_size
                old_loss = new_temp_loss
                new_temp_loss = dataset.step_log_loss(step_size, weights, search_direction)
                if new_temp_loss > old_loss:
                    break
        if alpha < 1:
            return step_size
        return (beta ** 2) * step_size


    def standard_newton(self, dataset, initial_weights):
        if len(initial_weights) - 1 == 2000 or dataset.repeated_features is True or Dataset.lambda_reg == 0:
            hess_trick = 1
        else:
            hess_trick = 0
        weights = initial_weights
        count_armijo = 0
        error = np.array([], dtype='float64')
        time_array = [0]
        time_index = 1
        optimal_loss = dataset.compute_log_loss(dataset.optimal_point)
        error = np.append(error, dataset.compute_log_loss(initial_weights) - optimal_loss)
        for i in range(max_iter):
            start = timer()
            grad = dataset.gradient(weights)
            if la.norm(grad) < eps:
                print("iter e armijo steps:", i, count_armijo)
                return i, weights, error, time_array
            hess = dataset.hessian(weights, hess_trick)
            direction = la.solve(hess, -grad)

            new_weights = weights + direction
            new_loss = dataset.compute_log_loss(new_weights)
            old_loss = dataset.compute_log_loss(weights)

            # If the objective function has increased, compute armijo step.
            if new_loss >= old_loss:
                count_armijo += 1
                step_size = self.armijo_line_search(dataset, weights, direction)
                new_weights = weights + step_size * direction

            weights = new_weights
            end = timer()
            time_array.append(end - start + time_array[time_index - 1])
            time_index += 1
            error = np.append(error, dataset.compute_log_loss(weights) - optimal_loss)
        print("Standard newton method exceeded max iter")
        print("iter e armijo steps:", i, count_armijo)
        return i, weights, error, time_array