import numpy as np

from Dataset import Dataset
from solvers import Solver

if __name__ == '__main__':
    dataset = Dataset()
    solver = Solver()
    X, true_weights, labels = dataset.generate_dataset(500, 20)
    initial_weights = np.zeros(true_weights.shape)
    dataset.set_optimal_point(initial_weights)

    dataset.test_solver(initial_weights, solver.gradient_descent)
    dataset.test_solver(initial_weights, solver.newton_armijo_cholesky)

    dataset.test_solver(initial_weights, solver.standard_newton_cholesky)
    dataset.test_solver(initial_weights, solver.standard_newton)
