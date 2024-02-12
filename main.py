import numpy as np

from Dataset import Dataset
from solvers import Solver



if __name__ == '__main__':
    dataset = Dataset()
    solver = Solver()
    X, true_weights, labels = dataset.generate_dataset(500, 20)
    initial_weights = np.zeros(X.shape[1])
    dataset.test_solver(initial_weights, true_weights,solver.gradient_descent)
    dataset.test_solver(initial_weights, true_weights,solver.newton_armijo)
    dataset.test_solver(initial_weights, true_weights, solver.newton_armijo_slower)
    dataset.test_solver(initial_weights, true_weights, solver.standard_newton)
    dataset.test_solver(initial_weights, true_weights,solver.standard_newton_slower)