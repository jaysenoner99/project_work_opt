import numpy as np
from matplotlib import pyplot as plt
from Dataset import Dataset
from solvers import Solver

if __name__ == '__main__':
    dataset = Dataset()
    solver = Solver()
    X, true_weights, labels = dataset.generate_dataset(500, 20)
    initial_weights = np.zeros(true_weights.shape,dtype='float64')
    dataset.set_optimal_point(initial_weights)
    dataset.test_solver(initial_weights, solver.gradient_descent)
    dataset.test_solver(initial_weights, solver.newton_armijo)
    dataset.test_solver(initial_weights, solver.standard_newton)
    dataset.test_solver(initial_weights, solver.gradient_descent_exact)
    dataset.test_solver(initial_weights, solver.greedy_newton)
    dataset.test_solver(initial_weights, solver.hybrid_newton)
    plt.savefig("Plot/all")