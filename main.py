import numpy as np
from matplotlib import pyplot as plt
from Dataset import Dataset
from solvers import Solver


def test_all_and_plot(num_observations=500, num_features=20):
    solver = Solver()
    dataset = Dataset()

    # Prepare the dataset
    X, true_weights, labels = dataset.generate_dataset(num_observations, num_features)
    initial_weights = np.zeros(true_weights.shape, dtype='float64')
    dataset.set_optimal_point(initial_weights)

    # Tests

    # dataset.test_solver(initial_weights, solver.gradient_descent)
    # dataset.test_solver(initial_weights, solver.newton_armijo)
    # dataset.test_solver(initial_weights, solver.standard_newton)
    # dataset.test_solver(initial_weights, solver.gradient_descent_exact)
    # dataset.test_solver(initial_weights, solver.greedy_newton)
    # dataset.test_solver(initial_weights, solver.hybrid_newton)

    gradient_descent = dataset.test_solver(initial_weights, solver.gradient_descent)
    newton_armijo = dataset.test_solver(initial_weights, solver.newton_armijo)
    standard_newton = dataset.test_solver(initial_weights, solver.standard_newton)
    gradient_descent_exact = dataset.test_solver(initial_weights, solver.gradient_descent_exact)
    greedy_newton = dataset.test_solver(initial_weights, solver.greedy_newton)
    hybrid_newton = dataset.test_solver(initial_weights, solver.hybrid_newton)
    x = np.arange(8)
    # Plotting all values in the same figure

    plt.plot(x, gradient_descent[:8], label='Gradient Descent')
    plt.plot(x, newton_armijo[:8], label='Newton Armijo')
    plt.plot(x, standard_newton[:8], label='Standard Newton')
    plt.plot(x, gradient_descent_exact[:8], label='Gradient Descent Exact')
    plt.plot(x, greedy_newton[:8], label='Greedy Newton')
    plt.plot(x, hybrid_newton[:8], label='Hybrid Newton')

    plt.xlabel('Iteration')
    plt.ylabel('f(xk) - f*')
    plt.legend()
    plt.title("all")
    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    test_all_and_plot(500,20)
