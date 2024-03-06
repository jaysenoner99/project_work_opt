import numpy as np
from matplotlib import pyplot as plt
from Dataset import Dataset
from solvers import Solver
from numpy import linalg as la


def adapt_array(error_array, max_iter_reached):
    for i in range(max_iter_reached - len(error_array)):
        error_array = np.append(error_array, error_array[-1])
    return error_array


def test_all_and_plot(plot_name,num_observations=500, num_features=20, repeated_features=False):
    solver = Solver()
    dataset = Dataset()
    # Prepare the dataset
    X, true_weights, labels = dataset.generate_dataset(num_observations, num_features,repeated_features)
    initial_weights = np.zeros(true_weights.shape, dtype='float64')
    dataset.set_optimal_point(initial_weights)

    # Tests

    gradient_descent,time_gd = dataset.test_solver(initial_weights, solver.gradient_descent)
    newton_armijo,time_na = dataset.test_solver(initial_weights, solver.newton_armijo)
    standard_newton,time_sn = dataset.test_solver(initial_weights, solver.standard_newton)
    gradient_descent_exact,time_gde = dataset.test_solver(initial_weights, solver.gradient_descent_exact)
    greedy_newton,time_gn = dataset.test_solver(initial_weights, solver.greedy_newton)
    hybrid_newton,time_hn = dataset.test_solver(initial_weights, solver.hybrid_newton)


    #Rescale time measure from seconds to milliseconds
    time_gd = np.multiply(time_gd,1000)
    time_na = np.multiply(time_na,1000)
    time_sn = np.multiply(time_sn,1000)
    time_gde = np.multiply(time_gde,1000)
    time_gn = np.multiply(time_gn,1000)
    time_hn = np.multiply(time_hn,1000)
    max_iter_reached = max([len(gradient_descent), len(newton_armijo),
                            len(standard_newton), len(gradient_descent_exact), len(greedy_newton)
                               , len(hybrid_newton)])
    gradient_descent = adapt_array(gradient_descent, max_iter_reached)
    newton_armijo = adapt_array(newton_armijo, max_iter_reached)
    standard_newton = adapt_array(standard_newton, max_iter_reached)
    gradient_descent_exact = adapt_array(gradient_descent_exact, max_iter_reached)
    greedy_newton = adapt_array(greedy_newton, max_iter_reached)
    hybrid_newton = adapt_array(hybrid_newton, max_iter_reached)


    #print("difference between error vectors of egd and gn:",gradient_descent_exact-greedy_newton)
    #print("precisione",np.finfo(float).eps)
    x = np.arange(max_iter_reached)


    # Plotting all values in the same figure
    fig, ax = plt.subplots()
    plt.plot(x, gradient_descent[:len(x)], label='Gradient Descent')
    plt.plot(x, gradient_descent_exact[:len(x)], label='Gradient Descent Exact')
    plt.plot(x, newton_armijo[:len(x)], label='Newton Armijo')
    plt.plot(x, standard_newton[:len(x)], label='Standard Newton')
    plt.plot(x, greedy_newton[:len(x)], label='Greedy Newton')
    plt.plot(x, hybrid_newton[:len(x)], label='Hybrid Newton')


    plt.xlabel('Iteration')
    plt.ylabel('f(xk) - f*')
    plt.legend()
    plt.grid(True)
    ax.set_yscale('log')
    plt.savefig("Plot/" + plot_name + ".pdf")

    fig,ax = plt.subplots()
    plt.plot(time_gd,gradient_descent[:len(time_gd)],label = 'Gradient Descent', marker='.')
    plt.plot(time_gde,gradient_descent_exact[:len(time_gde)],label = 'Gradient Descent Exact', marker='.')
    plt.plot(time_na,newton_armijo[:len(time_na)],label = 'Newton Armijo', marker='.')
    plt.plot(time_sn,standard_newton[:len(time_sn)],label = 'Standard Newton', marker='.')
    plt.plot(time_gn,greedy_newton[:len(time_gn)],label = 'Greedy Newton', marker='.')
    plt.plot(time_hn,hybrid_newton[:len(time_hn)],label = 'Hybrid Newton', marker='.')

    plt.xlabel('time(ms)')
    plt.ylabel('f(xk) - f*')
    plt.legend()
    plt.grid(True)
    ax.set_yscale('log')
    plt.savefig("Plot/" + plot_name + "time" + ".pdf")

if __name__ == '__main__':
    test_all_and_plot("p=20,regularized",500, 20)
    #test_all_and_plot("p=20,regularized,repeated_features",500,20,True)
    #test_all_and_plot("p=200,regularized",500,200)
    #test_all_and_plot("p=2000,regularized",500, 2000)
