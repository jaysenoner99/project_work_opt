import numpy as np
from matplotlib import pyplot as plt
from Dataset import Dataset
from solvers import Solver
from numpy import linalg as la
from timeit import default_timer as timer


def adapt_array(error_array, max_iter_reached):
    for i in range(max_iter_reached - len(error_array)):
        error_array = np.append(error_array, error_array[-1])
    return error_array


def init(num_observations, num_features, initial_weights, repeated_features=False):
    solver = Solver()
    dataset = Dataset()
    # Prepare the dataset
    X, true_weights, labels = dataset.generate_dataset(num_observations, num_features, repeated_features)
    dataset.set_optimal_point(initial_weights)
    return dataset, solver


def test_all_and_plot(plot_name, num_observations=500, num_features=20, repeated_features=False):
    # Dataset setup
    initial_weights = np.zeros(num_features + 1, dtype='float64')
    dataset, solver = init(num_observations, num_features, initial_weights, repeated_features)

    # Tests
    gradient_descent, time_gd = dataset.test_solver(initial_weights, solver.gradient_descent)
    gradient_descent_exact, time_gde = dataset.test_solver(initial_weights, solver.gradient_descent_exact)
    newton_armijo, time_na = dataset.test_solver(initial_weights, solver.newton_armijo)
    standard_newton, time_sn = dataset.test_solver(initial_weights, solver.standard_newton)

    greedy_newton, time_gn = dataset.test_solver(initial_weights, solver.greedy_newton)
    hybrid_newton, time_hn = dataset.test_solver(initial_weights, solver.hybrid_newton)

    # Rescale time measure from seconds to milliseconds

    time_gd = np.multiply(time_gd, 1000)
    time_gde = np.multiply(time_gde, 1000)
    time_na = np.multiply(time_na, 1000)
    time_sn = np.multiply(time_sn, 1000)

    time_gn = np.multiply(time_gn, 1000)
    time_hn = np.multiply(time_hn, 1000)

    max_iter_reached = max([len(gradient_descent), len(gradient_descent_exact), len(newton_armijo),
                            len(standard_newton), len(greedy_newton), len(hybrid_newton)])

    # Adapt arrays to fit the plot
    gradient_descent = adapt_array(gradient_descent, max_iter_reached)
    gradient_descent_exact = adapt_array(gradient_descent_exact, max_iter_reached)
    newton_armijo = adapt_array(newton_armijo, max_iter_reached)
    standard_newton = adapt_array(standard_newton, max_iter_reached)
    greedy_newton = adapt_array(greedy_newton, max_iter_reached)
    hybrid_newton = adapt_array(hybrid_newton, max_iter_reached)

    # print("difference between error vectors of egd and gn:",gradient_descent_exact-greedy_newton)
    # print("precisione",np.finfo(float).eps)
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

    fig, ax = plt.subplots()
    plt.plot(time_gd, gradient_descent[:len(time_gd)], label='Gradient Descent', marker='.')
    plt.plot(time_gde, gradient_descent_exact[:len(time_gde)], label='Gradient Descent Exact', marker='.')
    plt.plot(time_na, newton_armijo[:len(time_na)], label='Newton Armijo', marker='.')
    plt.plot(time_sn, standard_newton[:len(time_sn)], label='Standard Newton', marker='.')
    plt.plot(time_gn, greedy_newton[:len(time_gn)], label='Greedy Newton', marker='.')
    plt.plot(time_hn, hybrid_newton[:len(time_hn)], label='Hybrid Newton', marker='.')

    plt.xlabel('time(ms)')
    plt.ylabel('f(xk) - f*')
    plt.legend()
    plt.grid(True)
    ax.set_yscale('symlog')
    if num_features >= 2000:
        ax.set_xscale('symlog')
    plt.savefig("Plot/" + plot_name + "time" + ".pdf")


def test_compare(plot_name, num_observations=500, num_features=20, repeated_features=False):
    # Dataset setup
    initial_weights = np.ones(num_features + 1, dtype='float64')
    dataset, solver = init(num_observations, num_features, initial_weights, repeated_features)

    # Test methods without exact line search selection
    gradient_descent, time_gd = dataset.test_solver(initial_weights, solver.gradient_descent)
    newton_armijo, time_na = dataset.test_solver(initial_weights, solver.newton_armijo)
    standard_newton, time_sn = dataset.test_solver(initial_weights, solver.standard_newton)

    # Test methods with exact linesearch techniques(AELS and Brent method from scipy "minimize_scalar")
    # Without any specification the selected technique is AELS

    gradient_descent_exact_AELS, time_gdeaels = dataset.test_solver(initial_weights, solver.gradient_descent_exact)
    greedy_newton_AELS, time_gneaels = dataset.test_solver(initial_weights, solver.greedy_newton)
    hybrid_newton_AELS, time_hnaels = dataset.test_solver(initial_weights, solver.hybrid_newton)

    iter_gde_scipy, sol_gde, gde_scipy, time_gde_scipy = solver.gradient_descent_exact(dataset, initial_weights, False)
    iter_gn_scipy, sol_gn, gn_scipy, time_gn_scipy = solver.greedy_newton(dataset, initial_weights, False)
    iter_hn_scipy, sol_hn, hn_scipy, time_hn_scipy = solver.hybrid_newton(dataset, initial_weights, False)

    # Check the exact search technique that has better performance in terms of number of iterations
    # #of iterations:
    if iter_gde_scipy > gradient_descent_exact_AELS.shape[0]:
        gradient_descent_exact = gradient_descent_exact_AELS
        print("gde iterations: aels")
        iter_gde_label = 'gde_aels'
    else:
        gradient_descent_exact = gde_scipy
        iter_gde_label = 'gde_scipy'

    if iter_gn_scipy > greedy_newton_AELS.shape[0]:
        greedy_newton = greedy_newton_AELS
        print("greedy newton iterations: aels")
        iter_gn_label = 'gn_aels'
    else:
        greedy_newton = gn_scipy
        iter_gn_label = 'gn_scipy'

    if iter_hn_scipy > hybrid_newton_AELS.shape[0]:
        hybrid_newton = hybrid_newton_AELS
        print("hybrid newton iterations: aels")
        iter_hn_label = 'hn_aels'
    else:
        hybrid_newton = hn_scipy
        iter_hn_label = 'hn_scipy'

    # Plot iteration-error graph using the error array of the algorithm that performed better
    # wrt number of iterations

    max_iter_reached = max([len(gradient_descent), len(gradient_descent_exact), len(newton_armijo),
                            len(standard_newton), len(greedy_newton), len(hybrid_newton)])

    # Adapt arrays to fit the plot
    gradient_descent = adapt_array(gradient_descent, max_iter_reached)
    gradient_descent_exact = adapt_array(gradient_descent_exact, max_iter_reached)
    newton_armijo = adapt_array(newton_armijo, max_iter_reached)
    standard_newton = adapt_array(standard_newton, max_iter_reached)
    greedy_newton = adapt_array(greedy_newton, max_iter_reached)
    hybrid_newton = adapt_array(hybrid_newton, max_iter_reached)

    x = np.arange(max_iter_reached)

    # Plotting all values in the same figure
    fig, ax = plt.subplots()
    plt.plot(x, gradient_descent[:len(x)], label='gd_armijo')
    plt.plot(x, gradient_descent_exact[:len(x)], label=iter_gde_label)
    plt.plot(x, newton_armijo[:len(x)], label='nt_armijo')
    plt.plot(x, standard_newton[:len(x)], label='sn')
    plt.plot(x, greedy_newton[:len(x)], label=iter_gn_label)
    plt.plot(x, hybrid_newton[:len(x)], label=iter_hn_label)

    plt.xlabel('Iteration')
    plt.ylabel('f(xk) - f*')
    plt.legend()
    plt.grid(True)
    ax.set_yscale('log')
    plt.savefig("Plot/" + plot_name + "aels vs. scipy.pdf")

    # Check the exact search technique that has better performance in terms of time elapsed
    if time_gde_scipy[-1] > time_gdeaels[-1]:
        time_gde = time_gdeaels
        gradient_descent_exact = gradient_descent_exact_AELS
        print("gde time: aels")
        time_gde_label = 'gde_aels'
    else:
        time_gde = time_gde_scipy
        time_gde_label = 'gde_scipy'

    if time_gn_scipy[-1] > time_gneaels[-1]:
        time_gn = time_gneaels
        greedy_newton = greedy_newton_AELS
        print("greedy newton time: aels")
        time_gn_label = 'gn_aels'
    else:
        time_gn = time_gn_scipy
        time_gn_label = 'gn_scipy'

    if time_hn_scipy[-1] > time_hnaels[-1]:
        time_hn = time_hnaels
        hybrid_newton = hybrid_newton_AELS
        print("hybrid newton time: aels")
        time_hn_label = 'hn_aels'
    else:
        time_hn = time_hn_scipy
        time_hn_label = 'hn_scipy'

    time_gd = np.multiply(time_gd, 1000)
    time_gde = np.multiply(time_gde, 1000)
    time_na = np.multiply(time_na, 1000)
    time_sn = np.multiply(time_sn, 1000)
    time_gn = np.multiply(time_gn, 1000)
    time_hn = np.multiply(time_hn, 1000)

    fig, ax = plt.subplots()
    plt.plot(time_gd, gradient_descent[:len(time_gd)], label='gd_armijo', marker='.')
    plt.plot(time_gde, gradient_descent_exact[:len(time_gde)], label=time_gde_label, marker='.')
    plt.plot(time_na, newton_armijo[:len(time_na)], label='nt_armijo', marker='.')
    plt.plot(time_sn, standard_newton[:len(time_sn)], label='sn', marker='.')
    plt.plot(time_gn, greedy_newton[:len(time_gn)], label=time_gn_label, marker='.')
    plt.plot(time_hn, hybrid_newton[:len(time_hn)], label=time_hn_label, marker='.')

    plt.xlabel('time(ms)')
    plt.ylabel('f(xk) - f*')
    plt.legend()
    plt.grid(True)
    ax.set_yscale('symlog')
    if num_features == 2000:
        ax.set_xscale('symlog')
    plt.savefig("Plot/" + plot_name + ":aels vs. scipy time" + ".pdf")

# def check_equal_hessian(h1, h2):
#     for i in range(h1.shape[0]):
#         for j in range(h2.shape[0]):
#             if h1[i, j] != h2[i, j]:
#                 flag = False
#             else:
#                 flag = True
#
#     return flag
#
# def is_simmetric(h1):
#     for i in range(h1.shape[0]):
#         for j in range(h1.shape[1]):
#             if np.allclose(h1[i, j],h1[j, i]):
#                 flag = True
#             else:
#                 flag = False
#
#     return flag

if __name__ == '__main__':

    # Here we test the L2-regularized istances of log-loss minimizing problems.(set lambda_reg=1) in the
    # Dataset module

    # test_all_and_plot("p=20,regularized", 500, 20)
    # test_all_and_plot("p=20,regularized,repeated_features",500,20,True)
    # test_all_and_plot("p=200,regularized",500,200)
    # test_all_and_plot("p=2000,regularized",500, 2000)
    #test_compare("p=20,regularized", 500, 20)
    #test_compare("p=20,regularized,repeated_features",500,20,True)
    # test_compare("p=200,regularized",500,200)
    # test_compare("p=2000,regularized", 500, 2000)

    # To test the unregularized istances set the parameter lambda_reg = 0 in the Dataset module

    test_all_and_plot("p=20,unregularized", 500, 20)
    #test_all_and_plot("p=20,unregularized,repeated_features",500,20,True)
    #test_all_and_plot("p=200,unregularized",500,200)
    #test_all_and_plot("p=2000,unregularized",500, 2000)
    # test_compare("p=20,unregularized", 500, 20)
    # test_compare("p=20,unregularized,repeated_features",500,20,True)
    # test_compare("p=200,unregularized", 500, 200)
    # test_compare("p=2000,unregularized", 500, 2000)





    # Here we test different implementations of the hessian of the logistic function to test
    # which one is better in terms of numerical stability and time needed to complete the computation

    # initial_weights = np.zeros(21)
    # dataset, solver = init(500,20,initial_weights)
    # error_sn,time_sn = dataset.test_solver(initial_weights,solver.standard_newton)
    # error_sn_new_hess,time_sn_new_hess = dataset.test_solver(initial_weights,solver.standard_newton_new_hess)
    # print("time sn",time_sn)
    # print("time_sn_new_hess",time_sn_new_hess)

    # features = 2000
    # data = np.random.rand(500, features)
    # labels = np.random.choice([-1, 1], 500)
    #
    # dataset = Dataset(data, labels)
    # for i in range(100):
    #     w = np.random.rand(features)
    #     start = timer()
    #     h1 = dataset.hessian(w,1)
    #     end = timer()
    #     print("original hessian time:", end - start)
    #     start = timer()
    #     h2 = dataset.new_hessian(w,1)
    #     end = timer()
    #     print("new hessian time:", end - start)
    #     print("norm h1", la.norm(h1))
    #     print("norm h2", la.norm(h2))
    #     print("Equal Hessian:", check_equal_hessian(h1, h2))
    #     print("hessian norm difference", abs(la.norm(h1) - la.norm(h2)))



    #print(h1,h2)



