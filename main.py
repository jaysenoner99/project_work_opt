import numpy as np

from Dataset import Dataset
from solvers import Solver


if __name__ == '__main__':
    dataset = Dataset()
    solver = Solver()
    X, true_weights, labels = dataset.generate_dataset(500, 20)
    initial_weights = np.zeros(X.shape[1])
    #num_iter,solution = solver.gradientDescent(dataset,initial_weights)

    print("better gradient:",dataset.better_gradient(true_weights))
    print(" gradient:", dataset.gradient(true_weights))
    """""
    print("Number of iterations:",num_iter)
    print("Solution:",solution)
    print("Real optimal value",dataset.computeLogLoss(true_weights))
    print("Optimal value found:",dataset.computeLogLoss(solution))
    print("True weights:",true_weights)
    print("Percentage of good classifications:", dataset.test_prediction(200,true_weights,solution))
    """