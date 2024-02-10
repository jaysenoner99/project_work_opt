import numpy as np

from Dataset import Dataset
from solvers import Solver


if __name__ == '__main__':
    dataset = Dataset()
    solver = Solver()
    X, weights, labels = dataset.generateData(5000,20)

    num_iter,solution = solver.gradientDescent(dataset,weights)
    print("gradient norm at the solution point:",dataset.gradient(solution))
    print("Number of iterations:",num_iter)
    print("Solution:",solution)
    print("Percentage of good classifications:",dataset.test(solution))