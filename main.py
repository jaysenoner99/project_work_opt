
from Dataset import Dataset
from solvers import Solver


if __name__ == '__main__':
    dataset = Dataset()
    solver = Solver()
    X, weights, labels = dataset.generateData(500,20)
    print("Dataset: \n")
    print(X)
    print("Weights: \n")
    print(weights)
    print("Labels: \n")
    print(labels)
    print(dataset.computeLogLoss(weights,1))
    print("gradient:")
    print(dataset.gradient(weights))
    print("hessian:")
    print(dataset.hessian(weights))
    print(solver.standardNewton(dataset, 1000, weights, 0.001, 0.001))