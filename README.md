# Optimization Techniques for Machine Learning Project Work

Source code for the project assignment associated with the course "Optimization Techniques for Machine Learning" instructed by Professor Matteo Lapucci for the second cycle degree in Artificial Intelligence at the University of Florence.

The primary focus of this project is to compare various implementations of Newton methods, differing mainly in their approach to selecting the step size at each iteration.

Six different algorithms are compared and tested in this project:

- Gradient descent with Armijo line search
- Gradient descent with Greedy (exact) line search
- Standard Newton's Method (α = 1)
- Newton's Method with Armijo line search
- Newton's Method with Exact line search (also known as Greedy Newton from [1])
- Hybrid Newton's method [1]

These algorithms undergo testing for a logistic regression problem, wherein there are $m$ training examples $(x_i, y_i)$ with dense features and binary labels . The objective function to minimize is defined as:

$$f(w) = \sum_{i=1}^m \log(1 + \exp(-y_i w^T x_i))$$

The tests include both L2-regularized and unregularized instances. The dataset used for testing is synthetic, generated by a Data Generating Process as specified in section 3 of [1].

For the cases where the algorithm employs an exact line search technique, two different techniques were tested:

- minimize_scalar: A function implementing the Brent method for exact line search (imported from scipy)
- AELS (Approximately Exact Line Search): An "exact" line search technique described in [2]

For further technical details on what has been done on this project, please refer to the PWOPT pdf file
## References

[1] Shea, B., & Schmidt, M. (2024). "Greedy Newton: Newton's Method with Exact Line Search." arXiv preprint. Retrieved from [https://arxiv.org/abs/2401.06809](https://arxiv.org/abs/2401.06809)

[2] Fridovich-Keil, S., & Recht, B. (2022). "Approximately Exact Line Search." arXiv preprint. Retrieved from [https://arxiv.org/abs/2011.04721](https://arxiv.org/abs/2011.04721)
