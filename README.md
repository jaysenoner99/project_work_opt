# Project Work Optimization Techniques for Machine Learning 

Soruce code for the project work assignment related to the course "Optimization techniques for machine learning" held by prof. Matteo Lapucci. 
The main topic of this project work is related to the comparison of different implementations of Newton methods, where the difference between those variants is in the way the step size at each Newton iteration is selected.
In this project we compare and test 6 different algorithms:
<ul>
  <li>Gradient descent with Armijo line search;</li>
  <li>Gradient descent with Greedy (exact) line search</li>
  <li>Standard Newton's Method (α = 1);</li>
  <li>Newton's Method with Armijo line search</li>
  <li>Newton's Method with Exact line search</li>
  <li>Hybrid Newton's method </li>
</ul>

All those algorithms are tested on a log-loss minimizing problem, considering both L2-regularized and unregularized istances. The dataset for the testing process is given by a Data generating process, as specified in section 3 of [1]. For the cases where the 
algorithm uses an exact line search technique, two different techniques were tested: 
<ul>
  <li>minimize_scalar : A function that implements the Brent method for exact line search(imported from scipy)</li>
  <li>AELS(Approximately exact line search) : An "exact" line search technique taken from [2] </li>
</ul>



<h1>References</h1>

[1]   Shea, B., & Schmidt, M. (2024). “Greedy Newton: Newton’s Method with Exact Line Search.” arXiv preprint. Retrieved from <a href="https://arxiv.org/abs/2401.06809">https://arxiv.org/abs/2401.06809</a>

[2]   Fridovich-Keil, S., & Recht, B. (2022). “Approximately Exact Line Search.” arXiv preprint. Retrieved from <a href="https://arxiv.org/abs/2011.04721">https://arxiv.org/abs/2011.04721</a>
