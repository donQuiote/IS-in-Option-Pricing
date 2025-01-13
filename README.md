### Goal
Option pricing is a fundamental topic in finance. In many cases, analytical formulas for pricing are unavailable, necessitating numerical methods to accurately estimate the asset's value. Monte Carlo techniques are widely used in this context, leveraging the law of large numbers and the property of asymptotic convergence to approximate the true price. However, a key drawback of Monte Carlo methods is the high variance of the estimates, which impacts the confidence in the results.

To address this limitation and obtain more accurate estimates with the same number of trials, a popular approach is importance sampling (IS). This variance reduction technique relies on altering the probability distribution under which the simulation is conducted, effectively reducing the confidence intervals of the estimate.

In this project, we aim to compare crude Monte Carlo (CMC) with IS-enhanced simulations. The first approach involves an adaptive method that iteratively adjusts the measure to improve accuracy, while the second focuses on parameterizing the measure as the solution to an optimal control problem. Through these methods, our objective is to demonstrate the efficacy of importance sampling in reducing variance and improving price accuracy.

## Installation 💻
The code is optimized for Python 3.11.

### Library
The following library are used:
- Numpy
- Polars
- Matplotlib
- plotly
- Scikit-Learn
- Pandas
- Scipy
- tqdm

## Main Directories
### [European Call Option](European%20Call%20Option)
This directory contains the files to run European Call option pricing.
- [Call_option.py](European%20Call%20Option/Call_Option.py) : Runs the Crude Monte Carlo, Importance sampling algorithm and the adaptive search for optimal change of measure
- [Optimal control.py](European%20Call%20Option/Optimal%20control.py) : Runs the crank-nicholson method to solve for the optimal control of the stocks trajectories

### [Up-and-out Call Option](Up-and-out%20Call%20Option)
This directory contains the files to run Up-and-Out Call Option pricing.
- [Up_and_out_call.py](Up-and-out%20Call%20Option/Up_and_out_call.py) : Runs the Crude Monte Carlo and the adaptive search for optimal change of measure
- [Optimal_Up_and_Out_Call.py](Up-and-out%20Call%20Option/Optimal_Up_and_Out_Call.py) : Runs the crank-nicholson method to solve for the optimal control of the stocks trajectories with barrier

### [Algorithms](Algorithms)
Contains the main algorithm to generate path under a change a measure or not. Run monte carlo process, search for the optimal change of measure or apply the backward induction crank-nicholson algorithm
- [generation.py](Algorithms/generation.py) : Generate stocks paths, simulations, Monte Carlo, Confidence intervals and controlled paths
- [adaptive_strategy.py](Algorithms/adaptive_strategy.py) : Computes the optimal change of measure via minimizing the estimator variance
- [crank_nicholson.py](Algorithms/crank_nicholson.py) : Solves a stochastic differential equation (SDE) backward in time using the Crank-Nicholson method.

### [Graphs](Graphs)
Contains multiple graphs created throughout the project

### [Utils](Utils)
Contains various utility files.
- [grapher.py](Utils/grapher.py) : Contains multiple functions to plot various graphs
- [helpers.py](Utils/helpers.py) : Functions with single use, analytical price or latex table creation


## Usage 🫳
The code can be downloaded on the GitHub repository. Usage is of a standard Python code.

## Contact 📒
- Guillaume Ferrer: guillaume[dot]ferrer[at]epfl[dot]ch
- Agustina Maria Zein: agustina[dot]zein[at]epfl[dot]ch
