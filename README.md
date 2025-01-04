### Goal
Option pricing is a fundamental topic in finance. In many cases, analytical formulas for pricing are unavailable, necessitating numerical methods to accurately estimate the asset's value. Monte Carlo techniques are widely used in this context, leveraging the law of large numbers and the property of asymptotic convergence to approximate the true price. However, a key drawback of Monte Carlo methods is the high variance of the estimates, which impacts the confidence in the results.

To address this limitation and obtain more accurate estimates with the same number of trials, a popular approach is importance sampling (IS). This variance reduction technique relies on altering the probability distribution under which the simulation is conducted, effectively reducing the confidence intervals of the estimate.

In this project, we aim to implement variance reduction techniques and identify an optimal change of measure for importance sampling. To achieve this, we will compare crude Monte Carlo (CMC) with IS-enhanced simulations while exploring two approaches for deriving the optimal change of measure. The first approach involves an adaptive method that iteratively adjusts the measure to improve accuracy, while the second focuses on parameterizing the measure as the solution to an optimal control problem. Through these methods, we aim to demonstrate the efficacy of importance sampling in reducing variance and improving pricing accuracy.

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

## Files 📁

### Main files
- [main.py](main.py) : Master file, desired processes are called from it. 

## Directories

### [Data](Data)
Contains the data used in the project in different formats(.txt, .csv).
Additionally some contains shorten dataset computed for shortened run times. The required data to run the code is available here https://drive.google.com/drive/folders/1ivoX5Kiannv-GN9mML8K0n7tHz3baE38?usp=share_link.

### [Utils](Utils)
Contains various utility files.
- [Grapher.py](Utils%2FGrapher.py) : Contains multiple functions to plot various graphs
- [Utilities.py](Utils%2FUtilities.py) : Functions with single use


## Usage 🫳
The code can be downloaded on the GitHub repository. Usage is of a standard Python code.

## Contact 📒
- Guillaume Ferrer: guillaume[dot]ferrer[at]epfl[dot]ch
- Agustina Maria Zein: agustina[dot]zein[at]epfl[dot]ch
