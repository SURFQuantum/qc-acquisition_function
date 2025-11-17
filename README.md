# QC-AcquisitionFunction
## Introduction
This repository contains the code for the quantum acquisition function used in quantum Bayesian optimisation (QBO). This is modified code from the following paper: Quantum Bayesian Optimization (https://arxiv.org/abs/2310.05373). The code can be run, but further research is needed. Suggested topics can also be found in the repository.

## Building a suited environment
Follow the steps below to build a virtual environment using Conda and then install the right packages using Pip:

1. ***Install Python***: Make sure you have Python 3.11 installed, as this was the version used during the code's modification, though other versions might also work.
2. ***Install Conda***: Make sure you have Conda installed to create an environment. The version does not matter.
3. ***Create environment***: Run the following command to create and activate an environment. Make sure you replace '[environment_name]' with the name you want to use.
```bash
conda create -n [environment_name]
conda activate [environment_name]
```
4. ***Installing required packages***: Run the following code to install the necessary packages:
```bash
pip install -r requirements.txt
```

## Project Structure
qc-acquisition_function/ 
- README.md 
- requirements.txt
- QuantumBayesianOptimization/ # The quantum bayesian optimization algorithm
- RBFKernelApproximation/ # The approximation of the kernel used, 1D and 2D

## Explanation algorithm

This repository's Quantum Bayesian Optimization script combines classical Bayesian Optimization with quantum techniques to accelerate the search for the maximum of an unknown function. In this approach, the Gaussian process model typically used in classical Bayesian Optimization is approximated using random Fourier features. This provides an efficient representation of kernels, such as the RBF kernel, which are specifically used for finding the maximum. These features are then used in a quantum Monte Carlo routine that uses amplitude amplification, similar to Grover-like methods, to estimate the acquisition function more efficiently than classical sampling can.

Using quantum concepts to speed up evaluation of the acquisition function enables the algorithm to identify promising query points with fewer evaluations. Theoretical regret analysis shows that this framework can achieve polylogarithmic cumulative regret, which represents a significant improvement on the $\sqrt{T}$ scaling observed in classical Bayesian Optimization. Results from simulations, demonstrate how the quantum-enhanced method can converge on the optimum more quickly. However, performance still depends on factors such as the dimensionality of the feature map.

Once the maximum has been identified, the quality of the overall function approximation depends on the accuracy of the underlying kernel approximation. Improving this approximation requires increasing the number of Fourier features to yield a more accurate kernel representation. However, doing so affects the regret behaviour; rather than achieving a capped regret, the regret becomes linear. As a result, classical Bayesian Optimization remains superior to the current quantum implementation for general function approximation rather than pure optimization of the maximum.

Plots illustrating the kernel approximation, regret behaviour, function approximation and mean squared error as a function of the number of Fourier features are provided in the repository. These figures offer a visual explanation of the algorithm and its performance.

## Future research topics

Future research topics:
- Evaluate the regret equation.

    The cumulative regret equation used in the repository is derived from the standard definition of bandit-based regret, which measures the difference between the maximum achievable reward and the reward obtained. In the referenced paper, this 'maximum reward' corresponds to the maximum value of the function. However, in the context of function approximation, it is not necessarily the case that the 'best reward' corresponds to the global maximum; accurately estimating the true function values at the sampled points may be more relevant. For this reason, this definition of regret may not be fully appropriate in the context of function approximation.

- Check whether the algorithm can locate the global minimum.

    Since the algorithm can efficiently find the global maximum, it should also be able to identify the global minimum.

- Experiment further with the number of iterations and Fourier features.

    Test different iterations and Fourier feature amounts to determine the most accurate estimates. 