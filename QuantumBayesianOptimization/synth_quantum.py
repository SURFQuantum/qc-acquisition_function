from qiskit import QuantumCircuit
from qiskit.algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit.circuit.library import LinearAmplitudeFunction
from qiskit_aer.primitives import Sampler
from qiskit_finance.circuit.library import NormalDistribution

import numpy as np

def synth_func(param, eps, domain, f_real, random_features):
    """
    Quantum Monte Carlo oracle for noisy function evaluation using IterativeAmplitudeEstimation from Qiskit. 

    
    This function simulates a quantum oracle that evaluates a noisy version of a classical objective function. Instead of returning a
    single deterministic value, the function models the objective function as a Gaussian random variable with a known observation noise 
    and estimates its expectation value using quantum amplitude estimation.

    The uncertainty in the function value is encoded as a truncated normal distribution in a quantum state. A linear payoff function
    is then applied so that the expected value of the distribution is encoded to the amplitude of a designated objective qubit.
    IterativeAmplitudeEstimation is used to estimate this amplitude with target precision eps, achieving a quadratic speedup compared
    to the classical Monte Carlo approach.

    This function returns both the estimated value and the number of oracle queries required to achieve desired precision, which allows
    Bayesian Optimization to account for variable evaluation costs.

    Notes:
    ------
    The true function values f_real are only used for simulation and benchmarking; they are not available to the optimizer
    The Gaussian distribution is truncated to ±3σ to obtain a finite support suitable for quantum circuit   

    

    Parameters:
    -----------
    param: query point
    eps: target precision for amplitude estimation 
    domain: domain of oracle function 
    f_real: true f values evaluated of oracle function 
    random_features: random fourier features 

    Returns:
    --------
    est: estimated expectation value (oracle output)
    mean: true f value of closest domain point
    num_oracle_queries: number of oracle queries/calls used 
    """

    x = param

    # Find which domain point is the closest to x
    ind = np.argmin(np.abs(domain - x))

    # Number of uncertainty qubits used for uncertainty encoding 
    num_uncertainty_qubits = 6 

    # Read the value of f at the closest point
    mean = f_real[ind]
    
    # Determine standard deviation of the variance, Gaussian kernel
    variance = random_features['obs_noise']
    stddev = np.sqrt(variance)

    # Compute the uncertainty bounds using ±3σ
    low = mean - 3 * stddev
    high = mean + 3 * stddev

    # Compute the uncertainty model: truncated normal distribution  
    uncertainty_model = NormalDistribution(num_uncertainty_qubits, mu=mean, sigma=stddev**2, bounds=(low, high))    
    
    # Determine parameters for linear payoff function 
    c_approx = 1
    slopes = 1
    offsets = 0
    f_min = low # Global minimum of the NormalDistribution
    f_max = high # Global maxima of the NormalDistribution

    # The LinearAmplitudeFunction is a piecewise linear function
    linear_payoff = LinearAmplitudeFunction(
        num_uncertainty_qubits,
        slopes,
        offsets,
        domain=(low, high),
        image=(f_min, f_max),
        rescaling_factor=c_approx,
    )

    # Compose uncertainty model and payoff function into a quantum circuit 
    num_qubits = linear_payoff.num_qubits
    monte_carlo = QuantumCircuit(num_qubits)
    monte_carlo.append(uncertainty_model, range(num_uncertainty_qubits))
    monte_carlo.append(linear_payoff, range(num_qubits))

    # Set target precision for amplitude estimation 
    epsilon = eps / (3 * stddev)
    epsilon = np.clip(epsilon, 1e-6, 0.5)

    # List of qubits used for encoding the payoff
    objective_qubits = [0]

    # Determine parameters
    seed = 0 # Simulation seed 
    alpha = 0.05 # Confidence level
    max_shots = 32 * np.log(2/alpha*np.log2(np.pi/(4*epsilon))) # Shot budget 

    # Construct estimation problem. post_processing is the inverse of the rescaling, i.e., it maps the [0, 1] interval to the 
    # original one. Objective_qubits is the list of qubits that are used to encode the objective function. Problem is the 
    # estimation problem that is passed to the QAE algorithm.
    problem = EstimationProblem(state_preparation=monte_carlo, objective_qubits=objective_qubits, post_processing=linear_payoff.post_processing, )
    
    # Construct iterative amplitude estimation, grover like
    ae = IterativeAmplitudeEstimation(epsilon_target=epsilon, alpha=alpha, sampler=Sampler(run_options={"shots": int(np.ceil(max_shots)),"seed_simulator":seed}))

    # Running estimation 
    result = ae.estimate(problem)
    est = result.estimation_processed
    
    # Number of times the oracle is called
    num_oracle_queries = result.num_oracle_queries

    # Fallback oracle query count if not reported
    if num_oracle_queries == 0:
        num_oracle_queries = int(np.ceil((0.8 / epsilon) * np.log((2 / alpha) * np.log2(np.pi / (4 * epsilon)))))

    return est, mean, num_oracle_queries