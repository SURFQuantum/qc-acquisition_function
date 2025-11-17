from qiskit import QuantumCircuit
from qiskit.algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit.circuit.library import LinearAmplitudeFunction
from qiskit_aer.primitives import Sampler
from qiskit_finance.circuit.library import NormalDistribution

import numpy as np

def synth_func(param, eps, domain, f_real, random_features):
    
    x = param
    #detect which domain point is the closest to x
    ind = np.argmin(np.abs(domain - x))

    num_uncertainty_qubits = 6

    #read the value of f on the closest point
    mean = f_real[ind]
    
    #determine uncertainty
    variance = random_features['obs_noise']
    stddev = np.sqrt(variance)

    low = mean - 3 * stddev
    high = mean + 3 * stddev

    uncertainty_model = NormalDistribution(num_uncertainty_qubits, mu=mean, sigma=stddev**2, bounds=(low, high))    
    
    c_approx = 1
    slopes = 1
    offsets = 0
    f_min = low #global minimum of the NormalDistribution
    f_max = high #global maxima of the NormalDistribution

    # The LinearAmplitudeFunction is a piecewise linear function
    linear_payoff = LinearAmplitudeFunction(
        num_uncertainty_qubits,
        slopes,
        offsets,
        domain=(low, high),
        image=(f_min, f_max),
        rescaling_factor=c_approx,
    )

    # construct A operator for QAE for the payoff function by
    # composing the uncertainty model and the objective
    num_qubits = linear_payoff.num_qubits
    monte_carlo = QuantumCircuit(num_qubits)
    monte_carlo.append(uncertainty_model, range(num_uncertainty_qubits))
    monte_carlo.append(linear_payoff, range(num_qubits))

    # set target precision and confidence level
    epsilon = eps / (3 * stddev)

    objective_qubits = [0]
    seed = 0 #???

    epsilon = np.clip(epsilon, 1e-6, 0.5)

    alpha = 0.05
    max_shots = 32 * np.log(2/alpha*np.log2(np.pi/(4*epsilon))) 

    # construct estimation problem. post_processing is the inverse of the rescaling, i.e., it maps the [0, 1] interval to the original one.
    # objective_qubits is the list of qubits that are used to encode the objective function.
    # problem is the estimation problem that is passed to the QAE algorithm.
    problem = EstimationProblem(state_preparation=monte_carlo, objective_qubits=objective_qubits, post_processing=linear_payoff.post_processing, )
    
    # construct amplitude estimation, grover basically
    ae = IterativeAmplitudeEstimation(epsilon_target=epsilon, alpha=alpha, sampler=Sampler(run_options={"shots": int(np.ceil(max_shots)),"seed_simulator":seed}))

    # Running result
    result = ae.estimate(problem)
    est = result.estimation_processed
    
    # number of times the oracle is called
    num_oracle_queries = result.num_oracle_queries

    if num_oracle_queries == 0:
        # use the number of oracle calls given by the paper if num_oracle_queries == 0
        num_oracle_queries = int(np.ceil((0.8 / epsilon) * np.log((2 / alpha) * np.log2(np.pi / (4 * epsilon)))))

    return est, mean, num_oracle_queries