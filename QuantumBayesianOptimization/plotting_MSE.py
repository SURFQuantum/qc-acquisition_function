from bayesian_optimization_quantum import QBO
from synth_quantum import synth_func

import numpy as np
import matplotlib.pyplot as plt

quantum_noise = False  # Whether to consider quantum noise
linear_bandit = False  # Use linear bandit instead of kernel approximation 

iterations = 10000 # Number of iterations 
N_repeat = 1 # Number of times to repeat each iterations

# Determining oracle function
domain = np.linspace(0, 1, 4000)
def function(x):
    return np.sin(5 * np.pi * x) + 0.3 * np.sin(20 * np.pi * x)
f_real = np.array([function(x) for x in domain])

# Approximate kernel length-scale from domain spacing 
l = np.diff(domain) # Euclidean distance
sigma = np.median(l)
gamma = 1/(2*sigma**2)

# Create an range of random fourier features to calculate the MSE over
M_begin = 100
M_end = 2900
M_timestep = 400
MSE_array = np.arange(M_begin, M_end+1, M_timestep)

# Dimension of data
d = 1

# List to save MSE values
MSE_list = []

# Loop over all number of random fourier features
for M in MSE_array:
    print(f"Running QBO with {M} Fourier features")

    # Sample random Fourier features
    random_features = {
        "s": np.random.randn(M, d) * np.sqrt(2*gamma), # Random frequencies
        "b": np.random.uniform(0, 2*np.pi, M), # Random phases
        "obs_noise": 0.01**2, # Noise of your sample points
        "v_kernel": 1.0
    }

    # Beta parameter in UCB, determines exploration vs exploitation
    ts = np.arange(1, iterations)
    beta_t = 1 + np.sqrt(np.log(ts) ** 2)

    # Create a list to store MSEs over repetitions 
    all_run_MSEs = []

    for itr in range(N_repeat):
        print(f"Run {itr} started!")
        np.random.seed(itr) # Random seed for the iteration 

        # Initialize and run QBO
        quantum_BO = QBO(
            f=synth_func,
            pbounds={'x1': (0,1)},
            beta_t=beta_t,
            random_features=random_features,
            linear_bandit=linear_bandit,
            domain=domain,
            f_real=f_real
        )

        # Extract the results of the QBO
        X, res, Phi_pred, features, nu_t, Sigma_t, _ = quantum_BO.maximize(
            n_iter=iterations,
            init_points=5
        )

        # Random fourier features 
        s = random_features['s']
        b = random_features['b']
        obs_noise = random_features['obs_noise']

        # Determine kernel approximation over the whole domain
        Phi_domain = np.sqrt(2/M) * np.cos(domain.reshape(-1,1) @ s.T + b)
        Phi_domain /= np.linalg.norm(Phi_domain, axis=1, keepdims=True)
        Phi_domain *= np.sqrt(random_features['v_kernel'])

        # Predict posterior mean
        y_pred_domain = Phi_domain @ nu_t
        y_pred_domain = y_pred_domain.flatten()

        # Compute MSE value of current run
        MSE = np.sum((f_real - y_pred_domain)**2)/len(f_real)
        all_run_MSEs.append(MSE)

    # Average MSE over repetitions 
    MSE_list.append(np.mean(all_run_MSEs))

# Plot MSE results
plt.figure(figsize=(10,6))
plt.plot(MSE_array, MSE_list, label='MSE Function', marker='v', linewidth=2)
plt.xlabel('Fourier Features')
plt.ylabel('MSE')
plt.title('QBO: MSE over Fourier features')
plt.grid(True)
plt.legend()
plt.xlim([M_begin, M_end])
plt.savefig(f'plots/MSE_Plot_{iterations}_and_{M}_fourrier_features_{obs_noise}_noise.png', dpi=120)
plt.show()
