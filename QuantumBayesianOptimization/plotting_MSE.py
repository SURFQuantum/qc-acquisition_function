#%% Importing
from bayesian_optimization_quantum import QBO
from synth_quantum import synth_func

import numpy as np
import matplotlib.pyplot as plt

quantum_noise = False  # whether to consider quantum noise
linear_bandit = False  # whether to run quantum linear bandit algorithm

iterations = 10000
N_repeat = 1

#%% Determining function
domain = np.linspace(0, 1, 4000)
def function(x):
    return np.sin(5 * np.pi * x) + 0.3 * np.sin(20 * np.pi * x)

f_real = np.array([function(x) for x in domain])

# Euclidean distance
l = np.diff(domain)
sigma = np.median(l)
gamma = 1/(2*sigma**2)
print('sigma:', sigma)

#%% Random Features / MSE over Fourier features
M_begin = 100
M_end = 2900
M_timestep = 400
d = 1  # dimension of data

MSE_list = []

MSE_array = np.arange(M_begin, M_end+1, M_timestep)
for M in MSE_array:
    print(f"Running QBO with {M} Fourier features")

    # Random Fourier features
    random_features = {
        "s": np.random.randn(M, d) * np.sqrt(2*gamma),
        "b": np.random.uniform(0, 2*np.pi, M),
        "obs_noise": 0.01**2,
        "v_kernel": 1.0
    }

    # Beta_t for uncertainty
    ts = np.arange(1, iterations)
    beta_t = 1 + np.sqrt(np.log(ts) ** 2)

    all_run_MSEs = []

    for itr in range(N_repeat):
        print(f"Run {itr} started!")
        np.random.seed(itr)

        # Run QBO
        quantum_BO = QBO(
            f=synth_func,
            pbounds={'x1': (0,1)},
            beta_t=beta_t,
            random_features=random_features,
            linear_bandit=linear_bandit,
            domain=domain,
            f_real=f_real
        )

        X, res, Phi_pred, features, nu_t, Sigma_t, _ = quantum_BO.maximize(
            n_iter=iterations,
            init_points=5
        )

        #%% Calculate MSE over full domain
        s = random_features['s']
        b = random_features['b']
        obs_noise = random_features['obs_noise']

        # Determine features over the whole domain
        Phi_domain = np.sqrt(2/M) * np.cos(domain.reshape(-1,1) @ s.T + b)
        Phi_domain /= np.linalg.norm(Phi_domain, axis=1, keepdims=True)
        Phi_domain *= np.sqrt(random_features['v_kernel'])

        y_pred_domain = Phi_domain @ nu_t
        y_pred_domain = y_pred_domain.flatten()

        #Caclulate MSE value for each run
        MSE = np.sum((f_real - y_pred_domain)**2)/len(f_real)
        all_run_MSEs.append(MSE)

    # calculate mean per repeats
    MSE_list.append(np.mean(all_run_MSEs))

#%% Plot results
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
