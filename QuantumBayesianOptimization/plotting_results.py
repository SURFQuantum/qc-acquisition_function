from bayesian_optimization_quantum import QBO
from synth_quantum import synth_func

from sklearn.metrics import pairwise_distances
import numpy as np
import matplotlib.pyplot as plt

quantum_noise = False # Whether to consider quantum noise
linear_bandit = False # Use linear bandit instead of kernel approximation 

iterations = 10000 # Number of iterations 
N_repeat = 1 # Number of times to repeat each iterations

# Determine oracle function and domain
f_real = []
domain = np.linspace(0,1,4000)
def function(x):
    f = np.sin(5 * np.pi * x) + 0.3 * np.sin(20 * np.pi * x)
    return f

# Determine the f values of oracle function over the domain 
for i in range(len(domain)):
    f_real.append(function(domain[i]))

M = 100 # Number of random fourier features
d = 1 # Dimensions of the data

# Approximate kernel length-scale from domain spacing 
l = np.diff(domain) # Euclidean distance
sigma = np.median(l)
gamma = 1/(2*sigma**2)

# Sample random fourier features
random_features = {
    "s": np.random.randn(M, d)*np.sqrt(2*gamma), # Random frequencies
    "b": np.random.uniform(0, 2*np.pi, M), # Random phases
    "obs_noise": 0.01**2, # Noise of your sample points
    "v_kernel": 1.0
}

# Extract random fourier features parameters 
s = random_features['s']
b = random_features['b']
obs_noise = random_features['obs_noise']

# Beta parameter in UCB, determines exploration vs exploitation
ts = np.arange(1, iterations)
beta_t = 1 + np.sqrt(np.log(ts) ** 2)

# Create a list to store the results of each run  
res_list = [] 

# Determine the QBO using the chosen parameters
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

    # Save the results in the list
    res_list.append(res)

# Calculate the predicted f values, posterior mean 
y_pred = Phi_pred @ nu_t
y_pred = y_pred.flatten() #flatten to make the array 1D

# Calculate the uncertainty per value
y_var = np.sum(Phi_pred @ Sigma_t * Phi_pred, axis=1)
y_std = np.sqrt(y_var)

# Extract selected points
X_selected = X.flatten() 
Y_selected = res['all']['f_values']

# Sort X and Y values for plotting
X_sorted, Y_sorted = zip(*sorted(zip(X_selected, y_pred)))
X_sorted = list(X_sorted)
Y_sorted = list(Y_sorted)

# Plot the function approximation 
plt.figure(figsize=(12,6))
plt.plot(domain, f_real, 'b-', label='True function')
plt.plot(X_sorted, Y_sorted, 'r--', label='QBO prediction')
plt.scatter(X_selected, Y_selected, color='orange', label='Selected points', s=50, zorder=5, alpha=0.7)
plt.fill_between(X_sorted, Y_sorted - y_std, y_pred + y_std, color='gray', alpha=0.3, label='Prediction ±1σ')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.ylim([-3,3])
plt.xlim([0,1])
plt.grid()
plt.savefig(f'plots/QBO_prediction_with_{iterations}_iterations_and_{M}_fourrier_features_{obs_noise}_noise.png', dpi=120)
plt.show() 

# Compute the max f_value of the real function
f_max = np.max(f_real)

# Length of all the iterations you want to calculate
min_len = int(1e6)

# Create list to store the cumulative regret values 
all_regrets = []

# Loop over res_list / iterations
for itr in range(N_repeat):
    f_values = res_list[itr]["all"]["f_values"] # f values of measured points
    track_queries = res_list[itr]["all"]["track_queries"] # Number of oracle queries needed, times the oracle is called

    # Expand f_values according to the oracle query counts
    f_values_expanded = []
    for i in range(len(f_values)):
        # Create an array with the length of the amount of oracle queries needed per amount, [4,1] and [2,3] gives [[2,2,2,2],[3]]
        # This way you can represent the f_value of each query independently 
        f_values_expanded += list(np.repeat(f_values[i], track_queries[i]))
    f_values_expanded = np.array(f_values_expanded)

    # Calculate the regret by comparing it to the maximum f value
    f_values_regret = np.squeeze(f_max - f_values_expanded)

    # Cumulative regret
    f_values_cumregret = np.cumsum(f_values_regret)

    # Save the regrets 
    all_regrets.append(f_values_cumregret)

# Convert regrets to numpy array 
all_regrets_np = np.array(all_regrets)

# Compute mean and standard error
all_regrets_np_mean = np.mean(all_regrets_np, axis=0) # Calculate the mean of each column 
all_regrets_np_stderr = np.std(all_regrets_np, axis=0) / np.sqrt(len(all_regrets_np))

# Upper and lower bounds for plotting
all_regrets_np_ub = all_regrets_np_mean + all_regrets_np_stderr
all_regrets_np_lb = all_regrets_np_mean - all_regrets_np_stderr

# Plot cumulative regret function over iterations
inds = np.arange(0,len(all_regrets_np_mean))
plt.figure(figsize=(10,6))
plt.plot(inds[:min_len], all_regrets_np_mean,label='Predicted function')
plt.scatter(inds[:min_len],all_regrets_np_mean,marker='v', linewidth=3,label='Measured points')
plt.fill_between(inds[:min_len], all_regrets_np_ub, all_regrets_np_lb, alpha=0.2)
plt.xlabel('Iteration')
plt.ylabel('Cumulative Regret')
plt.title('Cumulative Regret of QBO')
plt.grid()
plt.legend()
plt.xlim([0,10000])
plt.savefig(f'plots/Regret_Func_Plot_{iterations}_and_{M}_fourrier_features_{obs_noise}_noise.png', dpi=120)
plt.show()

# Normalize the Phi_pred values
Phi_pred /= np.linalg.norm(Phi_pred, axis=1, keepdims=True)
K_rff_QBO = Phi_pred @ Phi_pred.T

# Compute exact Kernel  
domain = np.linspace(0,1,len(X_selected))
domain_vec = domain.reshape(-1, 1)
K_exact = np.exp(-pairwise_distances(domain_vec, metric='sqeuclidean') / (2 * sigma**2))

# Compute direct RFF approximation
Z = np.sqrt(2 / M) * np.cos(domain_vec @ s.T + b)
Z /= np.linalg.norm(Z, axis=1, keepdims=True)
K_rff_direct = Z @ Z.T

# Determine middle  slice of the 2D kernel matrix 
i = len(domain) // 2

# Plot 2D RBF kernel approximation, plotting the middle slice of the 3D kernel approximation 
plt.figure(figsize=(12,6))
plt.scatter(domain, K_exact[i, :], label='Exact RBF-kernel', linewidth=2)
plt.scatter(domain, K_rff_direct[i, :], alpha=0.5, label='RFF$_{direct}$-Approximation', linewidth=2)
plt.scatter(domain,K_rff_QBO[i,:],alpha = 0.5, label='RFF$_{QBO}$-Approximation', linewidth=2)
plt.xlabel('Domain (x)')
plt.ylabel('K(x, x$_i$)')
plt.legend()
plt.grid()
plt.savefig(f'plots/GP_Kernel_Approximation_with_{iterations}_Iterations_and_{M}_fourrier_features_{obs_noise}_noise.png', dpi=120)
plt.show()