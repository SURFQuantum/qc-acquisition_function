#%% Importing
from bayesian_optimization_quantum import QBO
from synth_quantum import synth_func

from sklearn.metrics import pairwise_distances
import numpy as np
import matplotlib.pyplot as plt

quantum_noise = False # whether to consider quantum noise
linear_bandit = False # whether to run quantum linear bandit algorithm; set it to False by default

iterations = 10000
N_repeat = 1

#%% Determining function
f_real = []
domain = np.linspace(0,1,4000)
def function(x):
    f = np.sin(5 * np.pi * x) + 0.3 * np.sin(20 * np.pi * x)
    return f

for i in range(len(domain)):
    f_real.append(function(domain[i]))

M = 100 #amount of fourrier features
d = 1 #dimensions of the data

# euclidean distance
l = []
for i in range(len(domain)-1):
    l.append(np.sqrt(np.sum((domain[i] - domain[i+1])**2)))

#%% Random Features
# determine the sigma of the GP kernel
sigma = np.median(l)
gamma = 1/(2*sigma**2)
print('sigma: ',sigma)

#random fourrier features to approximate the kernel.
random_features = {
    "s": np.random.randn(M, d)*np.sqrt(2*gamma), #random frequencies
    "b": np.random.uniform(0, 2*np.pi, M), #random phases
    "obs_noise": 0.01**2, #noise of your sample points
    "v_kernel": 1.0
}

s = random_features['s']
b = random_features['b']
obs_noise = random_features['obs_noise']

#%% Computing Results
ts = np.arange(1, iterations) #determine all the point of all the iterations
beta_t = 1 + np.sqrt(np.log(ts) ** 2) #determine beta_t for the uncertainty calculation

res_list = []

#determine the QBO of all data
for itr in range(N_repeat):
    print(f"Run {itr} started!")
    np.random.seed(itr)

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

    res_list.append(res)

#%% Plotting predicted vs real function
# Calculate the predicted f values
y_pred = Phi_pred @ nu_t
y_pred = y_pred.flatten() #flatten to make the array 1D

# Calculate the uncertainty per value
y_var = np.sum(Phi_pred @ Sigma_t * Phi_pred, axis=1)
y_std = np.sqrt(y_var)

# Determine 1D arrays for plotting
X_selected = X.flatten() 
Y_selected = res['all']['f_values']

# Sort X and Y values for convenience 
X_sorted, Y_sorted = zip(*sorted(zip(X_selected, y_pred)))
X_sorted = list(X_sorted)
Y_sorted = list(Y_sorted)

# Plot results 
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

#%% Plotting Regret function
# Caculating the max f_value of the real function
f_max = np.max(f_real)

# length of all the iterations you want to calculate
min_len = int(1e6)

all_regrets = []

# Loop over res_list / iterations
for itr in range(N_repeat):
    f_values = res_list[itr]["all"]["f_values"] #f value of measured points
    track_queries = res_list[itr]["all"]["track_queries"] #number of oracle queries needed, oracle is called

    # Repeat f_values according to queries and calculate regret
    f_values_expanded = []
    for i in range(len(f_values)):
        # Create an array with the length of the amount of oracle queries needed per amount, [4,1] and [2,3] gives [[2,2,2,2],[3]]
        # This way you can represent the f_value of each query independently 
        f_values_expanded += list(np.repeat(f_values[i], track_queries[i]))
    f_values_expanded = np.array(f_values_expanded)

    # calculate the regret the way the paper said
    f_values_regret = np.squeeze(f_max - f_values_expanded)  # calculate the f_max - f(x)

    # Cumulative regret
    f_values_cumregret = np.cumsum(f_values_regret) #cumulative sum of f_values

    all_regrets.append(f_values_cumregret)


all_regrets_np = np.array(all_regrets)

# Compute mean and standard error
all_regrets_np_mean = np.mean(all_regrets_np, axis=0) #take the first point of each array and calculate the mean, then continue to the second point of each array

all_regrets_np_stderr = np.std(all_regrets_np, axis=0) / np.sqrt(len(all_regrets_np))

# Upper and lower bounds for plotting
all_regrets_np_ub = all_regrets_np_mean + all_regrets_np_stderr
all_regrets_np_lb = all_regrets_np_mean - all_regrets_np_stderr

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

#%% Plotting RBF kernel approximation
# Normalize the Phi_pred values
domain = np.linspace(0,1,len(X_selected))
domain_vec = domain.reshape(-1, 1)

# Calculate RFF approximation
Phi_pred /= np.linalg.norm(Phi_pred, axis=1, keepdims=True)
K_rff_QBO = Phi_pred @ Phi_pred.T

#Calculating exact Kernell    
K_exact = np.exp(-pairwise_distances(domain_vec, metric='sqeuclidean') / (2 * sigma**2))

Z = np.sqrt(2 / M) * np.cos(domain_vec @ s.T + b)
Z /= np.linalg.norm(Z, axis=1, keepdims=True)

# Calculate RFF approximation
K_rff_direct = Z @ Z.T

# Determine middle of the 2D approximated kernel, the see the difference
i = len(domain) // 2

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