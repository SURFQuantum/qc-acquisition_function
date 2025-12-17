import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel

M = 1000 # Number of fourrier features
d = 1 # Dimensions of the data

# Determine the sigma, scale parameter, of the GP kernel
sigma = 0.3 # A too large sigma gives a bad RFF approximation
gamma = 1/(2*sigma**2)

# Random fourrier features to approximate the kernel.
random_features = {
    "s": np.random.randn(M, d)*np.sqrt(2*gamma), # Random frequencies
    "b": np.random.uniform(0, 2*np.pi, M), # Random phases
    "obs_noise": 0.01**2, # Noise of your sample points
    "v_kernel": 1.0 # Doesn't get used
}

# Extract the random fourrier features needed for the kernel approximation 
s = random_features['s']
b = random_features['b']

x = np.linspace(-1,1,100).reshape(-1,1) # Create a column for the x-axis

# Approximate the gaussian kernel using the fourrier features
Z = np.sqrt(2/M)*np.cos(x @ s.T + b) 
K_rff = Z @ Z.T

# Compute the true rbf kernel
K_true = rbf_kernel(x,x,gamma=gamma)

# Plot the 3D kernel approximation
plt.figure(figsize=(12,6))
plt.imshow(K_rff)
plt.colorbar(label='K$_{rff}$')
plt.title(f'RBF Kernell Approximation_{M}_features')
plt.savefig(f'plots/RBF_Kernell_Approximation_{M}_fourier_features.png', dpi=120)
plt.show()

# Plot the true 3D kernel
plt.figure(figsize=(12,6))
plt.imshow(K_true)
plt.colorbar(label='K$_{true}$')
plt.title('True RBF Kernell')
plt.savefig('plots/True_RBF_Kernell.png', dpi=120)
plt.show()

# Plot the true and approximated 2D kernel by taking the middle of the 3D kernels
i = len(x)//2 # Calculate where the middle is 
plt.figure(figsize=(12, 6))
plt.scatter(x, K_true[:, i], label="True kernel")
plt.scatter(x, K_rff[:, i], label="RFF approximation")
plt.xlabel("x")
plt.ylabel("k(x, x$_{o}$)")
plt.legend()
plt.grid()
plt.title(f'RBF Kernell Approximation 1D {M} fourier features')
plt.savefig(f'plots/RBF_Kernell_Approximation_1D_{M}_fourier_features.png', dpi=120)
plt.show()
