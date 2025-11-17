import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel

M = 1000 #amount of fourrier features
d = 1 #dimensions of the data

#%% Random Features
# determine the sigma of the GP kernel
sigma = 0.3 #a too large sigma gives a bad RFF approximation
gamma = 1/(2*sigma**2)

#random fourrier features to approximate the kernel.
random_features = {
    "s": np.random.randn(M, d)*np.sqrt(2*gamma), #random frequencies
    "b": np.random.uniform(0, 2*np.pi, M), #random phases
    "obs_noise": 0.01**2, #noise of your sample points
    "v_kernel": 1.0 #doesn't get used
}

s = random_features['s']
b = random_features['b']

x = np.linspace(-1,1,100).reshape(-1,1)

Z = np.sqrt(2/M)*np.cos(x @ s.T +b)
K_rff = Z @ Z.T

K_true = rbf_kernel(x,x,gamma=gamma)

plt.figure(figsize=(12,6))
plt.imshow(K_rff)
plt.colorbar(label='K$_{rff}$')
plt.title(f'RBF Kernell Approximation_{M}_features')
plt.savefig(f'plots/RBF_Kernell_Approximation_{M}_fourier_features.png', dpi=120)
plt.show()

plt.figure(figsize=(12,6))
plt.imshow(K_true)
plt.colorbar(label='K$_{true}$')
plt.title('True RBF Kernell')
plt.savefig('plots/True_RBF_Kernell.png', dpi=120)
plt.show()

i = len(x)//2
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
