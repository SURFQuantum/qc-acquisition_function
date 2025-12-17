import numpy as np
from datetime import datetime
from scipy.optimize import minimize
import pickle

def acq_max(ac, M, random_features, bounds, nu_t, Sigma_t_inv, beta, domain, linear_bandit):
    """
    Maximize the acquisition function over given domain 
    
    Parameters:
    -----------
    ac: acquisition function, UCB
    M: number of random fourier features
    random_features: random fourier features
    bounds: search space bounds
    nu_t: mean of regression weights
    Sigma_t_inv: inverse posterior covariance 
    beta: beta parameter in UCB, determines exploration vs exploitation
    domain: domain of oracle function 
    linear_bandit: use linear bandit instead of kernel approximation, True or False

    Returns:
    --------
    float(x_max): point in the domain where the acquisition function was maximum 
    """
    
    para_dict={"M":M, "random_features":random_features, "bounds":bounds, "nu_t":nu_t, "Sigma_t_inv":Sigma_t_inv, \
              "beta":beta, "linear_bandit":linear_bandit}
    ys = []
    
    # Evaluate acquisition function over domain 
    for i, x in enumerate(domain):
        ys.append(-ac(np.array([x]), para_dict))
    
    # Determine the new measured point by determining the argmin    
    ys = np.squeeze(np.array(ys))
    argmin_ind = np.argmin(ys)
    x_max = domain[argmin_ind]
    return float(x_max)
    
class UtilityFunction(object):
    def __init__(self):
        """
        The UtilityFunction (acquisition function) function wrapper
        """
        self.kind = "ucb"

    def utility(self, x, para_dict):
        """
        Unpack the needed parameters for the _ucb function
        
        Parameters:
        -----------
        x: x-coordinate of now predicted function 
        para_dict: dictionary with essential parameters
        """

        # Earlier mentioned parameters
        M = para_dict["M"]
        random_features = para_dict["random_features"]
        nu_t = para_dict["nu_t"]
        Sigma_t_inv = para_dict["Sigma_t_inv"]
        beta = para_dict["beta"]
        linear_bandit = para_dict["linear_bandit"]
        
        # Calculate the uncertainty if needed 
        if self.kind == "ucb":
            return self._ucb(x, random_features, nu_t, Sigma_t_inv, beta, linear_bandit)

    @staticmethod
    def _ucb(x, random_features, nu_t, Sigma_t_inv, beta, linear_bandit):
        """
        Upper Confidence Bound acquisition function   
        
        Parameters:
        -----------
        x: x-coordinates of now predicted function 
        random_features: random fourier features
        nu_t: mean of regression weights
        Sigma_t_inv: inverse posterior covariance   
        beta: beta parameter in UCB, determines exploration vs exploitation
        linear_bandit: use linear bandit instead of kernel approximation, True or False

        Returns:
        --------
        UCB: UCB acquisition function value
        """

        # If the linear bandit is not wanted, construct the features   
        if not linear_bandit:
            s = random_features["s"]
            b = random_features["b"]
            v_kernel = random_features["v_kernel"]
            M = b.shape[0]

            x = np.squeeze(x).reshape(1, -1)
            features = np.sqrt(2 / M) * np.cos(np.dot(x, s.T) + b)
            features = features.reshape(-1, 1)
            features = features / np.linalg.norm(features)
            features = np.sqrt(v_kernel) * features
        else: # Only reshape if linear bandit is used
            features = x.reshape(-1, 1)

        # Determine the predictive mean 
        mean = np.squeeze(np.dot(features.T, nu_t))

        # Observation noise 
        lam = 1

        # Compute predictive variance and standard deviation 
        var = lam * np.squeeze(np.dot(features.T, np.dot(Sigma_t_inv, features)))
        std = np.sqrt(var)
        
        # Compute the UCB acquisition function value
        UCB = mean + beta * std
        return UCB


