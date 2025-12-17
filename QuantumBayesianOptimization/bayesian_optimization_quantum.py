import numpy as np
from helper_funcs_quantum import UtilityFunction, acq_max
import pickle
import itertools
import time

class QBO(object):
    def __init__(self, f, pbounds, \
                      
                 beta_t=None, \
                 random_features=None, linear_bandit=False, domain=None, f_real=None):
        """
        This Quantum Bayesian Optimization script combines classical Bayesian Optimization with quantum techniques to accelerate 
        the search for the maximum of an unknown function. In this approach, the Gaussian process model typically used in classical 
        Bayesian Optimization is approximated using random Fourier features. This provides an efficient representation of kernels, 
        such as the RBF kernel, which are specifically used for finding the maximum. These features are then used in a quantum Monte 
        Carlo routine that uses amplitude amplification, similar to Grover-like methods, to estimate the acquisition function more 
        efficiently than classical sampling can.
        
        Parameters
        ----------
        f: the f(x) values of the oracle function
        pbounds: search space f(x) values of the function you want to approximate 
        beta_t: beta parameter in UCB, determines exploration vs exploitation
        random_features: random features for approximation the kernel
        linear_bandit: use linear bandit instead of kernel approximation, True or False
        domain: domain of the oracle function
        f_real: true function values (only for evaluation)
        """

        self.f = f
        self.pbounds = pbounds
        self.beta_t = beta_t
        self.random_features = random_features
        self.linear_bandit = linear_bandit
        self.domain = domain
        self.f_real = f_real

        self.initialized = False # Initialize the function for the first chosen iteration points
        self.incumbent = None
        
        self.keys = list(pbounds.keys())
        self.dim = len(pbounds)

        # Convert dict to array
        self.bounds = []
        for key in self.pbounds.keys():
            self.bounds.append(self.pbounds[key])
        self.bounds = np.asarray(self.bounds)

        # Save observations
        self.X = np.array([]).reshape(-1, 1)
        self.Y = np.array([])
        self.eps_list = np.array([])

        # Save initial points
        self.init_points = []
    
        self.i = 0 # Iteration counter
        
        # Results tracker
        self.res = {}
        self.res['max'] = {'max_val': None,
                           'max_params': None}
        self.res['all'] = {'values':[], 'params':[], 'init_values':[], 'init_params':[], 'init':[], \
                          'f_values':[], 'init_f_values':[], 'noise_var_values':[], 'init_noise_var_values':[], \
                          'incumbent_x':[], \
                          'track_queries':[]}

        self.total_used_queries = 0 # Queries used while approximating 
        self.x_max_list = []
        
    def init(self, init_points):
        """
        Random sample initial points and evaluate them.
        
        Parameters:
        -----------
        init_points: number of random chosen iteration points
        """
        
        # Choose random points to measure first
        l = [np.random.uniform(x[0], x[1], size=init_points)
             for x in self.bounds]
        self.init_points += list(map(list, zip(*l))) # Make a list of the x coordinates chosen to be measured

        y_init = []
        for x in self.init_points:
            # Measure the chosen random points
            y, f_value, num_oracle_queries = self.f(x, 1, self.domain, self.f_real, self.random_features)

            # Determine the number of oracle queries and save it
            self.total_used_queries += num_oracle_queries
            self.res['all']['track_queries'].append(num_oracle_queries)
            
            # Save data
            y_init.append(y)
            self.res['all']['init_values'].append(y)
            self.res['all']['init_f_values'].append(f_value)
            self.res['all']['f_values'].append(f_value)          
            self.res['all']['init_params'].append(dict(zip(self.keys, x)))
        
        # Save initial data
        self.X = np.asarray(self.init_points)
        self.Y = np.asarray(y_init)        
        self.eps_list = np.ones(len(self.X))
        self.incumbent = np.max(y_init)

        self.initialized = True
        init = {"X":self.X, "Y":self.Y, "f_values":self.res['all']['init_f_values']}
        self.res['all']['init'] = init


    def maximize(self, n_iter=1000, init_points=1):
        """
        Quantum bayesian optimize the function over the chosen amount of iteration points.
        
        Parameters:
        -----------
        n_iter: number of iteration points that need to be measured
        init_points: number of random chosen iteration points
        """

        self.util_ucb = UtilityFunction()
        
        # Initialize if needed
        if not self.initialized:
            self.init(init_points)
        
        # Load the random fourrier features
        s = self.random_features["s"]
        b = self.random_features["b"]
        obs_noise = self.random_features["obs_noise"]
        v_kernel = self.random_features["v_kernel"]
        M_target = b.shape[0]
        
        # If linear bandit then dimensions are the same as the features
        if self.linear_bandit:
            M_target = self.dim
        
        # Build feature matrix 
        Phi = np.zeros((self.X.shape[0], M_target))
            
        # Compute the features and the posterior
        for i, x in enumerate(self.X):
            x_vec = np.atleast_2d(x)
            if not self.linear_bandit:
                features = np.sqrt(2 / M_target) * np.cos(x_vec @ s.T + b)
                features = features / np.sqrt(np.inner(features, features))
                features = np.sqrt(v_kernel) * features
                features = features * (1 / self.eps_list[i])
            else:
                features = x_vec
                
            Phi[i, :] = features

        # Observation noise 
        lam = 1 
        
        # Post covariance and mean for linear regression  
        Sigma_t = Phi.T @ Phi + lam * np.eye(M_target)
        Sigma_t_inv = np.linalg.inv(Sigma_t)
        
        # Determine the weight of the points for the linear regression
        Y_weighted = self.Y.reshape(-1,1) / self.eps_list[:, None]**2
        nu_t = Sigma_t_inv @ Phi.T @ Y_weighted
        
        
        # Select next point to  measure via Upper Confidence Bound acquisition 
        x_max = acq_max(ac=self.util_ucb.utility, M=M_target, random_features=self.random_features, \
                        bounds=self.bounds, nu_t=nu_t, Sigma_t_inv=Sigma_t_inv, beta=self.beta_t[len(self.X)-1], \
                        domain=self.domain, linear_bandit=self.linear_bandit)
            
        self.x_max_list.append(x_max)
        
        # Feature vector for the selected points
        x_vec = np.atleast_2d(x_max)

        # Calculate features for kernel approximation 
        if not self.linear_bandit:
            x = np.squeeze(x_max).reshape(1, -1)
            features = np.sqrt(2 / M_target) * np.cos(np.squeeze(np.dot(x, s.T)) + b)
            features = features.reshape(-1, 1)
            features = features / np.sqrt(np.inner(np.squeeze(features), np.squeeze(features)))
            features = np.sqrt(v_kernel) * features # v_kernel is set to be 1 here in the synthetic experiments
        else:
            features = x.reshape(-1, 1) # Only reshape if linear bandit 
        
        # Predict variance and uncertainty 
        var = lam * (features.T @ Sigma_t_inv @ features).item()
        eps = np.sqrt(var) / np.sqrt(lam)
        self.eps_list = np.append(self.eps_list, eps)
        
        # Main optimization loop 
        while self.total_used_queries < n_iter:
            # Determine the y, f_value and the number of queries to find the best measured points
            y, f_value, num_oracle_queries = self.f(x_max, self.eps_list[-1], self.domain, self.f_real, self.random_features)
    
            self.total_used_queries += num_oracle_queries
            self.res['all']['track_queries'].append(num_oracle_queries)
            self.res['all']['f_values'].append(f_value)
            
            # Save the current iteration points
            self.Y = np.append(self.Y, y)
            self.X = np.vstack((self.X, x_max))
            
            # Save incumbent 
            incumbent_x = self.X[np.argmax(self.Y)]
            self.res['all']['incumbent_x'].append(incumbent_x)

            # Recompute the features and the posterior
            Phi = np.zeros((self.X.shape[0], M_target)) # Determine the feature matrix for all the current measured points
            for i, x in enumerate(self.X):
                if not self.linear_bandit:
                    x = np.squeeze(x).reshape(1, -1)
                    features = np.sqrt(2 / M_target) * np.cos(np.squeeze(np.dot(x, s.T)) + b)
                    features = features / np.sqrt(np.inner(features, features))
                    features = np.sqrt(v_kernel) * features
                    features = features * (1 / self.eps_list[i])
                else:
                    features = x
                
                Phi[i, :] = features

            # Determine the new covariance
            Sigma_t = np.dot(Phi.T, Phi) + lam * np.identity(M_target)
            Sigma_t_inv = np.linalg.inv(Sigma_t)
            
            # Calculate the new weighted points
            Y_weighted = np.matmul(np.diag(1 / self.eps_list**2), self.Y.reshape(-1, 1))
            nu_t = np.dot(np.dot(Sigma_t_inv, Phi.T), Y_weighted)
            
            # Select next point to measure 
            x_max = acq_max(ac=self.util_ucb.utility, M=M_target, random_features=self.random_features, \
                            bounds=self.bounds, nu_t=nu_t, Sigma_t_inv=Sigma_t_inv, beta=self.beta_t[len(self.X)-1], \
                            domain=self.domain, linear_bandit=self.linear_bandit)
            
            self.x_max_list.append(x_max)
                
            # Update features
            if not self.linear_bandit:
                x = np.squeeze(x_max).reshape(1, -1)
                features = np.sqrt(2 / M_target) * np.cos(np.squeeze(np.dot(x, s.T)) + b)
                features = features.reshape(-1, 1)
                features = features / np.sqrt(np.inner(np.squeeze(features), np.squeeze(features)))
                features = np.sqrt(v_kernel) * features # V_kernel is set to be 1 here in the synthetic experiments
            else:
                features = x.reshape(-1, 1)
            
            # Calculate the new variance and uncertainty 
            var = lam * np.squeeze(np.dot(np.dot(features.T, Sigma_t_inv), features))
            eps = np.sqrt(var) / np.sqrt(lam)
            self.eps_list = np.append(self.eps_list, eps)
            
            # Add 1 for chosen iterations 
            self.i += 1

            print(f"iter {self.i} ------ x_t: {x_max}, y_t: {y}")

            # Evaluate chosen points
            x_max_param = self.X[self.Y.argmax(), :-1]

            # Save results
            self.res['max'] = {'max_val': self.Y.max(), 'max_params': dict(zip(self.keys, x_max_param))}
            self.res['all']['values'].append(self.Y[-1])
            self.res['all']['params'].append(self.X[-1])

        return self.X, self.res, Phi, features, nu_t, Sigma_t, self.i
