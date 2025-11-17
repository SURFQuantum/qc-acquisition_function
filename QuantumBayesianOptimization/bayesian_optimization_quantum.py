# -*- coding: utf-8 -*-
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
        """
        self.linear_bandit = linear_bandit
        self.domain = domain
        self.f_real = f_real

        self.random_features = random_features

        self.pbounds = pbounds
        self.incumbent = None
        self.beta_t = beta_t
        
        self.keys = list(pbounds.keys())
        self.dim = len(pbounds)

        self.bounds = []
        for key in self.pbounds.keys():
            self.bounds.append(self.pbounds[key])
        self.bounds = np.asarray(self.bounds)
        
        self.f = f

        self.initialized = False

        self.init_points = []
        self.x_init = []
        self.y_init = []

        self.X = np.array([]).reshape(-1, 1)
        self.Y = np.array([])
        
        self.i = 0

        self.util = None
        
        self.res = {}
        self.res['max'] = {'max_val': None,
                           'max_params': None}
        self.res['all'] = {'values':[], 'params':[], 'init_values':[], 'init_params':[], 'init':[], \
                          'f_values':[], 'init_f_values':[], 'noise_var_values':[], 'init_noise_var_values':[], \
                          'incumbent_x':[], \
                          'track_queries':[]}

        self.total_used_queries = 0
        self.eps_list = np.array([])
        
        self.x_max_list = []
        self.Z_list = []
        

    def init(self, init_points):
        # choose random points to measure first
        l = [np.random.uniform(x[0], x[1], size=init_points)
             for x in self.bounds]
        
        #print('Inizialize point: ',l)
        self.init_points += list(map(list, zip(*l)))
        y_init = []
        for x in self.init_points:
            #measure the chosen random points
            y, f_value, num_oracle_queries = self.f(x, 1, self.domain, self.f_real, self.random_features)

            self.total_used_queries += num_oracle_queries
            self.res['all']['track_queries'].append(num_oracle_queries)
            
            # save data
            y_init.append(y)
            self.res['all']['init_values'].append(y)
            self.res['all']['init_f_values'].append(f_value)
            self.res['all']['f_values'].append(f_value)
            
            #print('measured f_values of the initialize point: ',self.res['all']['f_values'])
            
            self.res['all']['init_params'].append(dict(zip(self.keys, x)))
        
        # save the chosen points
        self.X = np.asarray(self.init_points)
        self.Y = np.asarray(y_init)        
        
        self.eps_list = np.ones(len(self.X))
        
        # ???
        self.incumbent = np.max(y_init)
        self.initialized = True

        init = {"X":self.X, "Y":self.Y, "f_values":self.res['all']['init_f_values']}
        self.res['all']['init'] = init


    def maximize(self, n_iter=1000, init_points=1):
        self.util_ucb = UtilityFunction()
        
        #initizalizing
        if not self.initialized:
            self.init(init_points)
        
        #calling the features
        s = self.random_features["s"]
        b = self.random_features["b"]
        obs_noise = self.random_features["obs_noise"]
        v_kernel = self.random_features["v_kernel"]
        M_target = b.shape[0]
        
        # if linear bandit then dimensions are the same as the features
        if self.linear_bandit:
            M_target = self.dim
        
        
        Phi = np.zeros((self.X.shape[0], M_target)) #determine the feature matrix for all the current measured points
            
        # features are the same
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
            #Phi[i,:] /=  np.linalg.norm(Phi[i,:], axis=1, keepdims=True)
        #print('Phi1: ',Phi)

        lam = 1 #???
        
        # covariance for linear regression 
        Sigma_t = Phi.T @ Phi + lam * np.eye(M_target)
        Sigma_t_inv = np.linalg.inv(Sigma_t)
        
        # determine the weight of the points for the linear regression
        Y_weighted = self.Y.reshape(-1,1) / self.eps_list[:, None]**2
        nu_t = Sigma_t_inv @ Phi.T @ Y_weighted
        
        
        #acquosition function to determine the next spot
        x_max = acq_max(ac=self.util_ucb.utility, M=M_target, random_features=self.random_features, \
                        bounds=self.bounds, nu_t=nu_t, Sigma_t_inv=Sigma_t_inv, beta=self.beta_t[len(self.X)-1], \
                        domain=self.domain, linear_bandit=self.linear_bandit)
            
        self.x_max_list.append(x_max)
        
        # give the features the right shape, mostly don't do this
        x_vec = np.atleast_2d(x_max)
        if not self.linear_bandit:
            x = np.squeeze(x_max).reshape(1, -1)
            features = np.sqrt(2 / M_target) * np.cos(np.squeeze(np.dot(x, s.T)) + b)
            features = features.reshape(-1, 1)
            features = features / np.sqrt(np.inner(np.squeeze(features), np.squeeze(features)))
            features = np.sqrt(v_kernel) * features # v_kernel is set to be 1 here in the synthetic experiments
        else:
            features = x.reshape(-1, 1)
            #features /= np.linalg.norm(features, axis=1, keepdims=True)
        #print('features 1: ',features)
            
        var = lam * (features.T @ Sigma_t_inv @ features).item()
        eps = np.sqrt(var) / np.sqrt(lam)
        self.eps_list = np.append(self.eps_list, eps)
        
        #evaluate next point
        while self.total_used_queries < n_iter:
            y, f_value, num_oracle_queries = self.f(x_max, self.eps_list[-1], self.domain, self.f_real, self.random_features)
    
            self.total_used_queries += num_oracle_queries
            self.res['all']['track_queries'].append(num_oracle_queries)
    
            self.res['all']['f_values'].append(f_value)
            
            #print('\n Measured Xes: ',self.x_max_list)
            #print('Measured f_values of the x: ',self.res['all']['f_values'])
            
            # save the current itteration points
            self.Y = np.append(self.Y, y)
            self.X = np.vstack((self.X, x_max))
            
            incumbent_x = self.X[np.argmax(self.Y)]
            self.res['all']['incumbent_x'].append(incumbent_x)
            
            #load the random features again
            s = self.random_features["s"]
            b = self.random_features["b"]
            obs_noise = self.random_features["obs_noise"]
            v_kernel = self.random_features["v_kernel"]
            M_target = b.shape[0]
            
            #we don't do this
            if self.linear_bandit:
                M_target = self.dim

            # again but with the new points
            Phi = np.zeros((self.X.shape[0], M_target))
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
                #Phi[i,:] /=  np.linalg.norm(Phi[i,:], axis=1, keepdims=True)
            #print('Phi2: ',Phi)
            #determine the new covariance
            Sigma_t = np.dot(Phi.T, Phi) + lam * np.identity(M_target)
            Sigma_t_inv = np.linalg.inv(Sigma_t)
            
            #Calculate the weighted points again
            Y_weighted = np.matmul(np.diag(1 / self.eps_list**2), self.Y.reshape(-1, 1))
            nu_t = np.dot(np.dot(Sigma_t_inv, Phi.T), Y_weighted)
            
            #calculate the new points
            x_max = acq_max(ac=self.util_ucb.utility, M=M_target, random_features=self.random_features, \
                            bounds=self.bounds, nu_t=nu_t, Sigma_t_inv=Sigma_t_inv, beta=self.beta_t[len(self.X)-1], \
                            domain=self.domain, linear_bandit=self.linear_bandit)
            
            self.x_max_list.append(x_max)
                
            # we don't do this
            if not self.linear_bandit:
                x = np.squeeze(x_max).reshape(1, -1)
                features = np.sqrt(2 / M_target) * np.cos(np.squeeze(np.dot(x, s.T)) + b)
                
                features = features.reshape(-1, 1)
                features = features / np.sqrt(np.inner(np.squeeze(features), np.squeeze(features)))
                features = np.sqrt(v_kernel) * features # v_kernel is set to be 1 here in the synthetic experiments
            else:
                features = x.reshape(-1, 1)
                #features /= np.linalg.norm(features, axis=1, keepdims=True)
            #print('features2: ',features)
            
            #calculate the covariance once agian
            var = lam * np.squeeze(np.dot(np.dot(features.T, Sigma_t_inv), features))
            
            eps = np.sqrt(var) / np.sqrt(lam)

            self.eps_list = np.append(self.eps_list, eps)
            
            self.i += 1

            print("iter {0} ------ x_t: {1}, y_t: {2}".format(self.i, x_max, y))

            # evualute chosen points
            x_max_param = self.X[self.Y.argmax(), :-1]

            self.res['max'] = {'max_val': self.Y.max(), 'max_params': dict(zip(self.keys, x_max_param))}
            self.res['all']['values'].append(self.Y[-1])
            self.res['all']['params'].append(self.X[-1])

        return self.X, self.res, Phi, features, nu_t, Sigma_t, self.i
