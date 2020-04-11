from functools import partial

import numpy as np

from .utils import get_dt, shift_array
from .optimization import NewtonMethod

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Critic:
    
    def __init__(self, eta=None, beta=None, x0=None):
        self.u_kernel = eta
        self.beta = beta
        self.x0 = x0
        self.log_likelihood_iterations = []
        
    def copy(self):
        return self.__class__(eta=self.u_kernel.copy(), beta=self.beta, x0=self.x0)
        
    def fit(self, t, mask_spikes, u, y, newton_kwargs=None, verbose=False):

        newton_kwargs = {} if newton_kwargs is None else newton_kwargs
        
        objective_kwargs = self.objective_kwargs(t, mask_spikes, u, y)

        gh_objective = partial(self.gh_objective, **objective_kwargs)

        optimizer = NewtonMethod(model=self, gh_objective=gh_objective, verbose=verbose, **newton_kwargs)
        optimizer.optimize()

        return optimizer

    # def update_theta(self, dt, X_u, mask_spikes, y, lr):
    #
    #     theta = self.get_params()
    #     w_distance, g_w_distance, h_w_distance = self.gh_wasserstein_distance(dt, X_u, mask_spikes, y)
    #     theta = theta + lr * g_w_distance
    #     # self.log_likelihood_iterations.append(log_likelihood)
    #
    #     return theta
    
    def get_params(self):
#         n_eta = self.u_kernel.nbasis
        # theta = np.zeros((1 + len(self.beta) + n_eta))
        # theta[0] = self.x0
        theta= self.u_kernel.coefs.copy()
        return theta
    
    def set_params(self, theta):
        # self.x0 = theta[0]
        # self.beta = theta[1:]
#         self.u0 = theta[2]
        self.u_kernel.coefs = theta[:]
        return self
    
    def gh_wasserstein_distance(self, dt, mask_spikes, X_u, y):

        theta = self.get_params()

        X_u_theta = np.einsum('tka,a->k', X_u, theta)
        a = X_u_theta

        w_distance = np.mean(a[y == 1]) - np.mean(a[y == 0])
        g_w_distance = np.mean(np.sum(X_u[:, y == 1, :], 0), 0) - np.mean(np.sum(X_u[:, y == 0, :], 0), 0)
        h_w_distance = None

        return w_distance, g_w_distance, h_w_distance

    def gh_objective(self,  dt, mask_spikes, X_u, y):
        return self.gh_wasserstein_distance(dt, mask_spikes, X_u, y)
            
    def wasserstein_distance_kwargs(self, t, mask_spikes, u, y):

        # args = np.where(shift_array(mask_spikes, 1, fill_value=False))
        # t_spk = (t[args[0]],) + args[1:]
        if self.u_kernel is not None:
            n_u_kernel = self.u_kernel.nbasis
            X_u = self.u_kernel.convolve_basis_continuous(t, u)
            # X[:, :, 1:n_u_kernel + 1] = X_u
        # X = X_u.copy()
        # if self.u_kernel is not None:
        #     n_eta = self.u_kernel.nbasis
        #     X_eta = self.u_kernel.convolve_basis_discrete(t, t_spk, shape=mask_spikes.shape)
        #     X = np.zeros(mask_spikes.shape + (1 + n_eta,))
        #     X[:, :, 1:] = -X_eta
        # else:
        #     n_eta = 0
        #     X = np.zeros(mask_spikes.shape + (1,))

        wasserstein_kwargs = dict(dt=get_dt(t), mask_spikes=mask_spikes, X_u=X_u, y=y)

        return wasserstein_kwargs

    def objective_kwargs(self, t, mask_spikes, u, y):
        return self.wasserstein_distance_kwargs(t=t, mask_spikes=mask_spikes, u=u, y)