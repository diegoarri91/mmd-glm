from functools import partial

import numpy as np

from .utils import get_dt, shift_array
from .optimization import NewtonMethod


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Critic:
    
    def __init__(self, u_kernel=None, u_spk_kernel=None, r_kernel=None, r_spk_kernel=None, beta=None, features=None):
        self.u_kernel = u_kernel
        self.u_spk_kernel = u_spk_kernel
        self.r_kernel = r_kernel
        self.r_spk_kernel = r_spk_kernel
        self.features = features if features is not None else []
        self.beta = beta
        
    def copy(self):
        u_kernel = None if self.u_kernel is None else self.u_kernel.copy()
        u_spk_kernel = None if self.u_spk_kernel is None else self.u_spk_kernel.copy()
        r_kernel = None if self.r_kernel is None else self.r_kernel.copy()
        r_spk_kernel = None if self.r_spk_kernel is None else self.r_spk_kernel.copy()
        beta = None if self.beta is None else self.beta.copy()
        return self.__class__(u_kernel=u_kernel, u_spk_kernel=u_spk_kernel, r_kernel=r_kernel, r_spk_kernel=r_spk_kernel, features=self.features.copy(), 
                              beta=beta)
        
    def fit(self, t, mask_spikes, u, r, y, u0=None, newton_kwargs=None, verbose=False):

        newton_kwargs = {} if newton_kwargs is None else newton_kwargs
        
        objective_kwargs = self.objective_kwargs(t, mask_spikes, u, r, y, u0=u0)
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
        
        theta = []
        if self.u_kernel is not None:
            theta.append(self.u_kernel.coefs.copy())
            
        if self.u_spk_kernel is not None:
            theta.append(self.u_spk_kernel.coefs.copy())
            
        if self.r_kernel is not None:
            theta.append(self.r_kernel.coefs.copy())
            
        if self.r_spk_kernel is not None:
            theta.append(self.r_spk_kernel.coefs.copy())
            
        ii = 0
        if 'r_spk' in self.features:
            theta.append(np.array([self.beta[ii]]))
        theta = np.concatenate(theta)
        return theta
    
    def set_params(self, theta):
        # self.x0 = theta[0]
        # self.beta = theta[1:]
#         self.u0 = theta[2
        ii = 0
        if self.u_kernel is not None:
            self.u_kernel.coefs = theta[ii:ii + len(self.u_kernel.coefs)]
            ii += len(self.u_kernel.coefs)
        
        if self.u_spk_kernel is not None:
            self.u_spk_kernel.coefs = theta[ii:ii + len(self.u_spk_kernel.coefs)]
            ii += len(self.u_spk_kernel.coefs)
        
        if self.r_kernel is not None:
            self.r_kernel.coefs = theta[ii:ii + len(self.r_kernel.coefs)]
            ii += len(self.r_kernel.coefs)
            
        if self.r_spk_kernel is not None:
            self.r_spk_kernel.coefs = theta[ii:ii + len(self.r_spk_kernel.coefs)]
            ii += len(self.r_spk_kernel.coefs)
            
        if 'r_spk' in self.features:
            self.beta[ii] = theta[ii]
            ii += 1 
        
        return self

    def transform(self, t, mask_spikes, u, r, u0=None):

        theta = self.get_params()
        wasserstein_kwargs = self.wasserstein_distance_kwargs(t, mask_spikes, u, r, None, u0=u0)
        X_u = wasserstein_kwargs['X_u']
        X_u_spk = wasserstein_kwargs['X_u_spk']
        r_spk = wasserstein_kwargs['r_spk']
        X_r = wasserstein_kwargs['X_r']
        X_r_spk = wasserstein_kwargs['X_r_spk']
        
        a = np.zeros(mask_spikes.shape[1])

        ii = 0
        if X_u is not None:
            X_u_theta = np.einsum('tka,a->k', X_u, theta[ii:ii + len(self.u_kernel.coefs)])
            a = a + X_u_theta
            ii += len(self.u_kernel.coefs)
        
        if X_u_spk is not None:
            X_u_spk_theta = np.einsum('tka,a->k', X_u_spk, theta[ii:ii + len(self.u_spk_kernel.coefs)])
            a = a + X_u_spk_theta
            ii += len(self.u_spk_kernel.coefs)
            
        if X_r is not None:
            X_r_theta = np.einsum('tka,a->k', X_r, theta[ii:ii + len(self.r_kernel.coefs)])
            a = a + X_r_theta
            ii += len(self.r_kernel.coefs)
        
        if X_r_spk is not None:
            X_r_spk_theta = np.einsum('tka,a->k', X_r_spk, theta[ii:ii + len(self.r_spk_kernel.coefs)])
            a = a + X_r_spk_theta
            ii += len(self.r_spk_kernel.coefs)
            
        if r_spk is not None:
            a = a + theta[ii] * r_spk
            ii += 1

        return a
    
    def gh_wasserstein_distance(self, dt, mask_spikes, X_u, X_u_spk, r_spk, X_r, X_r_spk, y):

        theta = self.get_params()
            
        a = np.zeros(len(y))
        g_w_distance = []
        
        ii = 0
#         print('X_u', X_u)
        if X_u is not None:
            X_u_theta = np.einsum('tka,a->k', X_u, theta[ii:ii + len(self.u_kernel.coefs)])
            a = a + X_u_theta
            g_w_distance.append(np.mean(np.sum(X_u[:, y == 1, :], 0), 0) - np.mean(np.sum(X_u[:, y == 0, :], 0), 0))
            ii += len(self.u_kernel.coefs)

        if X_u_spk is not None:
            X_u_spk_theta = np.einsum('tka,a->k', X_u_spk, theta[ii:ii + len(self.u_spk_kernel.coefs)])
            a = a + X_u_spk_theta
#             print(np.mean(a[y==1]))
            g_w_distance.append(np.mean(np.sum(X_u_spk[:, y == 1, :], 0), 0) - np.mean(np.sum(X_u_spk[:, y == 0, :], 0), 0))
            ii += len(self.u_spk_kernel.coefs)
            
        if X_r is not None:
            X_r_theta = np.einsum('tka,a->k', X_r, theta[ii:ii + len(self.r_kernel.coefs)])
            a = a + X_r_theta
#             print(np.mean(a[y==1]))
            g_w_distance.append(np.mean(np.sum(X_r[:, y == 1, :], 0), 0) - np.mean(np.sum(X_r[:, y == 0, :], 0), 0))
            ii += len(self.r_kernel.coefs)
    
        if X_r_spk is not None:
            X_r_spk_theta = np.einsum('tka,a->k', X_r_spk, theta[ii:ii + len(self.r_spk_kernel.coefs)])
            a = a + X_r_spk_theta
#             print(np.mean(a[y==1]))
            g_w_distance.append(np.mean(np.sum(X_r_spk[:, y == 1, :], 0), 0) - np.mean(np.sum(X_r_spk[:, y == 0, :], 0), 0))
            ii += len(self.r_spk_kernel.coefs)
            
        if r_spk is not None:
            a = a + theta[ii] * r_spk
            g_w_distance.append(np.array([np.mean(r_spk[y == 1]) - np.mean(r_spk[y == 0])]))
            ii += 1
    
        w_distance = np.mean(a[y == 1]) - np.mean(a[y == 0])
        g_w_distance = np.concatenate((g_w_distance))
#         print(g_w_distance)
        h_w_distance = None

        return w_distance, g_w_distance, h_w_distance, None

    def gh_objective(self,  dt, mask_spikes, X_u, X_u_spk, r_spk, X_r, X_r_spk, y):
        return self.gh_wasserstein_distance(dt, mask_spikes, X_u, X_u_spk, r_spk, X_r, X_r_spk, y)
            
    def wasserstein_distance_kwargs(self, t, mask_spikes, u, r, y, u0=None):


        if self.u_kernel is not None:
            n_u_kernel = self.u_kernel.nbasis
            X_u = self.u_kernel.convolve_basis_continuous(t, u)
        else:
            X_u = None
            
#         if self.u_spk_kernel is not None:
#             n_u_spk_kernel = self.u_spk_kernel.nbasis
#             _u = u.copy()
#             _u[~mask_spikes] = 0
#             X_u_spk = self.u_spk_kernel.convolve_basis_continuous(t, _u)
#         else:
#             X_u_spk = None

        if self.u_spk_kernel is not None:
            args = np.where(shift_array(mask_spikes, 1, fill_value=False))
            t_spk = (t[args[0]],) + args[1:]
            n_eta = self.u_spk_kernel.nbasis
            X_u_spk = self.u_spk_kernel.convolve_basis_discrete(t, t_spk, shape=mask_spikes.shape)
#             print(u0, np.median(u))
            X_u_spk = X_u_spk * (u[..., None] + u0)
        else:
            X_u_spk = None
            
        if self.r_kernel is not None:
            n_r_kernel = self.r_kernel.nbasis
            X_r = self.r_kernel.convolve_basis_continuous(t, r)
        else:
            X_r = None
        
#         if self.r_spk_kernel is not None:
#             n_r_spk_kernel = self.r_spk_kernel.nbasis
#             _r = r.copy()
#             _r[~mask_spikes] = 0
#             X_r_spk = self.r_spk_kernel.convolve_basis_continuous(t, _r)
#         else:
#             X_r_spk = None

        if self.r_spk_kernel is not None:
            args = np.where(shift_array(mask_spikes, 1, fill_value=False))
            t_spk = (t[args[0]],) + args[1:]
            X_r_spk = self.r_spk_kernel.convolve_basis_discrete(t, t_spk, shape=mask_spikes.shape)
#             print(u0, np.median(u))
            X_r_spk = X_r_spk * r[..., None]
        else:
            X_r_spk = None
        
#         r_spk = np.array([np.sum(r[mask_spikes[:, sw], sw]) for sw in range(mask_spikes.shape[1])]) if 'r_spk' in self.features else None
        r_spk = None if 'r_spk' not in self.features else np.array([np.sum(r[mask_spikes[:, sw], sw]) for sw in range(mask_spikes.shape[1])])

        wasserstein_kwargs = dict(dt=get_dt(t), mask_spikes=mask_spikes, X_u=X_u, X_u_spk=X_u_spk, r_spk=r_spk, X_r=X_r, 
                                  X_r_spk=X_r_spk, y=y)

        return wasserstein_kwargs

    def objective_kwargs(self, t, mask_spikes, u, r, y, u0=None):
        return self.wasserstein_distance_kwargs(t=t, mask_spikes=mask_spikes, u=u, r=r, y=y, u0=u0)