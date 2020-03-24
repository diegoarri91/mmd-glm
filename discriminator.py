import numpy as np

from functools import partial

from icglm.masks import shift_mask
from icglm.utils.time import get_dt
from icglm.optimization import NewtonMethod

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class DiscriminatorGLM:
    
    def __init__(self, eta=None, beta=None, x0=None):
        self.eta = eta
        self.beta = beta
        self.x0 = x0
        self.log_likelihood_iterations = []
        
    def copy(self):
        return self.__class__(u0=self.u0, eta=self.eta.copy(), beta=self.beta, x0=self.x0)
        
    def fit(self, t, stim, mask_spikes, X, y, newton_kwargs=None, verbose=False, seed=0):

        newton_kwargs = {} if newton_kwargs is None else newton_kwargs

        np.random.seed(seed)
        
        newton_kwargs = newton_kwargs if newton_kwargs is not None else {}
        
        max_iterations = newton_kwargs.get('max_iterations', 100)
        lr = newton_kwargs.get('lr', 1e-1)
        
        dt = t[1]
        log_likelihood_iterations = []
        theta = np.zeros(X.shape[1] + 1)
        theta[1:X.shape[1] + 1] = np.random.rand(X.shape[1])
        theta[0] = (np.mean(X[y == 1, :] @ theta[1:]) + np.mean(X[y == 0, :] @ theta[1:])) / 2
        for ii in range(max_iterations):
            theta = self.update_theta(theta, dt, X, mask_spikes, y)
            
        self.x0 = theta[0]
        self.beta = theta[1:X.shape[1] + 1]
        
        return self
    
    def update_theta(self, theta, dt, X, mask_spikes, y):
        
        log_likelihood, g_log_likelihood = self.gh_log_likelihood(theta, dt, X, mask_spikes, y)
        g_log_likelihood[np.abs(g_log_likelihood) >1e3] = np.sign(g_log_likelihood[np.abs(g_log_likelihood) >1e3]) * 1e3
        self.log_likelihood_iterations.append(log_likelihood)
        theta = theta + lr * g_log_likelihood
        
        return theta
    
    def predict(self, t, stim, mask_spikes, X):
        y_proba = self.predict_proba(t, stim, mask_spikes, X)
        return np.array(y_proba > 0.5, dtype=int)
    
    def predict_proba(self, t, stim, mask_spikes, X):
#         likelihood_kwargs = self.get_likelihood_kwargs(t, stim, mask_spikes)
#         dt, X, mask_spikes = likelihood_kwargs['dt'], likelihood_kwargs['X'], likelihood_kwargs['mask_spikes']
        theta = self.get_theta()

        a = X @ theta[1:X.shape[1] + 1] - theta[0]
        y_proba = sigmoid(a)
        
#         u = X @ theta[2:]
#         r = np.exp(u)
#         u_spk = np.array([np.sum(u[mask_spikes[:, j], j]) for j in range(mask_spikes.shape[1])])
#         a = theta[1] * (u_spk - dt * np.sum(r, 0)) - theta[0]
        
        return y_proba
    
    def get_theta(self):
#         n_eta = self.eta.nbasis
        n_eta = 0
        theta = np.zeros((1 + len(self.beta) + n_eta))
        theta[0] = self.x0
        theta[1:] = self.beta
#         theta[2] = self.u0
#         theta[1:1 + n_kappa] = self.kappa.coefs
#         theta[3 + n_kappa:] = self.eta.coefs
        return theta
    
    def set_params(self, theta):
        self.x0 = theta[0]
        self.beta = theta[1:]
#         self.u0 = theta[2]
#         self.eta.coefs = theta[n_kappa + 3:]
        return self
    
    def gh_log_likelihood(self, theta, dt, X, mask_spikes, y):
        
        a = X @ theta[1:X.shape[1] + 1] - theta[0]
        da_dtheta = np.zeros((X.shape[0], len(theta)))
        da_dtheta[:, 0] = -1.
        da_dtheta[:, 1:] = X
        y_pred = sigmoid(a)
        
        eps = 1e-20
        
        log_likelihood = np.sum(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))
        g_log_likelihood = np.sum(da_dtheta.T * (y - y_pred), 1)
        
        return log_likelihood, g_log_likelihood
            
    def get_likelihood_kwargs(self, t, stim, mask_spikes):

#         n_kappa = self.kappa.nbasis
        n_kappa = 0
#         X_kappa = self.kappa.convolve_basis_continuous(t, stim - stim_h)

        args = np.where(shift_mask(mask_spikes, 1, fill_value=False))
        t_spk = (t[args[0]],) + args[1:]
        if self.eta is not None:
            n_eta = self.eta.nbasis
            X_eta = self.eta.convolve_basis_discrete(t, t_spk, shape=mask_spikes.shape)
            X = np.zeros(mask_spikes.shape + (1 + n_kappa + n_eta,))
            X[:, :, n_kappa + 1:] = -X_eta
        else:
            n_eta = 0
            X = np.zeros(mask_spikes.shape + (1 + n_kappa,))

        X[:, :, 0] = -1.
#         X[:, :, 1:n_kappa + 1] = X_kappa

#         X = X.reshape(-1, 1 + n_kappa + n_eta)
#         mask_spikes = mask_spikes.reshape(-1)

        likelihood_kwargs = dict(dt=get_dt(t), X=X, mask_spikes=mask_spikes)

        return likelihood_kwargs