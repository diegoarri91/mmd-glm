from functools import partial
import pickle
import sys
sys.path.append("/home/diego/Dropbox/hold_noise/iclamp-glm/")

import numpy as np
from sklearn.metrics import recall_score

# from icglm.models.base import BayesianSpikingModel
from optimization import NewtonMethod
from icglm.masks import shift_mask
from icglm.utils.time import get_dt

class GLM:

    def __init__(self, u0, eta):
        self.u0 = u0
        self.eta = eta

    @property
    def r0(self):
        return np.exp(-self.u0)

    def copy(self):
        return self.__class__(u0=self.u0, eta=self.eta.copy())

    def save(self, path):
        params = dict(u0=self.u0, kappa=self.kappa, eta=self.eta)
        with open(path, "wb") as fit_file:
            pickle.dump(params, fit_file)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as fit_file:
            params = pickle.load(fit_file)
        glm = cls(u0=params['u0'], kappa=params['kappa'], eta=params['eta'])
        return glm

    def sample(self, t, stim, full=False):

        # np.seterr(over='ignore')  # Ignore overflow warning when calculating r[j+1] which can be very big

        dt = get_dt(t)

        if stim.ndim == 1:
            shape = (len(t), 1)
            stim = stim.reshape(len(t), 1)
        else:
            shape = stim.shape

        r = np.zeros(shape) * np.nan
        eta_conv = np.zeros(shape)
        mask_spk = np.zeros(shape, dtype=bool)

        j = 0
        while j < len(t):

            r[j, ...] = np.exp(-eta_conv[j, ...] - self.u0)

            p_spk = 1. - np.exp(-r[j, ...] * dt)
            aux = np.random.rand(*shape[1:])

            mask_spk[j, ...] = p_spk > aux

            if self.eta is not None and np.any(mask_spk[j, ...]) and j < len(t) - 1:
                eta_conv[j + 1:, mask_spk[j, ...]] += self.eta.interpolate(t[j + 1:] - t[j + 1])[:, None]

            j += 1
        v = - eta_conv - self.u0
        if full:
            return eta_conv, v, r, mask_spk
        else:
            return v, r, mask_spk

    def simulate_subthreshold(self, t, stim, mask_spk, stim_h=0, full=False):

        if stim.ndim == 1:
            # shape = (len(t), 1)
            stim = stim.reshape(len(t), 1)
        if mask_spk.ndim == 1:
            # shape = (len(t), 1)
            mask_spk = mask_spk.reshape(len(t), 1)

        shape = mask_spk.shape
        dt = get_dt(t)
        arg_spikes = np.where(shift_mask(mask_spk, 1, fill_value=False))
        t_spikes = (t[arg_spikes[0]], arg_spikes[1])

        if self.eta is not None and len(t_spikes[0]) > 0:
            eta_conv = self.eta.convolve_discrete(t, t_spikes, shape=shape[1:])
        else:
            eta_conv = np.zeros(shape)

        u = - eta_conv - self.u0
        r = np.exp(u)

        if full:
            return eta_conv, u, r
        else:
            return u, r

    def use_prior_kernels(self):
        return False

    def gh_log_likelihood(self, dt, X, mask_spikes):

        theta = self.get_params()
        u = np.einsum('tka,a->tk', X, theta)
        r = np.exp(u)

        log_likelihood = np.sum(u[mask_spikes]) - dt * np.sum(r)
        g_log_likelihood = np.sum(X[mask_spikes, :], axis=0) - dt * np.einsum('tka,tk->a', X, r)
        h_log_likelihood = - dt * np.einsum('tka,tk,tkb->ab', X, r, X)

        return log_likelihood, g_log_likelihood, h_log_likelihood
    
    def get_params(self):
        n_kappa = 0 
        n_eta = 0 if self.eta is None else self.eta.nbasis
        theta = np.zeros((1 + n_kappa + n_eta))
        theta[0] = self.u0
        theta[1:] = self.eta.coefs
        return theta

    def get_likelihood_kwargs(self, t, stim, mask_spikes, stim_h=0, newton_kwargs_d=None):

        n_kappa = 0
        n_eta = 0 if self.eta is None else self.eta.nbasis

        X = np.zeros(mask_spikes.shape + (1 + n_kappa + n_eta,))
        X[:, :, 0] = -1.

        if self.eta is not None:
            args = np.where(shift_mask(mask_spikes, 1, fill_value=False))
            t_spk = (t[args[0]],) + args[1:]
            n_eta = self.eta.nbasis
            X_eta = self.eta.convolve_basis_discrete(t, t_spk, shape=mask_spikes.shape)
            X[:, :, n_kappa + 1:] = -X_eta

        likelihood_kwargs = dict(dt=get_dt(t), X=X, mask_spikes=mask_spikes)

        return likelihood_kwargs

    def set_params(self, theta):
        n_kappa = 0
        self.u0 = theta[0]
        self.eta.coefs = theta[n_kappa + 1:]
        return self

    def fit(self, t, stim, mask_spikes, stim_h=0, newton_kwargs=None, verbose=False, newton_kwargs_d=None, 
            discriminator='r_sum'):
        
        newton_kwargs = {} if newton_kwargs is None else newton_kwargs

        theta0 = self.get_params()
        likelihood_kwargs = self.get_likelihood_kwargs(t, stim, mask_spikes, stim_h=stim_h)

        gh_log_prior = None if not(self.use_prior_kernels()) else self.gh_log_prior_kernels
        if discriminator == 'r_sum':
            gh_log_likelihood = partial(self.gh_log_likelihood_r_sum, **likelihood_kwargs)
        else:
            gh_log_likelihood = partial(self.gh_log_likelihood_kernels, **likelihood_kwargs)

        optimizer = NewtonMethod(theta0=theta0, gh_log_prior=gh_log_prior, gh_log_likelihood=gh_log_likelihood,
                                 verbose=verbose, **newton_kwargs)
        optimizer.optimize()

        theta = optimizer.theta_iterations[:, -1]
        self.set_params(theta)

        return optimizer
    