from functools import partial
import pickle

import numpy as np

from ..optimization import NewtonMethod
from ..utils import get_dt, shift_array

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
        params = dict(u0=self.u0, eta=self.eta)
        with open(path, "wb") as fit_file:
            pickle.dump(params, fit_file)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as fit_file:
            params = pickle.load(fit_file)
        glm = cls(u0=params['u0'], eta=params['eta'])
        return glm

    def sample(self, t, stim=None, shape=(1,), full=False):

        dt = get_dt(t)

        shape = (len(t), ) + shape
        r = np.zeros(shape) * np.nan
        eta_conv = np.zeros(shape)
        mask_spikes = np.zeros(shape, dtype=bool)

        j = 0
        while j < len(t):

            r[j, ...] = np.exp(-eta_conv[j, ...] - self.u0)

            p_spk = 1. - np.exp(-r[j, ...] * dt)
            aux = np.random.rand(*shape[1:])

            mask_spikes[j, ...] = p_spk > aux

            if self.eta is not None and np.any(mask_spikes[j, ...]) and j < len(t) - 1:
                eta_conv[j + 1:, mask_spikes[j, ...]] += self.eta.interpolate(t[j + 1:] - t[j + 1])[:, None]

            j += 1
        u = - eta_conv - self.u0
        if full:
            return eta_conv, u, r, mask_spikes
        else:
            return u, r, mask_spikes

    def simulate_subthreshold(self, t, stim, mask_spk, stim_h=0, full=False):

        if stim.ndim == 1:
            # shape = (len(t), 1)
            stim = stim.reshape(len(t), 1)
        if mask_spk.ndim == 1:
            # shape = (len(t), 1)
            mask_spk = mask_spk.reshape(len(t), 1)

        shape = mask_spk.shape
        dt = get_dt(t)
        arg_spikes = np.where(shift_array(mask_spk, 1, fill_value=False))
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

        return log_likelihood, g_log_likelihood, h_log_likelihood, None

    def gh_objective(self,  dt, X, mask_spikes):
        return self.gh_log_likelihood(dt, X, mask_spikes)
    
    def get_params(self):
        n_kappa = 0 
        n_eta = 0 if self.eta is None else self.eta.nbasis
        theta = np.zeros((1 + n_kappa + n_eta))
        theta[0] = self.u0
        theta[1:] = self.eta.coefs
        return theta

    def likelihood_kwargs(self, t, mask_spikes, stim=None):

        n_kappa = 0
        n_eta = 0 if self.eta is None else self.eta.nbasis

        X = np.zeros(mask_spikes.shape + (1 + n_kappa + n_eta,))
        X[:, :, 0] = -1.

        if self.eta is not None:
            args = np.where(shift_array(mask_spikes, 1, fill_value=False))
            t_spk = (t[args[0]],) + args[1:]
            n_eta = self.eta.nbasis
            X_eta = self.eta.convolve_basis_discrete(t, t_spk, shape=mask_spikes.shape)
            X[:, :, n_kappa + 1:] = -X_eta

        likelihood_kwargs = dict(dt=get_dt(t), X=X, mask_spikes=mask_spikes)

        return likelihood_kwargs

    def objective_kwargs(self, t, mask_spikes, stim=None):
        return self.likelihood_kwargs(t=t, mask_spikes=mask_spikes, stim=stim)

    def set_params(self, theta):
        n_kappa = 0
        self.u0 = theta[0]
        self.eta.coefs = theta[n_kappa + 1:]
        return self

    def fit(self, t, mask_spikes, stim=None, newton_kwargs=None, verbose=False):

        newton_kwargs = {} if newton_kwargs is None else newton_kwargs

        objective_kwargs = self.objective_kwargs(t, mask_spikes)
        gh_objective = partial(self.gh_objective, **objective_kwargs)

        optimizer = NewtonMethod(model=self, gh_objective=gh_objective, verbose=verbose, **newton_kwargs)
        optimizer.optimize()

        return optimizer
