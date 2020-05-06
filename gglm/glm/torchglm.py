from functools import partial
import pickle

import numpy as np

from .base import GLM
from ..optimization import NewtonMethod
from ..utils import get_dt, shift_array

class TorchGLM(GLM):

    def __init__(self, u0=0, kappa=None, eta=None):
        super().

    def gh_log_likelihood(self, dt, X, mask_spikes):

        theta = self.get_params()
        # TODO. adapt for arbitrary number of dims in X
        u = np.einsum('tka,a->tk', X, theta)
        r = np.exp(u)

        log_likelihood = np.sum(u[mask_spikes]) - dt * np.sum(r)
        g_log_likelihood = np.sum(X[mask_spikes, :], axis=0) - dt * np.einsum('tka,tk->a', X, r)
        h_log_likelihood = - dt * np.einsum('tka,tk,tkb->ab', X, r, X)

        return log_likelihood, g_log_likelihood, h_log_likelihood, None

    def gh_objective(self,  dt, X, mask_spikes):
        return self.gh_log_likelihood(dt, X, mask_spikes)

    def fit(self, t, mask_spikes, stim=None, newton_kwargs=None, verbose=False):
        
        newton_kwargs = {} if newton_kwargs is None else newton_kwargs
        objective_kwargs = self.objective_kwargs(t, mask_spikes, stim=stim)
        pass
