import numpy as np

from .base import GLM
from ..utils import get_dt, shift_array

# should leave if I implement things properly
from functools import partial
from ..optimization import NewtonMethod

class WGLM(GLM):

    def __init__(self, u0, eta, critic):
        super().__init__(u0=u0, eta=eta)
        self.critic = critic

    def gh_objective(self, dt, X_te, mask_spikes_te, n_samples_fr, newton_kwargs_critic):

        theta = self.get_params()
        n_samples_te = mask_spikes_te.shape[1]
        t = np.arange(X_te.shape[0]) * dt

        u_te = np.einsum('tka,a->tk', X_te, theta)
        r_te = np.exp(u_te)

        u_fr, r_fr, mask_spikes_fr = self.sample(t, shape=(n_samples_fr,))
        X_fr = self.objective_kwargs(t, mask_spikes_fr)['X_te']

        mask_spikes = np.concatenate((mask_spikes_te, mask_spikes_fr), axis=1)
        u = np.concatenate((u_te, u_fr), axis=1)
        y = np.concatenate((np.ones(n_samples_te), np.zeros(n_samples_fr)))

        # print(mask_spikes.shape, u.shape)
        optimizer = self.critic.fit(t, mask_spikes, u, y, newton_kwargs=newton_kwargs_critic)

        log_likelihood = np.sum(u_te[mask_spikes_te]) - dt * np.sum(r_te)
        g_log_likelihood = np.sum(X_te[mask_spikes_te, :], axis=0) - dt * np.einsum('tka,tk->a', X_te, r_te)
        h_log_likelihood = - dt * np.einsum('tka,tk,tkb->ab', X_te, r_te, X_te)

        w_distance = optimizer.obj_iterations[-1]
        c_w_distance = np.mean(self.critic.transform(t, mask_spikes_fr, u_fr))
        g_c_w_distance = np.sum(self.critic.u_kernel.convolve_continuous(t, X_fr), axis=(0, 1)) / n_samples_fr
        # print(c_w_distance, g_c_w_distance)
        h_c_w_distance = np.zeros((len(theta), len(theta)))

        obj = log_likelihood + c_w_distance
        g_obj = g_log_likelihood + g_c_w_distance
        h_obj = h_log_likelihood + h_c_w_distance

        metrics = dict(w_distance=w_distance, c_w_distance=c_w_distance)

        return obj, g_obj, h_obj, metrics

    def objective_kwargs(self, t, mask_spikes, stim=None, n_samples_fr=100, newton_kwargs_critic=None):

        n_kappa = 0
        n_eta = 0 if self.eta is None else self.eta.nbasis

        X_te = np.zeros(mask_spikes.shape + (1 + n_kappa + n_eta,))
        X_te[:, :, 0] = -1.

        if self.eta is not None:
            args = np.where(shift_array(mask_spikes, 1, fill_value=False))
            t_spk = (t[args[0]],) + args[1:]
            n_eta = self.eta.nbasis
            X_eta = self.eta.convolve_basis_discrete(t, t_spk, shape=mask_spikes.shape)
            X_te[:, :, n_kappa + 1:] = -X_eta

        obj_kwargs = dict(dt=get_dt(t), X_te=X_te, mask_spikes_te=mask_spikes, n_samples_fr=n_samples_fr,
                          newton_kwargs_critic=newton_kwargs_critic)

        return obj_kwargs

    def fit(self, t, mask_spikes, stim=None, newton_kwargs=None, verbose=False, n_samples_fr=100,
            newton_kwargs_critic=None):

        newton_kwargs = {} if newton_kwargs is None else newton_kwargs

        objective_kwargs = self.objective_kwargs(t, mask_spikes, n_samples_fr=n_samples_fr,
                                                 newton_kwargs_critic=newton_kwargs_critic)
        gh_objective = partial(self.gh_objective, **objective_kwargs)

        optimizer = NewtonMethod(model=self, gh_objective=gh_objective, verbose=verbose, **newton_kwargs)
        optimizer.optimize()

        return optimizer
