import numpy as np

from .base import GLM


class FeatureMatchingGLM(GLM):

    def __init__(self, u0, eta):
        super().__init__(u0=u0, eta=eta)

    def gh_objective(self, dt, X_te, mask_spikes, n_samples_fr):

        theta = self.get_params()
        t = np.arange(X_te.shape[0]) * dt

        u_te = np.einsum('tka,a->tk', X_te, theta)
        r_te = np.exp(u_te)

        u_fr, r_fr, mask_spikes_fr = self.sample(t, shape=(n_samples_fr,))
        # X_fr = self.get_likelihood_kwargs(t, np.zeros(mask_spikes_fr.shape), mask_spikes_fr)['X_te']

        log_likelihood = np.sum(u_te[mask_spikes]) - dt * np.sum(r_te)
        g_log_likelihood = np.sum(X_te[mask_spikes, :], axis=0) - dt * np.einsum('tka,tk->a', X_te, r_te)
        h_log_likelihood = - dt * np.einsum('tka,tk,tkb->ab', X_te, r_te, X_te)

        fm_score

        # r = np.concatenate((r_te, r_fr), axis=1)
        # mask_discriminator = np.concatenate((mask_spikes_te.copy(), mask_spikes_fr.copy()), axis=1)
        # y = np.concatenate((np.ones(n_te), np.zeros(n_fr)))

        log_likelihood, g_log_likelihood, h_log_likelihood = self.gh_log_likelihood(dt, X_te, mask_spikes)

        fm_score, g_fm_score, h_fm_score

        metrics = dict(log_likelihood=log_likelihood, fm_score=fm_score)

        return obj, g_obj, h_obj, metrics

