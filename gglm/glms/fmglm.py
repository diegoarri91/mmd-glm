import numpy as np

from .baseglm import GLM


class FeatureMatchingGLM(GLM):

    def __init__(self, u0, eta):
        super().__init__(u0=u0, eta=eta)

    def gh_objective(self, dt, X=None, mask_spikes=None):

        theta = self.get_params()
        X_te = X.copy()
        u_te = np.einsum('ijk,k->ij', X_te, theta)
        u_te_mean = np.mean(u_te, 1)
        # r_te = np.exp(u_te)
        # mask_spikes_te = mask_spikes.copy()
        # n_te = mask_spikes_te.shape[1]

        t = np.arange(u_te.shape[0]) * dt
        n_fr = 100
        stim = np.zeros((len(t), n_fr))

        # n_discriminator_iterations, lr = newton_kwargs_d['max_iterations'], newton_kwargs_d['learning_rate']
        # lam_ml, lam_d = newton_kwargs_d['lam_ml'], newton_kwargs_d['lam_d']
        # standardize = newton_kwargs_d['standardize']
        # warm_up_iterations = newton_kwargs_d['warm_up_iterations']

        u_fr, r_fr, mask_spikes_fr = self.sample(t, stim)
        X_fr = self.get_likelihood_kwargs(t, np.zeros(mask_spikes_fr.shape), mask_spikes_fr)['X']

        # r = np.concatenate((r_te, r_fr), axis=1)
        # mask_discriminator = np.concatenate((mask_spikes_te.copy(), mask_spikes_fr.copy()), axis=1)
        # y = np.concatenate((np.ones(n_te), np.zeros(n_fr)))

        obj, g_obj, h_obj = self.gh_log_likelihood()