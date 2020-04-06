from functools import partial
import pickle
import sys
sys.path.append("/home/diego/Dropbox/hold_noise/iclamp-glm/")

import numpy as np

# from icglm.models.base import BayesianSpikingModel
from optimization import NewtonMethod
from icglm.masks import shift_mask
from icglm.utils.time import get_dt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class AdversarialGLM:

    def __init__(self, u0, eta, discriminator, mu=None, sd=None):
        self.u0 = u0
        self.eta = eta
        self.discriminator = discriminator
        self.mu = mu
        self.sd = sd
#         self.log_likelihood_iterations_d = []
        self.c_iterations = []
        self.discriminator_accuracy = []

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

        v = - eta_conv - self.u0
        r = np.exp(v)

        if full:
            return eta_conv, v, r
        else:
            return v, r

    def use_prior_kernels(self):
        return False

    def gh_log_likelihood_r_sum(self, theta, dt, X=None, mask_spikes=None, newton_kwargs_d=None):
#         u_new_te, r_new_te = aglm.simulate_subthreshold(t, stim_naive, mask_spikes_te)
        X_te = X.copy()
        u_te = np.einsum('ijk,k->ij', X_te, theta)
        r_te = np.exp(u_te)
        
        mask_spikes_te = mask_spikes.copy() 
        n_samples = mask_spikes_te.shape[1]
        
        t = np.arange(mask_spikes_te.shape[0]) * dt
        n = 100
  
        n_discriminator_iterations, lr = newton_kwargs_d['max_iterations'], newton_kwargs_d['learning_rate']
        lam_ml, lam_d = newton_kwargs_d['lam_ml'], newton_kwargs_d['lam_d']
        standardize = newton_kwargs_d['standardize']
        warm_up_iterations = newton_kwargs_d['warm_up_iterations']

#         u_fr, r_fr, mask_spikes_fr = self.sample(t, np.zeros((len(t), n)))
        aux = AdversarialGLM(u0=theta[0], eta=self.eta.copy(), discriminator=None)
        aux.eta.coefs = theta[1:]
        u_fr, r_fr, mask_spikes_fr = aux.sample(t, np.zeros((len(t), n)))
        dic_fr = self.get_likelihood_kwargs(t, np.zeros(mask_spikes_fr.shape), mask_spikes_fr)        
        X_fr = dic_fr['X']
        
        r = np.concatenate((r_te, r_fr), axis=1)

        mask_discriminator = np.concatenate((mask_spikes_te.copy(), mask_spikes_fr.copy()), axis=1)
        y = np.concatenate((np.ones(n_samples), np.zeros(n)))
        X_discriminator = np.sum(r, 0)[:, None]
#         print('\n', np.mean(X_discriminator[:n_samples]), np.mean(X_discriminator[n_samples:]))
#         print(np.any(np.isnan(X_discriminator)))
        if standardize:
            mu, sd = np.mean(X_discriminator, 0), np.std(X_discriminator, 0)
            X_discriminator = (X_discriminator - mu) / sd
        else:
            sd = 1
            
        theta_d = self.discriminator.get_theta()
        y_proba = self.discriminator.predict_proba(t, np.zeros(mask_discriminator.shape), 
                                                  mask_discriminator, X_discriminator)
        y_pred = y_proba > 0.5
#         print('\n accuracy_score before', np.mean(y == y_pred), self.discriminator.gh_log_likelihood(theta_d, dt, X_discriminator, mask_discriminator, y)[0])
        
        if len(self.c_iterations) > warm_up_iterations:
            for k in range(n_discriminator_iterations):
                theta_d = self.discriminator.update_theta(theta_d, dt, X_discriminator, mask_discriminator, y, lr)
        log_l_d, _ = self.discriminator.gh_log_likelihood(theta_d, dt, X_discriminator, mask_discriminator, y)
        self.discriminator.log_likelihood_iterations.append(log_l_d)
        self.discriminator = self.discriminator.set_params(theta_d)
        
        y_proba = self.discriminator.predict_proba(t, np.zeros(mask_discriminator.shape), 
                                                  mask_discriminator, X_discriminator)
        y_pred = y_proba > 0.5
        self.discriminator_accuracy.append(np.mean(y == y_pred))
#         print('accuracy_score after', np.mean(y == y_pred))
        
        y_proba_fr = self.discriminator.predict_proba(t, np.zeros(mask_spikes_fr.shape), 
                                                  mask_spikes_fr, X_discriminator[n_samples:, :])
        eps = 1e-20

        X_r_fr = np.einsum('ijk,ij->jk', X_fr, r_fr)
        X_r_X_fr = np.einsum('ijk,ij,ijl->jkl', X_fr, r_fr, X_fr)
        c1_pf =  np.sum(np.log(y_proba_fr + eps))
        g_c1_pf = theta_d[1] / sd * np.einsum('j,jk->k', 1 - y_proba_fr,  X_r_fr)
        h_c1_pf = theta_d[1] / sd * np.einsum('j,jkl->kl', 1 - y_proba_fr,  X_r_X_fr) -\
                  theta_d[1]**2 / sd**2 * np.einsum('j,jkl->kl', y_proba_fr * (1 - y_proba_fr), np.einsum('jk,jl->jkl', X_r_fr, X_r_fr))
        
        y_proba_te = self.discriminator.predict_proba(t, np.zeros(mask_spikes_te.shape), 
                                                  mask_spikes_te, X_discriminator[:n_samples, :])
        X_r_te = np.einsum('ijk,ij->jk', X_te, r_te)
        X_r_X_te = np.einsum('ijk,ij,ijl->jkl', X_te, r_te, X_te)
        c2_pf =  np.sum(np.log(1 - y_proba_te + eps))
        g_c2_pf = theta_d[1] / sd * np.einsum('j,jk->k', y_proba_te,  X_r_te)        
        h_c2_pf = theta_d[1] / sd * np.einsum('j,jkl->kl', y_proba_te,  X_r_X_te) +\
                  theta_d[1]**2 / sd**2 * np.einsum('j,jkl->kl', y_proba_te * (1 - y_proba_te), np.einsum('jk,jl->jkl', X_r_te, X_r_te))
        
        c_pf = c1_pf + c2_pf
        g_pf = g_c1_pf + g_c2_pf
        h_pf = g_c1_pf + g_c2_pf

#         c_pf = c1_pf
#         g_pf = g_c1_pf
#         h_pf = h_c1_pf
#         print('hola', X_te.shape, mask_spikes_te.shape, X_te[mask_spikes_te, :].shape)
        log_likelihood = lam_ml * (np.sum(u_te[mask_spikes_te]) - dt * np.sum(r_te)) + lam_d * c_pf
        g_log_likelihood = lam_ml * (np.sum(X_te[mask_spikes_te, :], axis=0) - dt * np.einsum('ijk,ij->k', X_te, r_te)) + lam_d * g_pf
        h_log_likelihood = lam_ml *(- dt * np.einsum('ijk,ij,ijl->kl', X_te, r_te, X_te)) + lam_d * h_pf
        
        self.c_iterations.append(lam_d * c_pf)

        return log_likelihood, g_log_likelihood, h_log_likelihood

#     def gh_log_likelihood_kernels(self, theta, dt, X=None, mask_spikes=None, newton_kwargs_d=None):
#         u_te = np.einsum('ijk,k->ij', X, theta)
#         r_te = np.exp(u_te)
#         X_te = X.copy()
#         mask_spikes_te = mask_spikes.copy() 
#         n_samples = mask_spikes_te.shape[1]
        
#         t = np.arange(mask_spikes_te.shape[0]) * dt
#         n = 100
  
#         n_discriminator_iterations, lr = newton_kwargs_d['max_iterations'], newton_kwargs_d['learning_rate']
#         lam_ml, lam_d = newton_kwargs_d['lam_ml'], newton_kwargs_d['lam_d']

#         u_fr, r_fr, mask_spikes_fr = self.sample(t, np.zeros((len(t), n)))

#         mask_discriminator = np.concatenate((mask_spikes_te.copy(), mask_spikes_fr.copy()), axis=1)
#         y = np.concatenate((np.ones(n_samples), np.zeros(n)))
#         X_discriminator = np.zeros((n_samples + n, 1))
#         X_discriminator[:n_samples, 0] = np.array([np.mean(u_te[mask_spikes_te[:, sw], sw]) for sw in range(n_samples)])
#         X_discriminator[n_samples:, 0] = np.array([np.mean(u_fr[mask_spikes_fr[:, sw], sw]) for sw in range(n)])
# #         sd = np.std(X_discriminator, 0)
# #         X_discriminator = (X_discriminator - np.mean(X_discriminator, 0)) / sd

#         theta_d = self.discriminator.get_theta()
#         y_proba = self.discriminator.predict_proba(t, np.zeros(mask_discriminator.shape), 
#                                                   mask_discriminator, X_discriminator)
#         y_pred = y_proba > 0.5
# #         print('\n accuracy_score before', np.mean(y == y_pred), self.discriminator.gh_log_likelihood(theta_d, dt, X_discriminator, mask_discriminator, y)[0])
        
#         for k in range(n_discriminator_iterations):
#             theta_d = self.discriminator.update_theta(theta_d, dt, X_discriminator, mask_discriminator, y, lr)
        
#         self.discriminator = self.discriminator.set_params(theta_d)
        
#         y_proba = self.discriminator.predict_proba(t, np.zeros(mask_discriminator.shape), 
#                                                   mask_discriminator, X_discriminator)
#         y_pred = y_proba > 0.5
# #         print('accuracy_score after', np.mean(y == y_pred))
        
#         y_proba = self.discriminator.predict_proba(t, np.zeros(mask_spikes_fr.shape), 
#                                                   mask_spikes_fr, X_discriminator[n_samples:, :])
#         eps = 1e-20
#         dic_fr = self.get_likelihood_kwargs(t, np.zeros(mask_spikes_fr.shape), mask_spikes_fr)        
#         X_fr = dic_fr['X']
#         X_sum_fr = np.array([np.mean(X_fr[mask_spikes_fr[:, ii], ii, :], 0) for ii in range(mask_spikes_fr.shape[1])])
#         c1_pf =  np.sum(np.log(y_proba + eps))
#         g_c1_pf = theta_d[1] * np.einsum('j,jk->k', (1 - y_proba),  X_sum_fr)        
#         h_c1_pf = -theta_d[1]**2 * np.einsum('j,jkl->kl', y_proba * (1 - y_proba), np.einsum('jk,jl->jkl', X_sum_fr, X_sum_fr))
        
#         y_proba = self.discriminator.predict_proba(t, np.zeros(mask_spikes_te.shape), 
#                                                   mask_spikes_te, X_discriminator[:n_samples, :])
#         X_sum_te = np.array([np.mean(X_te[mask_spikes_te[:, ii], ii, :], 0) for ii in range(n_samples)])
#         c2_pf =  np.sum(np.log(1 - y_proba + eps))
#         g_c2_pf = theta_d[1] * np.einsum('j,jk->k', y_proba,  X_sum_te)        
#         h_c2_pf = theta_d[1]**2 * np.einsum('j,jkl->kl', y_proba * (1 - y_proba), np.einsum('jk,jl->jkl', X_sum_te, X_sum_te))
        
# #         log_likelihood = lam_ml * (np.sum(u_te[mask_spikes_te]) - dt * np.sum(r_te)) + lam_d * c1_pf
# #         g_log_likelihood = lam_ml * (np.sum(X_te[mask_spikes_te, :], axis=0) - dt * np.einsum('ijk,ij->k', X_te, r_te)) + lam_d * g_c1_pf
# #         h_log_likelihood = lam_ml *(- dt * np.einsum('ijk,ij,ijl->kl', X_te, r_te, X_te)) + lam_d * h_c1_pf
        
#         log_likelihood = lam_ml * (np.sum(u_te[mask_spikes_te]) - dt * np.sum(r_te)) + lam_d * (c1_pf + c2_pf)
#         g_log_likelihood = lam_ml * (np.sum(X_te[mask_spikes_te, :], axis=0) - dt * np.einsum('ijk,ij->k', X_te, r_te)) + lam_d * (g_c1_pf + g_c2_pf)
#         h_log_likelihood = lam_ml *(- dt * np.einsum('ijk,ij,ijl->kl', X_te, r_te, X_te)) + lam_d * (h_c1_pf + h_c2_pf)
        
#         self.c_iterations.append(lam_d * (c1_pf + c2_pf))

#         return log_likelihood, g_log_likelihood, h_log_likelihood
    
#     def gh_log_likelihood_kernels2(self, theta, dt, X=None, mask_spikes=None, newton_kwargs_d=None):
#         u = np.einsum('ijk,k->ij', X, theta)
#         r = np.exp(u)
#         n_samples = mask_spikes.shape[1]
        
#         t = np.arange(mask_spikes.shape[0]) * dt
#         theta_d = self.discriminator.get_theta()
  
#         n = 100
#         n_discriminator_iterations, lr = newton_kwargs_d['max_iterations'], newton_kwargs_d['learning_rate']
#         u_fr, r_fr, mask_spikes_fr = self.sample(t, np.zeros((len(t), n)))

# #         print(X_fr.shape)
#         mask_discriminator = np.concatenate((mask_spikes[...], np.zeros((len(t), n))), axis=1)
#         mask_discriminator[:, n_samples:] = mask_spikes_fr[...]            
#         y = np.concatenate((np.ones(n_samples), np.zeros(n)))
#         X_discriminator = np.zeros((n_samples + n, 1))
#         X_discriminator[:n_samples, 0] = np.sum(r, 0)
#         X_discriminator[n_samples:, 0] = np.sum(r_fr, 0)
#         X_discriminator = (X_discriminator - np.mean(X_discriminator, 0)) / np.std(X_discriminator, 0)
# #         X_discriminator = (X_discriminator - self.mu) / self.sd

#         y_pred = self.discriminator.predict_proba(t, np.zeros(mask_discriminator.shape), 
#                                                   mask_discriminator, X_discriminator)
#         print('\n accuracy_score before', np.sum(y == (y_pred > 0.5)) / len(y), 
#               self.discriminator.gh_log_likelihood(theta_d, dt, X_discriminator, mask_discriminator, y)[0])
        
#         dic_fr = self.get_likelihood_kwargs(t, np.zeros(mask_spikes_fr.shape), mask_spikes_fr)        
#         X_fr = np.concatenate((X, dic_fr['X']), axis=1)
#         r_fr = np.concatenate((r, r_fr), axis=1)
        
#         for k in range(n_discriminator_iterations):
            
# #             u_fr, r_fr, mask_spikes_fr = self.sample(t, np.zeros(mask_spikes.shape))
# #             mask_discriminator[:, n_samples:] = mask_spikes_fr[...]            
# #             X_discriminator = np.zeros((2 * n_samples, 1))
# #             X_discriminator[:n_samples, 0] = np.sum(r, 0)
# #             X_discriminator[n_samples:, 0] = np.sum(r_fr, 0)
# #             X_discriminator = (X_discriminator - np.mean(X_discriminator, 0)) / np.std(X_discriminator, 0)
            
# #             log_likelihood_d, g_log_likelihood_d = self.discriminator.gh_log_likelihood(theta_d, dt, X_discriminator, mask_discriminator, y)
# #             g_log_likelihood_d[np.abs(g_log_likelihood_d) >1e3] = np.sign(g_log_likelihood_d[np.abs(g_log_likelihood_d) > 1e3]) * 1e3
# #             theta_d = theta_d + lr * g_log_likelihood_d
            
#             theta_d = self.discriminator.update_theta(theta_d, dt, X_discriminator, mask_discriminator, y, lr)
        
#         self.discriminator = self.discriminator.set_params(theta_d)
        
# #         u_fr, r_fr, mask_spikes_fr = self.sample(t, np.zeros((len(t), n)))
        
# #         dic_fr = self.get_likelihood_kwargs(t, np.zeros(mask_spikes_fr.shape), mask_spikes_fr)
# #         X_fr = dic_fr['X']
        
# #         X_discriminator = np.sum(r_fr, 0)[:, None]
# #         X_discriminator = (X_discriminator - self.mu) / self.sd
# #         y_pred = self.discriminator.predict_proba(t, np.zeros(mask_spikes_fr.shape), mask_spikes_fr, X_discriminator)    
        
# #         X_fr, r_fr = X.copy(), r.copy()
# #         X_discriminator = np.sum(r, 0)[:, None]
# #         mu, sd = np.mean(X_discriminator, 0), np.std(X_discriminator, 0)
# #         X_discriminator = (X_discriminator - self.mu) / self.sd
# #         y_pred = self.discriminator.predict_proba(t, np.zeros(mask_spikes.shape), mask_spikes, X_discriminator)
# #         print((y_pred > 0.5))

#         y_pred = self.discriminator.predict_proba(t, np.zeros(mask_discriminator.shape), 
#                                                   mask_discriminator, X_discriminator)
    
#         print('accuracy_score after', np.sum(y == (y_pred > 0.5)) / len(y), 
#              self.discriminator.gh_log_likelihood(theta_d, dt, X_discriminator, mask_discriminator, y)[0])
        
#         eps = 1e-20
#         reg = 1e0
#         c1_pf =  np.sum((1 - y) * np.log(y_pred + eps))
#         X_r_fr = np.einsum('ijk,ij->jk', X_fr, r_fr)
#         print(X_fr.shape)
#         g_c1_pf = theta_d[1] / self.sd * np.einsum('j,jk->k', 1 - y_pred,  X_r_fr)
#         h_c1_pf = theta_d[1] / self.sd * np.einsum('j,jkl->kl', 1 - y_pred,  np.einsum('ijk,ij,ijl->jkl', X_fr, r_fr, X_fr)) -\
#                   theta_d[1]**2 / self.sd**2 * np.einsum('j,jkl->kl', y_pred * (1 - y_pred), np.einsum('jk,jl->jkl', X_r_fr, X_r_fr))
        
# #         log_likelihood = np.sum(u[mask_spikes]) - dt * np.sum(r) + reg * c1_pf
# #         g_log_likelihood = np.sum(X[mask_spikes, :], axis=0) - dt * np.einsum('ijk,ij->k', X, r) + reg * g_c1_pf
# #         h_log_likelihood = - dt * np.einsum('ijk,ij,ijl->kl', X, r, X) + reg * h_c1_pf
        
#         log_likelihood = reg * c1_pf
#         g_log_likelihood = reg * g_c1_pf
#         h_log_likelihood = reg * h_c1_pf
        
#         self.c_iterations.append(reg * c1_pf)

#         return log_likelihood, g_log_likelihood, h_log_likelihood

    def get_theta(self):
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

        likelihood_kwargs = dict(dt=get_dt(t), X=X, mask_spikes=mask_spikes, newton_kwargs_d=newton_kwargs_d)

        return likelihood_kwargs

    def set_params(self, theta):
        n_kappa = 0
        self.u0 = theta[0]
        self.eta.coefs = theta[n_kappa + 1:]
        return self

    def fit(self, t, stim, mask_spikes, stim_h=0, newton_kwargs=None, verbose=False, newton_kwargs_d=None, 
            discriminator='r_sum'):
        
        newton_kwargs = {} if newton_kwargs is None else newton_kwargs

        theta0 = self.get_theta()
        likelihood_kwargs = self.get_likelihood_kwargs(t, stim, mask_spikes, stim_h=stim_h, newton_kwargs_d=newton_kwargs_d)

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
    