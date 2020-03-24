import pickle
import sys
sys.path.append("/home/diego/Dropbox/hold_noise/iclamp-glm/")

import numpy as np

from icglm.models.base import BayesianSpikingModel
from icglm.masks import shift_mask
from icglm.utils.time import get_dt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class AdversarialGLM(BayesianSpikingModel):

    def __init__(self, u0, eta, discriminator, mu=None, sd=None):
        self.u0 = u0
        self.eta = eta
        self.discriminator = discriminator
        self.mu = mu
        self.sd = sd
#         self.log_likelihood_iterations_d = []
        self.c_iterations = []

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

    def gh_log_likelihood_kernels(self, theta, dt, X=None, mask_spikes=None):
        
        u = np.einsum('ijk,k->ij', X, theta)
        r = np.exp(u)
        n_samples = mask_spikes.shape[1]
        
        t = np.arange(mask_spikes.shape[0]) * dt
        theta_d = self.discriminator.get_theta()
  
        n = 5000
        n_discriminator_iterations, lr = 50, 1e-2
        u_fr, r_fr, mask_spikes_fr = self.sample(t, np.zeros((len(t), n)))

#         print(X_fr.shape)
        mask_discriminator = np.concatenate((mask_spikes[...], np.zeros((len(t), n))), axis=1)
        mask_discriminator[:, n_samples:] = mask_spikes_fr[...]            
        y = np.concatenate((np.ones(n_samples), np.zeros(n)))
        X_discriminator = np.zeros((n_samples + n, 1))
        X_discriminator[:n_samples, 0] = np.sum(r, 0)
        X_discriminator[n_samples:, 0] = np.sum(r_fr, 0)
#         X_discriminator = (X_discriminator - np.mean(X_discriminator, 0)) / np.std(X_discriminator, 0)
        X_discriminator = (X_discriminator - self.mu) / self.sd

        y_pred = self.discriminator.predict_proba(t, np.zeros(mask_discriminator.shape), 
                                                  mask_discriminator, X_discriminator)
        print(np.sum(1 == (y_pred > 0.5)))
        
        dic_fr = self.get_likelihood_kwargs(t, np.zeros(mask_spikes_fr.shape), mask_spikes_fr)        
        X_fr = np.concatenate((X, dic_fr['X']), axis=1)
        r_fr = np.concatenate((r, r_fr), axis=1)
        
        for k in range(n_discriminator_iterations):
            
#             u_fr, r_fr, mask_spikes_fr = self.sample(t, np.zeros(mask_spikes.shape))
#             mask_discriminator[:, n_samples:] = mask_spikes_fr[...]            
#             X_discriminator = np.zeros((2 * n_samples, 1))
#             X_discriminator[:n_samples, 0] = np.sum(r, 0)
#             X_discriminator[n_samples:, 0] = np.sum(r_fr, 0)
#             X_discriminator = (X_discriminator - np.mean(X_discriminator, 0)) / np.std(X_discriminator, 0)
            
#             log_likelihood_d, g_log_likelihood_d = self.discriminator.gh_log_likelihood(theta_d, dt, X_discriminator, mask_discriminator, y)
#             g_log_likelihood_d[np.abs(g_log_likelihood_d) >1e3] = np.sign(g_log_likelihood_d[np.abs(g_log_likelihood_d) > 1e3]) * 1e3
#             theta_d = theta_d + lr * g_log_likelihood_d
            
            theta_d = self.discriminator.update_theta(theta_d, dt, X_discriminator, mask_discriminator, y)
        
        self.discriminator = self.discriminator.set_params(theta_d)
        
#         u_fr, r_fr, mask_spikes_fr = self.sample(t, np.zeros((len(t), n)))
        
#         dic_fr = self.get_likelihood_kwargs(t, np.zeros(mask_spikes_fr.shape), mask_spikes_fr)
#         X_fr = dic_fr['X']
        
#         X_discriminator = np.sum(r_fr, 0)[:, None]
#         X_discriminator = (X_discriminator - self.mu) / self.sd
#         y_pred = self.discriminator.predict_proba(t, np.zeros(mask_spikes_fr.shape), mask_spikes_fr, X_discriminator)    
        
#         X_fr, r_fr = X.copy(), r.copy()
#         X_discriminator = np.sum(r, 0)[:, None]
#         mu, sd = np.mean(X_discriminator, 0), np.std(X_discriminator, 0)
#         X_discriminator = (X_discriminator - self.mu) / self.sd
#         y_pred = self.discriminator.predict_proba(t, np.zeros(mask_spikes.shape), mask_spikes, X_discriminator)
#         print((y_pred > 0.5))

        y_pred = self.discriminator.predict_proba(t, np.zeros(mask_discriminator.shape), 
                                                  mask_discriminator, X_discriminator)
        print(np.sum(1 == (y_pred > 0.5)))
        
        eps = 1e-20
        reg = 1e-5
        c1_pf =  np.sum((1 - y_pred) * np.log(y_pred + eps))
        X_r_fr = np.einsum('ijk,ij->jk', X_fr, r_fr)
        g_c1_pf = theta_d[1] / self.sd * np.einsum('j,jk->k', 1 - y_pred,  X_r_fr)
        h_c1_pf = theta_d[1] / self.sd * np.einsum('j,jkl->kl', 1 - y_pred,  np.einsum('ijk,ij,ijl->jkl', X_fr, r_fr, X_fr)) -\
                  theta_d[1]**2 / self.sd**2 * np.einsum('j,jkl->kl', y_pred * (1 - y_pred), np.einsum('jk,jl->jkl', X_r_fr, X_r_fr))
        
        log_likelihood = np.sum(u[mask_spikes]) - dt * np.sum(r) + reg * c1_pf
        g_log_likelihood = np.sum(X[mask_spikes, :], axis=0) - dt * np.einsum('ijk,ij->k', X, r) + reg * g_c1_pf
        h_log_likelihood = - dt * np.einsum('ijk,ij,ijl->kl', X, r, X) + reg * h_c1_pf
        
        self.c_iterations.append(reg * c1_pf)

        return log_likelihood, g_log_likelihood, h_log_likelihood

    def get_theta(self):
        n_kappa = 0 
        n_eta = 0 if self.eta is None else self.eta.nbasis
        theta = np.zeros((1 + n_kappa + n_eta))
        theta[0] = self.u0
        theta[1 + n_kappa:] = self.eta.coefs
        return theta

    def get_likelihood_kwargs(self, t, stim, mask_spikes, stim_h=0):

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

    def fit(self, t, stim, mask_spikes, stim_h=0, newton_kwargs=None, verbose=False, **kwargs):
        return super().fit(t, stim, mask_spikes, stim_h=stim_h, newton_kwargs=newton_kwargs, verbose=verbose)
    