from functools import partial
import pickle
# import sys
# sys.path.append("/home/diego/Dropbox/hold_noise/iclamp-glm/")

import numpy as np
from sklearn.metrics import recall_score

# from icglm.models.base import BayesianSpikingModel
from .base import GLM
from gglm.optimization import NewtonMethod

from ..utils import get_dt, shift_array
# from icglm.masks import shift_mask
# from icglm.utils.time import get_dt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class AdversarialGLM(GLM):

    def __init__(self, u0, eta, discriminator, discriminator_features=None, c2=True, mu=None, sd=None):
        self.u0 = u0
        self.eta = eta
        self.discriminator = discriminator
        self.discriminator_features = discriminator_features
        self.mu = mu
        self.sd = sd
#         self.log_likelihood_iterations_d = []
        self.c_iterations = []
        self.discriminator_recall_fr = []
        self.discriminator_recall_te = []
        self.c2 = c2

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

    def use_prior_kernels(self):
        return False

    def gh_log_likelihood_r_sum(self, dt, X=None, mask_spikes=None, newton_kwargs_d=None):

        theta = self.get_params()
        X_te = X.copy()
        u_te = np.einsum('ijk,k->ij', X_te, theta)
        r_te = np.exp(u_te)
        mask_spikes_te = mask_spikes.copy() 
        n_te = mask_spikes_te.shape[1]
        
        t = np.arange(mask_spikes_te.shape[0]) * dt
        n_fr = 100
#         stim = np.zeros((len(t), n_fr))
  
        n_discriminator_iterations, lr = newton_kwargs_d['max_iterations'], newton_kwargs_d['learning_rate']
        lam_ml, lam_d = newton_kwargs_d['lam_ml'], newton_kwargs_d['lam_d']
        standardize = newton_kwargs_d['standardize']
        warm_up_iterations = newton_kwargs_d['warm_up_iterations']

#         aglm = AdversarialGLM(u0=theta[0], eta=self.eta.copy(), discriminator=None)
#         aglm.eta.coefs = theta[1:]
        u_fr, r_fr, mask_spikes_fr = self.sample(t, shape=(n_fr,))
        X_fr = self.get_likelihood_kwargs(t, np.zeros(mask_spikes_fr.shape), mask_spikes_fr)['X']
#         print(mask_spikes_te.shape, mask_spikes_fr.shape)
        
        r = np.concatenate((r_te, r_fr), axis=1)
        mask_discriminator = np.concatenate((mask_spikes_te.copy(), mask_spikes_fr.copy()), axis=1)
        y = np.concatenate((np.ones(n_te), np.zeros(n_fr)))
        
#         X_discriminator = np.sum(r, 0)[:, None]
        X_discriminator = self.get_X_discriminator(r, mask_discriminator)
        
        if standardize:
            mu, sd = np.mean(X_discriminator, 0), np.std(X_discriminator, 0)
            X_discriminator = (X_discriminator - mu) / sd
        else:
            sd = 1
            
        theta_d = self.discriminator.get_theta()
#         y_proba = self.discriminator.predict_proba(t, np.zeros(mask_discriminator.shape), 
#                                                   mask_discriminator, X_discriminator)
#         y_pred = y_proba > 0.5
#         print('\n accuracy_score before', np.mean(y == y_pred), self.discriminator.gh_log_likelihood(theta_d, dt, X_discriminator, mask_discriminator, y)[0])
        
        if len(self.c_iterations) > warm_up_iterations:
            for k in range(n_discriminator_iterations):
                theta_d = self.discriminator.update_theta(theta_d, dt, X_discriminator, mask_discriminator, y, lr)
            
#         print(X_discriminator.shape, mask_discriminator.shape, y.shape)
        log_l_d, _ = self.discriminator.gh_log_likelihood(theta_d, dt, X_discriminator, mask_discriminator, y)
        self.discriminator.log_likelihood_iterations.append(log_l_d)
        self.discriminator = self.discriminator.set_params(theta_d)
        
        y_proba = self.discriminator.predict_proba(t, np.zeros(mask_discriminator.shape), 
                                                  mask_discriminator, X_discriminator)
        y_pred = y_proba > 0.5
        self.discriminator_recall_te.append(recall_score(y, y_pred, pos_label=1))
        self.discriminator_recall_fr.append(recall_score(y, y_pred, pos_label=0))
#         print('\n', np.mean(X_discriminator[:n_te]), np.mean(X_discriminator[n_te:]))
#         print('accuracy_score after', np.mean(y == y_pred))

        c_pf, g_pf, h_pf = self.gh_discriminator_terms(t, X_discriminator, mask_spikes_fr, mask_spikes_te, X_fr, X_te, r_fr, r_te, n_te)
        
        log_likelihood = lam_ml * (np.sum(u_te[mask_spikes_te]) - dt * np.sum(r_te)) + lam_d * c_pf
        g_log_likelihood = lam_ml * (np.sum(X_te[mask_spikes_te, :], axis=0) - dt * np.einsum('ijk,ij->k', X_te, r_te)) + lam_d * g_pf
        h_log_likelihood = lam_ml *(- dt * np.einsum('ijk,ij,ijl->kl', X_te, r_te, X_te)) + lam_d * h_pf
        
        self.c_iterations.append(lam_d * c_pf)

        return log_likelihood, g_log_likelihood, h_log_likelihood, None
    
    def gh_discriminator_terms(self, t, X_discriminator, mask_spikes_fr, mask_spikes_te, X_fr, X_te, r_fr, r_te, n_te):
        eps = 1e-20
        theta_d = self.discriminator.get_theta()
        y_proba_fr = self.discriminator.predict_proba(t, np.zeros(mask_spikes_fr.shape), 
                                                     mask_spikes_fr, X_discriminator[n_te:, :])
        c_pf =  np.sum(np.log(y_proba_fr + eps))
        if self.c2:
            y_proba_te = self.discriminator.predict_proba(t, np.zeros(mask_spikes_te.shape), 
                                                      mask_spikes_te, X_discriminator[:n_te, :])
            c_pf = c_pf + np.sum(np.log(1 - y_proba_te + eps))
        
        ii = 0
        sd = 1
        if 'r_sum' in self.discriminator_features:
            X_r_fr = np.einsum('ijk,ij->jk', X_fr, r_fr)
            X_r_X_fr = np.einsum('ijk,ij,ijl->jkl', X_fr, r_fr, X_fr)
#             X_r_fr = np.array([np.einsum('tk,t->k', X_fr[:, sw, :], r_fr[:, sw]) for sw in range(X_fr.shape[1])])
#             X_r_X_fr = np.array([np.einsum('tk,t,tl->kl', X_fr[:, sw, :], r_fr[:, sw], X_fr[:, sw, :]) for sw in range(X_fr.shape[1])])

            g_c_pf = theta_d[ii + 1] / sd * np.einsum('j,jk->k', 1 - y_proba_fr,  X_r_fr)
            h_c_pf = theta_d[ii + 1] / sd * np.einsum('j,jkl->kl', 1 - y_proba_fr,  X_r_X_fr) -\
                     theta_d[ii + 1]**2 / sd**2 * np.einsum('j,jkl->kl', y_proba_fr * (1 - y_proba_fr), np.einsum('jk,jl->jkl', X_r_fr, X_r_fr))
#             print(self.c2)
            if self.c2:
                X_r_te = np.einsum('ijk,ij->jk', X_te, r_te)
                X_r_X_te = np.einsum('ijk,ij,ijl->jkl', X_te, r_te, X_te)

                g_c_pf = g_c_pf + theta_d[ii + 1] / sd * np.einsum('j,jk->k', y_proba_te,  X_r_te)        
                h_c_pf = h_c_pf + theta_d[ii + 1] / sd * np.einsum('j,jkl->kl', y_proba_te,  X_r_X_te) +\
                          theta_d[ii + 1]**2 / sd**2 * np.einsum('j,jkl->kl', y_proba_te * (1 - y_proba_te), np.einsum('jk,jl->jkl', X_r_te, X_r_te))
            ii += 1
        if 'r_spk' in self.discriminator_features:
            X_r_fr_spk = np.array([np.einsum('tk,t->k', X_fr[mask_spikes_fr[:, sw], sw, :], r_fr[mask_spikes_fr[:, sw], sw]) for sw in range(X_fr.shape[1])])
            X_r_X_fr_spk = np.array([np.einsum('tk,t,tl->kl', X_fr[mask_spikes_fr[:, sw], sw, :], r_fr[mask_spikes_fr[:, sw], sw], X_fr[mask_spikes_fr[:, sw], sw, :]) for sw in range(X_fr.shape[1])])

            g_c_pf = theta_d[ii + 1] / sd * np.einsum('j,jk->k', 1 - y_proba_fr,  X_r_fr_spk)
            h_c_pf = theta_d[ii + 1] / sd * np.einsum('j,jkl->kl', 1 - y_proba_fr,  X_r_X_fr_spk) -\
                     theta_d[ii + 1]**2 / sd**2 * np.einsum('j,jkl->kl', y_proba_fr * (1 - y_proba_fr), np.einsum('jk,jl->jkl', X_r_fr_spk, X_r_fr_spk))
            if self.c2:
                X_r_te_spk = np.array([np.einsum('tk,t->k', X_te[mask_spikes_te[:, sw], sw, :], r_te[mask_spikes_te[:, sw], sw]) for sw in range(X_te.shape[1])])
                X_r_X_te_spk = np.array([np.einsum('tk,t,tl->kl', X_te[mask_spikes_te[:, sw], sw, :], r_te[mask_spikes_te[:, sw], sw], X_te[mask_spikes_te[:, sw], sw, :]) for sw in range(X_te.shape[1])])
                
                g_c_pf = g_c_pf + theta_d[ii + 1] / sd * np.einsum('j,jk->k', y_proba_te,  X_r_te_spk)  
                h_c_pf = h_c_pf + theta_d[ii + 1] / sd * np.einsum('j,jkl->kl', y_proba_te,  X_r_X_te_spk) +\
                                  theta_d[ii + 1]**2 / sd**2 * np.einsum('j,jkl->kl', y_proba_te * (1 - y_proba_te), np.einsum('jk,jl->jkl', X_r_te_spk, X_r_te_spk))
            ii += 1
        
        return c_pf, g_c_pf, h_c_pf
    
    def get_X_discriminator(self, r, mask_spikes):
        X_discriminator = np.zeros((mask_spikes.shape[1], len(self.discriminator_features)))
        ii = 0
        if 'r_sum' in self.discriminator_features:
            X_discriminator[:, ii] = np.sum(r, 0)
            ii += 1
        if 'r_spk' in self.discriminator_features:
            X_discriminator[:, ii] = np.array([np.sum(r[mask_spikes[:, sw], sw]) for sw in range(mask_spikes.shape[1])])
            ii += 1
        return X_discriminator

    def get_likelihood_kwargs(self, t, stim, mask_spikes, stim_h=0, newton_kwargs_d=None):

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

#         theta0 = self.get_params()
        likelihood_kwargs = self.get_likelihood_kwargs(t, stim, mask_spikes, stim_h=stim_h, newton_kwargs_d=newton_kwargs_d)

#         gh_log_prior = None if not(self.use_prior_kernels()) else self.gh_log_prior_kernels
        if discriminator == 'r_sum':
            gh_log_likelihood = partial(self.gh_log_likelihood_r_sum, **likelihood_kwargs)
        else:
            gh_log_likelihood = partial(self.gh_log_likelihood_kernels, **likelihood_kwargs)

        optimizer = NewtonMethod(model=self, gh_objective=gh_log_likelihood, verbose=verbose, **newton_kwargs)
        optimizer.optimize()

#         theta = optimizer.theta_iterations[:, -1]
#         self.set_params(theta)

        return optimizer
    