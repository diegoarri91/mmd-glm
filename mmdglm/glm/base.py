import pickle

import numpy as np
import torch

from ..utils import get_timestep, shift_array


class GLM:

    r"""Point process autoregressive GLM"""
    
    def __init__(self, bias=0, stim=None, hist=None):
        self.bias = float(bias)
        self.stim = stim
        self.hist = hist

    def sample(self, t, stim=None, shape=(), full_output=False):
        """Samples from the model at the given times"""

        dt = get_timestep(t)
        
        stim_shape = () if stim is None else stim.shape[1:]
        shape = (len(t), ) + stim_shape + shape
            
        log_lam = torch.full(shape, self.bias)
        eta_conv = torch.zeros(shape)
        mask_spikes = torch.zeros(shape, dtype=bool)

        if stim is not None:
            kappa_conv = self.stim.convolve_continuous(t, stim)
            kappa_conv = np.pad(kappa_conv, ((1, 0),) + ((0, 0),) * (stim.ndim - 1))[:-1]
            kappa_conv = np.expand_dims(kappa_conv, axis=tuple(range(stim.ndim, len(shape))))
            log_lam = log_lam + kappa_conv
        else:
            kappa_conv = None

        if self.hist:
            for j, _ in enumerate(t):

                log_lam[j] = log_lam[j] + eta_conv[j]
                lam = torch.exp(log_lam[j])
                p_spk = 1 - torch.exp(-lam * dt)

                rand = torch.rand(*shape[1:])
                mask_spikes[j] = p_spk > rand

                if torch.any(mask_spikes[j]) and j < len(t) - 1:
                    eta_values = self.hist.evaluate(t[j + 1:] - t[j + 1])
                    eta_conv[j + 1:, mask_spikes[j]] += eta_values[:, None]
        else:
            lam = torch.exp(log_lam)
            p_spk = 1 - np.exp(-lam * dt)
            rand = torch.rand(*shape)
            mask_spikes = p_spk > rand
        
        if full_output:
            return kappa_conv, eta_conv, log_lam, mask_spikes
        else:
            return log_lam, mask_spikes

    def sample_conditioned(self, t, mask_spikes, stim=None, full_output=False):
        """Returns the intensity predicted by the model using the given spike times"""
        shape = mask_spikes.shape
        dt = get_dt(t)
        
        u = np.full(shape, self.bias)
        arg_spikes = np.where(shift_array(mask_spikes, 1, fill_value=False))
        t_spikes = (t[arg_spikes[0]], ) + arg_spikes[1:]
        
        if stim is not None:
            assert shape[:stim.ndim] == stim.shape
            kappa_conv = self.stim.convolve_continuous(t, stim)
            kappa_conv = np.pad(kappa_conv, ((1, 0),) + ((0, 0),) * (stim.ndim - 1))[:-1]
            kappa_conv = np.expand_dims(kappa_conv, axis=tuple(range(stim.ndim, len(shape))))
            u = u + kappa_conv
        else:
            kappa_conv = None
            
        if self.hist is not None and len(t_spikes[0]) > 0:
            eta_conv = self.hist.convolve_discrete(t, t_spikes, shape=shape[1:])
            u = u + eta_conv
        else:
            eta_conv = np.zeros(shape)

        r = np.exp(u)

        if full_output:
            return kappa_conv, eta_conv, u, r
        else:
            return u, r
    
    def get_params(self):
        
        n_kappa = 0 if self.stim is None else self.stim.nbasis
        n_eta = 0 if self.hist is None else self.hist.nbasis
        theta = np.zeros(1 + n_kappa + n_eta)
        
        theta[0] = self.bias
        if self.stim is not None:
            theta[1:1 + n_kappa] = self.stim.coefs
        if self.hist is not None:
            theta[1 + n_kappa:] = self.hist.coefs
            
        return theta

    def set_params(self, theta):
        
        n_kappa = 0 if self.stim is None else self.stim.nbasis
        
        self.bias = theta[0]
        if self.stim is not None:
            self.stim.coefs = theta[1:1 + n_kappa]
        if self.hist is not None:
            self.hist.coefs = theta[1 + n_kappa:]
            
        return self

    def likelihood_kwargs(self, t, mask_spikes, stim=None):
        """Returns arguments to compute the likelihood function"""
        n_kappa = 0 if self.stim is None else self.stim.nbasis
        n_eta = 0 if self.hist is None else self.hist.nbasis

        X = np.zeros(mask_spikes.shape + (1 + n_kappa + n_eta, ))
        X[..., 0] = 1

        if stim is not None:
            X_kappa = self.stim.convolve_basis_continuous(t, stim)
            X_kappa = np.pad(X_kappa, ((1, 0),) + ((0, 0),) * (X.ndim - 1))[:-1]
            X_kappa = np.expand_dims(X_kappa, axis=tuple(range(X_kappa.ndim, X.ndim)))
            X[..., 1:1 + n_kappa] = X_kappa
        
        if self.hist is not None:
            args = np.where(shift_array(mask_spikes, 1, fill_value=False))
            t_spk = (t[args[0]], ) + args[1:]
            X_eta = self.hist.convolve_basis_discrete(t, t_spk, shape=mask_spikes.shape)
            X[..., 1 + n_kappa:] = X_eta

        likelihood_kwargs = dict(dt=get_dt(t), X=X, mask_spikes=mask_spikes)

        return likelihood_kwargs

    def copy(self):
        return self.__class__(b=self.bias, stim=self.stim.copy(), hist=self.hist.copy())

    def save(self, path):
        params = dict(b=self.bias, stim=self.stim, hist=self.hist)
        with open(path, "wb") as fit_file:
            pickle.dump(params, fit_file)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as fit_file:
            params = pickle.load(fit_file)
        glm = cls(b=params['b'], stim=params['stim'], hist=params['hist'])
        return glm
