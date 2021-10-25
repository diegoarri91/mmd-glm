import pickle

import numpy as np
import torch
import torch.nn as nn

from ..utils import get_timestep, shift_tensor


class GLM(nn.Module):

    r"""Point process autoregressive GLM"""
    
    def __init__(self, bias, stim_kernel=None, hist_kernel=None):
        super().__init__()
        self.bias = float(bias)
        self.stim_kernel = stim_kernel
        self.hist_kernel = hist_kernel

    def sample(self, t, stim=None, shape=(1,), full_output=False):
        """Samples from the model at the given times"""

        dt = get_timestep(t)
        
        stim_shape = () if stim is None else stim.shape[1:]
        shape = (len(t), ) + stim_shape + shape
            
        log_lam = torch.full(shape, self.bias)
        hist_conv = torch.zeros(shape)
        mask_spikes = torch.zeros(shape, dtype=torch.bool)

        if stim is not None and self.stim_kernel is not None:
            stim_conv = self.stim_kernel(stim, dt=dt)
            stim_conv = shift_tensor(stim_conv, 1, fill_value=0.)
            stim_conv = stim_conv.reshape(stim_conv.shape + (1,) * (len(shape) - stim.ndim))
            log_lam = log_lam + stim_conv
        elif stim is None:
            stim_conv = None
        else:
            raise RuntimeError('A stimulus was passed but no stimulus kernel was defined for the GLM')

        if self.hist_kernel:
            for j, _ in enumerate(t):

                log_lam[j] = log_lam[j] + hist_conv[j]
                lam = torch.exp(log_lam[j])
                p_spk = 1 - torch.exp(-lam * dt)

                rand = torch.rand(*shape[1:])
                mask_spikes[j] = p_spk > rand

                if torch.any(mask_spikes[j]) and j < len(t) - 1:
                    eta_values = self.hist_kernel.evaluate(t[j + 1:] - t[j + 1])
                    hist_conv[j + 1:, mask_spikes[j]] += eta_values[:, None]
        else:
            lam = torch.exp(log_lam)
            p_spk = 1 - np.exp(-lam * dt)
            rand = torch.rand(*shape)
            mask_spikes = p_spk > rand
        
        if full_output:
            return stim_conv, hist_conv, log_lam, mask_spikes
        else:
            return log_lam, mask_spikes

    def log_conditional_intensity(self, t, mask_spikes, stim=None, full_output=False):
        """Returns the intensity predicted by the model using the given spike times"""

        shape = mask_spikes.shape
        dt = get_timestep(t)
        
        log_lam = torch.full(shape, self.bias)
        mask_spikes = shift_tensor(mask_spikes, 1, fill_value=False)
        # arg_spikes = torch.where(shift_tensor(mask_spikes, 1, fill_value=False))
        # t_spikes = (t[arg_spikes[0]], ) + arg_spikes[1:]

        if stim is not None:
            stim_conv = self.stim_kernel(stim, dt=dt)
            stim_conv = shift_tensor(stim_conv, 1, fill_value=0.)
            stim_conv = stim_conv.reshape(stim_conv.shape + (1,) * (len(shape) - stim.ndim))
            log_lam = log_lam + stim_conv
        else:
            stim_conv = None
            
        if self.hist_kernel is not None:# and len(t_spikes[0]) > 0:
            eta_conv = self.hist_kernel(mask_spikes, dt=dt)
            log_lam = log_lam + eta_conv
        else:
            eta_conv = None

        lam = torch.exp(log_lam)

        if full_output:
            return stim_conv, eta_conv, log_lam, lam
        else:
            return log_lam
    
    def get_params(self):
        
        n_kappa = 0 if self.stim_kernel is None else self.stim_kernel.nbasis
        n_eta = 0 if self.hist_kernel is None else self.hist_kernel.nbasis
        theta = np.zeros(1 + n_kappa + n_eta)
        
        theta[0] = self.bias
        if self.stim_kernel is not None:
            theta[1:1 + n_kappa] = self.stim_kernel.coefs
        if self.hist_kernel is not None:
            theta[1 + n_kappa:] = self.hist_kernel.coefs
            
        return theta

    def set_params(self, theta):
        
        n_kappa = 0 if self.stim_kernel is None else self.stim_kernel.nbasis
        
        self.bias = theta[0]
        if self.stim_kernel is not None:
            self.stim_kernel.coefs = theta[1:1 + n_kappa]
        if self.hist_kernel is not None:
            self.hist_kernel.coefs = theta[1 + n_kappa:]
            
        return self

    def likelihood_kwargs(self, t, mask_spikes, stim=None):
        """Returns arguments to compute the likelihood function"""
        n_kappa = 0 if self.stim_kernel is None else self.stim_kernel.nbasis
        n_eta = 0 if self.hist_kernel is None else self.hist_kernel.nbasis

        X = np.zeros(mask_spikes.shape + (1 + n_kappa + n_eta, ))
        X[..., 0] = 1

        if stim is not None:
            X_kappa = self.stim_kernel.convolve_basis_continuous(t, stim)
            X_kappa = np.pad(X_kappa, ((1, 0),) + ((0, 0),) * (X.ndim - 1))[:-1]
            X_kappa = np.expand_dims(X_kappa, axis=tuple(range(X_kappa.ndim, X.ndim)))
            X[..., 1:1 + n_kappa] = X_kappa
        
        if self.hist_kernel is not None:
            args = np.where(shift_tensor(mask_spikes, 1, fill_value=False))
            t_spk = (t[args[0]], ) + args[1:]
            X_eta = self.hist_kernel.convolve_basis_discrete(t, t_spk, shape=mask_spikes.shape)
            X[..., 1 + n_kappa:] = X_eta

        likelihood_kwargs = dict(dt=get_dt(t), X=X, mask_spikes=mask_spikes)

        return likelihood_kwargs

    def clone(self):
        return self.__class__(bias=self.bias, stim_kernel=self.stim_kernel.clone(),
                              hist_kernel=self.hist_kernel.clone())

    def save(self, path):
        params = dict(b=self.bias, stim=self.stim_kernel, hist=self.hist_kernel)
        with open(path, "wb") as fit_file:
            pickle.dump(params, fit_file)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as fit_file:
            params = pickle.load(fit_file)
        glm = cls(b=params['b'], stim=params['stim'], hist=params['hist'])
        return glm
