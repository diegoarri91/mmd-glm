import pickle

import numpy as np
import torch
import torch.nn as nn

from ..metrics import _append_metrics, negative_log_likelihood
from ..utils import get_timestep, shift_tensor


class GLM(nn.Module):

    r"""Point process autoregressive GLM"""
    
    def __init__(self, bias, stim_kernel=None, hist_kernel=None):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor([float(bias)]))
        self.stim_kernel = stim_kernel
        self.hist_kernel = hist_kernel

    def clone(self):
        stim_kernel = self.stim_kernel.clone() if self.stim_kernel is not None else None
        hist_kernel = self.hist_kernel.clone() if self.hist_kernel is not None else None
        return self.__class__(bias=self.bias, stim_kernel=stim_kernel, hist_kernel=hist_kernel)

    def fit(self, t, mask_spikes, stim=None, alpha_l2=0, num_epochs=20, optim=None, metrics=None,
            n_metrics=10, verbose=False):

        dt = get_timestep(t)
        mask_spikes = mask_spikes.float()
        loss, metrics_list = [], None

        _loss = torch.tensor(float('nan'))

        for epoch in range(num_epochs):

            if verbose:
                print('\r', 'epoch', epoch, 'of', num_epochs, '||',
                      'loss %.4f' % _loss.item(), end='')

            optim.zero_grad()

            log_lam = self.log_conditional_intensity(t, mask_spikes, stim=stim)
            _loss = negative_log_likelihood(dt, mask_spikes, log_lam)
            if alpha_l2 > 0:
                _loss = _loss + alpha_l2 * torch.sum(self.eta_coefs ** 2)

            if (epoch % n_metrics) == 0:
                with torch.no_grad():
                    _metrics = metrics(self, t, mask_spikes, log_lam) if metrics is not None else {}
                    metrics_list = _append_metrics(metrics_list, _metrics)

            _loss.backward()
            optim.step()
            loss.append(_loss.item())

        return loss, metrics_list

    def sample(self, t, stim=None, shape=(1,), full_output=False):
        """Samples from the model at the given times"""

        dt = get_timestep(t)
        
        stim_shape = () if stim is None else stim.shape[1:]
        shape = (len(t), ) + stim_shape + shape

        log_lam = torch.zeros(shape) + self.bias
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
                    hist_ker_values = self.hist_kernel.evaluate(t[j + 1:] - t[j + 1])
                    hist_ker_values = hist_ker_values[(slice(None),) + (None,) * len(shape[1:])]
                    hist_conv[j + 1:, :] += hist_ker_values * mask_spikes[j:j+1].float()

                # if torch.any(mask_spikes[j]) and j < len(t) - 1:
                #     hist_ker_values = self.hist_kernel.evaluate(t[j + 1:] - t[j + 1])
                #     hist_conv[j + 1:, mask_spikes[j]] += hist_ker_values[:, None]
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
        mask_spikes = mask_spikes.float()
        dt = get_timestep(t)

        log_lam = torch.zeros(shape) + self.bias
        mask_spikes = shift_tensor(mask_spikes, 1, fill_value=0.)

        if stim is not None:
            stim_conv = self.stim_kernel(stim, dt=dt)
            stim_conv = shift_tensor(stim_conv, 1, fill_value=0.)
            stim_conv = stim_conv.reshape(stim_conv.shape + (1,) * (len(shape) - stim.ndim))
            log_lam = log_lam + stim_conv
        else:
            stim_conv = None
            
        if self.hist_kernel is not None:
            eta_conv = self.hist_kernel(mask_spikes, dt=dt)
            log_lam = log_lam + eta_conv
        else:
            eta_conv = None

        lam = torch.exp(log_lam)

        if full_output:
            return stim_conv, eta_conv, log_lam, lam
        else:
            return log_lam

    def save(self, path):
        params = dict(bias=self.bias, stim_kernel=self.stim_kernel, hist_kernel=self.hist_kernel)
        with open(path, "wb") as fit_file:
            pickle.dump(params, fit_file)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as fit_file:
            params = pickle.load(fit_file)
        glm = cls(bias=params['bias'], stim_kernel=params['stim_kernel'], hist_kernel=params['hist_kernel'])
        return glm
