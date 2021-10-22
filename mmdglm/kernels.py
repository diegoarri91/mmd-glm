from scipy.signal import fftconvolve
import torch
from torch.nn.functional import conv1d


def phi_autocor(t, mask_spikes, padding=250):
    r"""Computes the autocorrelation of the sample as a feature that can the be used for MMD minimization"""
    x = mask_spikes.numpy()
    autocor = fftconvolve(x, x[::-1], mode='full', axes=0)[::-1] / x.shape[0]
    autocor = torch.from_numpy(autocor[1:padding])
    return autocor


def phi_autocor_history(t, r, model, padding=250):
    r"""Computes the autocorrelation of the history term as a feature that can the be used for MMD minimization"""
    T = len(t)
    eta = torch.log(r) - model.bias
    autocov = conv1d(eta.T[None, :, :], eta.T[:, None, :], padding=padding, groups=eta.shape[1]) / T
    autocov = autocov[0, :, padding + 1:].T
    return autocov


def ker_schoenberg(t, mask_spikes1, mask_spikes2, sd2=1e0):
    r"""Computes the gramian matrix between the two samples using a Schoenberg kernel that can the be used for MMD 
    minimization"""
    cum1 = torch.cumsum(mask_spikes1, dim=0)
    cum2 = torch.cumsum(mask_spikes2, dim=0)
    gramian = torch.exp(-torch.sum((cum1[:, :, None] - cum2[:, None, :])**2, dim=0) / sd2)
    return gramian
