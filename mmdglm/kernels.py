from signals import raw_autocorrelation
import torch
from torch.nn.functional import conv1d


def phi_autocor(t, mask_spikes, padding=250):
    autocor = raw_autocorrelation(mask_spikes.numpy(), biased=True)
    autocor = torch.from_numpy(autocor[1:padding])
    return autocor


def phi_autocor_history(t, r, model, padding=250):
    T = len(t)
    eta = torch.log(r) - model.b.detach()
    autocov = conv1d(eta.T[None, :, :], eta.T[:, None, :], padding=padding, groups=eta.shape[1]) / T
    autocov = autocov[0, :, padding + 1:].T
    return autocov


def ker_schoenberg(t, mask_spikes1, mask_spikes2, sd2=1e0):
    cum1 = torch.cumsum(mask_spikes1, dim=0)
    cum2 = torch.cumsum(mask_spikes2, dim=0)
    gramian = torch.exp(-torch.sum((cum1[:, :, None] - cum2[:, None, :])**2, dim=0) / sd2)
    return gramian
