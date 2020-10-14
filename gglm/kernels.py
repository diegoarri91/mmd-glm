from signals import raw_autocorrelation
import torch
from torch.nn.functional import conv1d


def phi_autocov(t, mask_spikes, padding=250):
    autocor = raw_autocorrelation(mask_spikes.numpy(), biased=True)
    autocor = torch.from_numpy(autocor[1:padding])
    return autocor


def phi_autocor_history(t, r, model, padding=250):
    T = len(t)
    eta = torch.log(r) - model.b.detach()
    autocov = conv1d(eta.T[None, :, :], eta.T[:, None, :], padding=padding, groups=eta.shape[1]) / T
    autocov = autocov[0, :, padding + 1:].T
    return autocov
