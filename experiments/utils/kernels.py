import numpy as np
import torch

from signals import raw_autocorrelation


def phi_autocov(t, mask_spikes, padding=250):
    autocor = raw_autocorrelation(mask_spikes.numpy(), biased=True)
    autocor = torch.from_numpy(autocor[1:padding])
    return autocor


def phi_mean_fr(t, mask_spikes):
    T = t[-1] - t[0] + t[1]
    return torch.sum(mask_spikes, 0).double()[None, :] * 1000 / T


def ker_sch(t, mask_spikes1, mask_spikes2, sd2=1e6):
    cum1 = torch.cumsum(mask_spikes1, dim=0)
    cum2 = torch.cumsum(mask_spikes2, dim=0)
    diff_cum = torch.sum((cum1[:, :, None] - cum2[:, None, :])**2, dim=0)
    gramian = torch.exp(-diff_cum / sd2)# * sd2
#     for sd2 in sd2s[1:]:
#         gramian = gramian + torch.exp(-diff_cum / sd2) * sd2
    return gramian
