import numpy as np
import torch

from signals import raw_autocorrelation


def phi_autocov(t, mask_spikes, padding=200):
    autocor = raw_autocorrelation(mask_spikes.numpy(), biased=True)
    autocor = torch.from_numpy(autocor[1:padding])
    return autocor
