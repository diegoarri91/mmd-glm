import numpy as np
from scipy.stats import kstest
import torch


def log_likelihood_bernoulli(t, s):
    pass

def bernoulli_log_likelihood_poisson_process(mask_spikes):
    n_spk = np.sum(mask_spikes)
    n_nospk = mask_spikes.size - n_spk
    p_spk = n_spk / mask_spikes.size
    log_likelihood_poisson_process = n_spk * np.log(p_spk) + n_nospk * np.log(1 - p_spk)
    return log_likelihood_poisson_process

def poisson_log_likelihood_poisson_process(dt, mask_spikes, u, r):
    lent = mask_spikes.shape[0]
    n_spikes = np.sum(mask_spikes, 0)
    n_spikes = n_spikes[n_spikes > 0]
    log_likelihood = np.sum(u[mask_spikes]) - dt * np.sum(r) + np.sum(n_spikes) * np.log(dt)
    log_likelihood_poisson = np.sum(n_spikes * (np.log(n_spikes / lent) - 1))
    log_like_normed = (log_likelihood - log_likelihood_poisson) / np.log(2) / np.sum(n_spikes)
    return log_like_normed

def MMD(t, s1, s2, phi=None, kernel=None, biased=False):
    
    if kernel is not None:
        gramian_11 = kernel(t, s1, s1)
        gramian_22 = kernel(t, s2, s2)
        gramian_12 = kernel(t, s1, s2)
    
    if not(biased):
        idx_1 = np.triu_indices(gramian_11.shape[0], k=1)
        idx_1 = (torch.tensor(idx_1[0]), torch.tensor(idx_1[1]))
        idx_2 = np.triu_indices(gramian_22.shape[0], k=1)
        idx_2 = (torch.tensor(idx_2[0]), torch.tensor(idx_2[1]))
        mmd = torch.mean(gramian_11[idx_1]) + torch.mean(gramian_22[idx_2]) - 2 * torch.mean(gramian_12)
    else:
        if kernel is None:
            pass
#             mmd = torch.sum((torch.mean(phi_d, 1) - torch.mean(phi_fr, 1))**2)
        else:
            mmd = torch.mean(gramian_11) + torch.mean(gramian_22) - 2 * torch.mean(gramian_12)
            
    return mmd

def time_rescale_transform(dt, mask_spikes, r):
    
    integral_r = np.cumsum(r * dt, axis=0)

    z = []
    for sw in range(mask_spikes.shape[1]):
        integral_r_spikes = integral_r[mask_spikes[:, sw], sw] 
        z += [1. - np.exp(-(integral_r_spikes[1:] - integral_r_spikes[:-1]))]

    ks_stats = kstest(np.concatenate(z), 'uniform', args=(0, 1))

    return z, ks_stats