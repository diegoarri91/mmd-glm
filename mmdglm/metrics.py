import numpy as np
from scipy.stats import kstest
import torch


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


def MMD(t, s1, s2, phi=None, kernel=None, biased=False, **kwargs):
    
    with torch.no_grad():

        n1, n2 = s1.shape[1], s2.shape[1]
        
        if kernel is not None:
            gramian_11 = kernel(t, s1, s1, **kwargs)
            gramian_22 = kernel(t, s2, s2, **kwargs)
            gramian_12 = kernel(t, s1, s2, **kwargs)
            if not biased:
                gramian_11.fill_diagonal_(0)
                gramian_22.fill_diagonal_(0)
                mmd = torch.sum(gramian_11) / (n1 * (n1 - 1)) + torch.sum(gramian_22) / (n2 * (n2 - 1)) \
                                              - 2 * torch.mean(gramian_12)
            else:
                mmd = torch.mean(gramian_11) + torch.mean(gramian_22) - 2 * torch.mean(gramian_12)

        elif phi is not None:
            phi_1 = phi(t, s1, **kwargs)
            phi_2 = phi(t, s2, **kwargs)
            if biased:
                phi_1_mean = torch.mean(phi_1, 1)
                phi_2_mean = torch.mean(phi_2, 1)
                mmd = torch.sum((phi_1_mean - phi_2_mean)**2)
            else:
                sum_phi_1 = torch.sum(phi_1, 1)
                sum_phi_2 = torch.sum(phi_2, 1)
                norm2_1 = (torch.sum(sum_phi_1**2) - torch.sum(phi_1**2)) / (n1 * (n1 - 1))
                norm2_2 = (torch.sum(sum_phi_2**2) - torch.sum(phi_2**2)) / (n2 * (n2 - 1))
                mean_dot = torch.sum(sum_phi_1 * sum_phi_2) / (n1 * n2)     
                mmd = norm2_1 + norm2_2 - 2 * mean_dot
        
    return mmd


def time_rescale_transform(dt, mask_spikes, r):
    
    integral_r = np.cumsum(r * dt, axis=0)

    z = []
    for sw in range(mask_spikes.shape[1]):
        integral_r_spikes = integral_r[mask_spikes[:, sw], sw] 
        z += [1. - np.exp(-(integral_r_spikes[1:] - integral_r_spikes[:-1]))]

    ks_stats = kstest(np.concatenate(z), 'uniform', args=(0, 1))

    return z, ks_stats


def negative_log_likelihood(dt, mask_spikes, r):
    ll = torch.sum(
                torch.log(1 - torch.exp(-dt * r) * mask_spikes + 1e-24)
            ) - dt * torch.sum(r * (1 - mask_spikes))
    return -ll


def _mmd_from_gramians(t, gramian_11, gramian_22, gramian_12, biased=False):
    n1, n2 = gramian_11.shape[0], gramian_22.shape[0]
    if not biased:
        gramian_11.fill_diagonal_(0)
        gramian_22.fill_diagonal_(0)
        mmd = torch.sum(gramian_11) / (n1 * (n1 - 1)) + torch.sum(gramian_22) / (n2 * (n2 - 1)) \
                                              - 2 * torch.mean(gramian_12)
    else:
        mmd = torch.mean(gramian_11) + torch.mean(gramian_22) - 2 * torch.mean(gramian_12)
    return mmd


def _mmd_from_features(t, phi_1, phi_2, biased=False):
    n1, n2 = phi_1.shape[1], phi_2.shape[1]
    if biased:
        phi_1_mean = torch.mean(phi_1, 1)
        phi_2_mean = torch.mean(phi_2, 1)
        mmd = torch.sum((phi_1_mean - phi_2_mean)**2)
    else:
        sum_phi_1 = torch.sum(phi_1, 1)
        sum_phi_2 = torch.sum(phi_2, 1)
        norm2_1 = (torch.sum(sum_phi_1**2) - torch.sum(phi_1**2)) / (n1 * (n1 - 1))
        norm2_2 = (torch.sum(sum_phi_2**2) - torch.sum(phi_2**2)) / (n2 * (n2 - 1))
        mean_dot = torch.sum(sum_phi_1 * sum_phi_2) / (n1 * n2)     
        mmd = norm2_1 + norm2_2 - 2 * mean_dot
    return mmd


def _append_metrics(metrics_list, _metrics):
    if metrics_list is None:
        metrics_list = {key:[val] for key, val in _metrics.items()}
    else:
        for key, val in _metrics.items():
            metrics_list[key].append(val)
    return metrics_list