import numpy as np


def after_metrics(model, t, mask_spikes, mask_spikes_fr):
    b_grad = model.b.grad.detach().item()
    eta_coefs_grad = model.eta_coefs.grad.detach().clone().numpy()
    T_train = t[-1] - t[0] + t[1]
    fr = np.sum(mask_spikes_fr.double().numpy(), 0) / T_train * 1000
    mu_fr = np.mean(fr) 
    fr_max = np.max(fr)
    return dict(b_grad=b_grad, eta_coefs_grad=eta_coefs_grad, mu_fr=mu_fr, fr_max=fr_max)


def fun_metrics_mmd(model, t, mask_spikes, mask_spikes_fr):
    T_train = t[-1] - t[0] + t[1]
    fr = np.sum(mask_spikes_fr.double().numpy(), 0) / T_train * 1000
    mu_fr = np.mean(fr)
    max_fr = np.max(fr)
    min_fr = np.min(fr)
    b_grad = model.b.grad.detach().item()
    eta0_grad = model.eta_coefs.grad.detach().numpy().copy()
    return dict(mu_fr=mu_fr, max_fr=max_fr, min_fr=min_fr, mu2_fr=np.mean(fr**2), b_grad=b_grad, eta0_grad=eta0_grad)
