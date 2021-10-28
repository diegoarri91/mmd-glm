import numpy as np
import torch

from .base import GLM
from ..metrics import _mmd_from_features, _mmd_from_gramians
from ..utils import get_timestep, shift_tensor


class MMDGLM(GLM):
    """Implements a point process autoregressive GLM that minimizes a joint negative log-likelihood and MMD objective"""
    
    def __init__(self, bias=0, stim_kernel=None, hist_kernel=None):
        super().__init__(bias=bias, stim_kernel=stim_kernel, hist_kernel=hist_kernel)
    
    # def get_params(self):
    #     n_kappa = 0 if self.stim_kernel is None else self.stim_kernel.nbasis
    #     n_eta = 0 if self.hist_kernel is None else self.hist_kernel.nbasis
    #     theta = torch.zeros(1 + n_kappa + n_eta)
    #     theta[0] = self.bias
    #     if self.stim_kernel is not None:
    #         theta[1:1 + n_kappa] = self.kappa_coefs
    #     if self.hist_kernel is not None:
    #         theta[1 + n_kappa:] = self.eta_coefs
    #     theta = theta.double()
    #     return theta
    #
    # def _neg_log_likelihood(self, dt, mask_spikes, X_dc):
    #     theta_g = self.get_params()
    #     u_dc = torch.einsum('tka,a->tk', X_dc, theta_g)
    #     r_dc = torch.exp(u_dc)
    #     neg_log_likelihood = -(torch.sum(torch.log(1 - torch.exp(-dt * r_dc)) * mask_spikes.double()) - \
    #                            dt * torch.sum(r_dc * (1 - mask_spikes.double())))
    #     return neg_log_likelihood
    #
    # def _score(self, dt, mask_spikes, X):
    #     with torch.no_grad():
    #         theta_g = self.get_params().detach()
    #         u = torch.einsum('tka,a->tk', X, theta_g)
    #         r = torch.exp(u)
    #         exp_r = torch.exp(r * dt)
    #         score = dt * torch.einsum('tka,tk->ka', X, r / (exp_r - 1) * mask_spikes.double()) - \
    #                 dt * torch.einsum('tka,tk->ka', X, r * (1 - mask_spikes.double()))
    #     return score
    
    def fit(self, t, mask_spikes, stim=None, phi=None, kernel=None, biased=False, n_batch_fr=100, kernel_kwargs=None,
            alpha_ll=0., alpha_l2=0., num_epochs=20, optim=None, scheduler=None, clip=None, metrics=None, n_metrics=25,
            verbose=False):

        # n_d = mask_spikes.shape[1]
        dt = get_timestep(t)
        loss, nll, mmd = [], [], []
        
        # X_dc = torch.from_numpy(self.likelihood_kwargs(t.numpy(), mask_spikes.numpy(), stim=stim)['X']).float()
        
        kernel_kwargs = kernel_kwargs if kernel_kwargs is not None else {}
        
        if phi is not None:
            phi_d = phi(t, mask_spikes, **kernel_kwargs)
            sum_phi_d = torch.sum(phi_d, 1)
        elif kernel is not None:
            gramian_d_d = kernel(t, mask_spikes, mask_spikes, **kernel_kwargs)
        else:
            RuntimeError("Either the feature map phi or the kernel have to be not None")
            
        _loss = torch.tensor([float('nan')])

        for epoch in range(num_epochs):
            if verbose:
                print('\r', 'epoch', epoch, 'of', num_epochs, 
                      'loss', np.round(_loss.item(), 10), end='')
            
            optim.zero_grad()
            
            # r_fr, mask_spikes_fr, X_fr = self(t, stim=stim, n_batch_fr=n_batch_fr)
            log_lam_fr, mask_spikes_fr = self.sample(t, stim=stim, shape=(n_batch_fr,))
            log_lam_fr = log_lam_fr.detach()
            
            log_proba = torch.sum(torch.log(1 - torch.exp(-dt * r_fr) + 1e-24) * mask_spikes_fr.double(), 0) - \
                        dt * torch.sum(r_fr * (1 - mask_spikes_fr.double()), 0)
            
            if phi is not None:
                phi_fr = phi(t, mask_spikes_fr, **kernel_kwargs)
                
                if not biased:
                    log_proba_phi = log_proba[None, :] * phi_fr
                    sum_log_proba_phi_fr = torch.sum(log_proba_phi, 1)
                    sum_phi_fr = torch.sum(phi_fr, 1)
                    norm2_fr = (torch.sum(sum_log_proba_phi_fr * sum_phi_fr) - torch.sum(log_proba_phi * phi_fr)) / (n_batch_fr * (n_batch_fr - 1))
                    mmd_surr = 2 * norm2_fr - 2 / (n_d * n_batch_fr) * torch.sum(sum_phi_d * sum_log_proba_phi_fr)
                else:
                    mmd_surr = -2 * torch.sum((torch.mean(phi_d, 1) - torch.mean(phi_fr, 1)) * torch.mean(log_proba[None, :] * phi_fr, 1))
            else:
                gramian_fr_fr = kernel(t, mask_spikes_fr, mask_spikes_fr, **kernel_kwargs)
                gramian_d_fr = kernel(t, mask_spikes, mask_spikes_fr, **kernel_kwargs)
                if not biased:
                    gramian_fr_fr.fill_diagonal_(0)
                    mmd_surr = 2 * torch.sum(log_proba[:, None] * gramian_fr_fr) / (n_batch_fr * (n_batch_fr - 1)) \
                             - 2 * torch.mean(log_proba[None, :] * gramian_d_fr)
                else:
                    mmd_surr = torch.mean(((log_proba[:, None] + log_proba[None, :]) * gramian_fr_fr)) \
                                                  -2 * torch.mean(log_proba[None, :] * gramian_d_fr)
            
            _loss = mmd_surr
            
            if alpha_ll > 0:
                _nll = self._neg_log_likelihood(dt, mask_spikes, X_dc)
                _loss = _loss + alpha_ll * _nll
                nll.append(_nll.item())

            _loss.backward()
                    
            if clip is not None:
                torch.nn.utils.clip_grad_value_(self.parameters(), clip)
            
            if epoch % n_metrics == 0:
                
                if kernel is not None:
                    _metrics = metrics(self, t, mask_spikes, mask_spikes_fr, gramian_d_d=gramian_d_d, log_proba=log_proba, 
                                       gramian_fr_fr=gramian_fr_fr, gramian_d_fr=gramian_d_fr) if metrics is not None else {}
                    _metrics['mmd'] = _mmd_from_gramians(t, gramian_d_d, gramian_fr_fr, gramian_d_fr, biased=biased).item()
                else:
                    _metrics = metrics(self, t, mask_spikes, mask_spikes_fr, log_proba=log_proba, 
                                       phi_d=phi_d, phi_fr=phi_fr) if metrics is not None else {}
                    _metrics['mmd'] = _mmd_from_features(t, phi_d, phi_fr, biased=biased).item()

                if epoch == 0:
                    metrics_list = {key:[val] for key, val in _metrics.items()}
                else:
                    for key, val in _metrics.items():
                        metrics_list[key].append(val)
                        
            optim.step()
            if scheduler is not None:
                scheduler.step()
            
            theta_g = self.get_params()
            self.set_params(theta_g.data.detach().numpy())
            
            loss.append(_loss.item())
            
        if metrics is None:
            metrics_list = None
        
        return loss, nll, metrics_list

