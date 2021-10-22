import numpy as np
import torch
from torch.nn import Parameter

from sptr.sptr import SpikeTrain

from .base import GLM
from ..metrics import _mmd_from_features, _mmd_from_gramians
from ..utils import get_dt, shift_array


class MMDGLM(GLM, torch.nn.Module):
    
    """Implements a point process autoregressive GLM that minimizes a joint negative
    log-likelihood and MMD objective"""
    
    def __init__(self, bias=0, kappa=None, hist=None):
        torch.nn.Module.__init__(self)
        GLM.__init__(self, bias=bias, kappa=kappa, hist=hist, non_linearity=non_linearity)
        
        n_kappa = 0 if self.stim is None else self.stim.nbasis
        n_eta = 0 if self.hist is None else self.hist.nbasis

        bias = torch.tensor([bias]).double()
        self.register_parameter("b", torch.nn.Parameter(bias))
        
        if self.stim is not None:
            kappa_coefs = torch.from_numpy(kappa.coefs)
            self.register_parameter("kappa_coefs", torch.nn.Parameter(kappa_coefs))
            
        if self.hist is not None:
            eta_coefs = torch.from_numpy(hist.coefs)
            self.register_parameter("eta_coefs", torch.nn.Parameter(eta_coefs))
    
    def forward(self, t, stim=None, n_batch_fr=None):
        """Returns samples from the model"""
        dt = get_dt(t)
        theta_g = self.get_params()
        
        if stim is not None:
            _, _, mask_spikes_fr = self.sample(t, stim=stim)
        else:
            _, _, mask_spikes_fr = self.sample(t, shape=(n_batch_fr,))

        X_fr = torch.from_numpy(self.objective_kwargs(t, mask_spikes_fr, stim=stim)['X'])
        u_fr = torch.einsum('tka,a->tk', X_fr, theta_g)
        r_fr = torch.exp(u_fr)
        mask_spikes_fr = torch.from_numpy(mask_spikes_fr)
        
        return r_fr, mask_spikes_fr, X_fr
    
    def get_params(self):
        n_kappa = 0 if self.stim is None else self.stim.nbasis
        n_eta = 0 if self.hist is None else self.hist.nbasis
        theta = torch.zeros(1 + n_kappa + n_eta)
        theta[0] = self.bias
        if self.stim is not None:
            theta[1:1 + n_kappa] = self.kappa_coefs
        if self.hist is not None:
            theta[1 + n_kappa:] = self.eta_coefs
        theta = theta.double()
        return theta
    
    def _neg_log_likelihood(self, dt, mask_spikes, X_dc):
        theta_g = self.get_params()
        u_dc = torch.einsum('tka,a->tk', X_dc, theta_g)
        r_dc = torch.exp(u_dc)
        neg_log_likelihood = -(torch.sum(torch.log(1 - torch.exp(-dt * r_dc)) * mask_spikes.double()) - \
                               dt * torch.sum(r_dc * (1 - mask_spikes.double())))
        return neg_log_likelihood
    
    def _score(self, dt, mask_spikes, X):
        with torch.no_grad():
            theta_g = self.get_params().detach()
            u = torch.einsum('tka,a->tk', X, theta_g)
            r = torch.exp(u)
            exp_r = torch.exp(r * dt)
            score = dt * torch.einsum('tka,tk->ka', X, r / (exp_r - 1) * mask_spikes.double()) - \
                    dt * torch.einsum('tka,tk->ka', X, r * (1 - mask_spikes.double()))
        return score
    
    def train(self, t, mask_spikes, phi=None, kernel=None, stim=None, log_likelihood=False, lam_mmd=1e0, biased=False, 
              optim=None, scheduler=None, num_epochs=20, n_batch_fr=100, kernel_kwargs=None, clip=None, verbose=False, 
              metrics=None, n_metrics=25):

        n_d = mask_spikes.shape[1]
        dt = torch.tensor([get_dt(t)])
        loss, nll, mmd = [], [], []
        
        X_dc = torch.from_numpy(self.likelihood_kwargs(t.numpy(), mask_spikes.numpy(), stim=stim)['X']).float()
        
        kernel_kwargs = kernel_kwargs if kernel_kwargs is not None else {}
        
        if phi is not None:
            phi_d = phi(t, mask_spikes, **kernel_kwargs)
            sum_phi_d = torch.sum(phi_d, 1)
        else:
            gramian_d_d = kernel(t, mask_spikes, mask_spikes, **kernel_kwargs)
            
        _loss = torch.tensor([np.nan])

        for epoch in range(num_epochs):
            if verbose:
                print('\r', 'epoch', epoch, 'of', num_epochs, 
                      'loss', np.round(_loss.item(), 10), end='')
            
            optim.zero_grad()
            
            r_fr, mask_spikes_fr, X_fr = self(t, stim=stim, n_batch_fr=n_batch_fr)
            
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
            
            _loss = lam_mmd * mmd_surr
            
            if log_likelihood:
                _nll = self._neg_log_likelihood(dt, mask_spikes, X_dc)
                _loss = _loss + _nll
                nll.append(_nll.item())
                        
            if not control_variates:
                _loss.backward()
            else:
                _loss.backward(retain_graph=True)
                    
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

