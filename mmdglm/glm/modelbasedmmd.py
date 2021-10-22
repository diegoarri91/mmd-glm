import numpy as np
import torch

from .torch import TorchGLM
from ..metrics import _append_metrics, _mmd_from_features, _mmd_from_gramians, negative_log_likelihood
from ..utils import get_dt, shift_array


class ModelBasedMMDGLM(TorchGLM, torch.nn.Module):

    """Implements a point process autoregressive GLM that minimizes a joint negative
    log-likelihood and model based MMD objective"""
    
    def __init__(self, bias=0, kappa=None, hist=None):
        torch.nn.Module.__init__(self)
        TorchGLM.__init__(self, bias=bias, kappa=kappa, hist=hist)
        
    def fit(self, t, mask_spikes, stim=None, phi=None, kernel=None, log_likelihood=True, alpha_mmd=1e0, biased=True,
            n_batch_fr=50, kernel_kwargs=None, num_epochs=20, optim=None, clip=None, metrics=None, n_metrics=1, verbose=False):

        n_d = mask_spikes.shape[1]
    
        dt = torch.tensor([get_dt(t)])
        loss, metrics_list = [], None
        
        X_dc = torch.from_numpy(self.likelihood_kwargs(t.numpy(), mask_spikes.numpy(), stim=stim)['X']).float()
        
        kernel_kwargs = kernel_kwargs if kernel_kwargs is not None else {}
                
        _loss = torch.tensor([np.nan])

        for epoch in range(num_epochs):
            
            if verbose:
                print('\r', 'epoch', epoch, 'of', num_epochs, '||', 
                      'loss %.4f' % _loss.item(), end='')

            optim.zero_grad()
                        
            r_dc = self(dt, X_dc)
            r_fr, mask_spikes_fr = self.sample_free_running(t, stim, n_batch_fr)

            if phi is not None:
                phi_d = phi(t, r_dc, model=self, **kernel_kwargs)
                phi_fr = phi(t, r_fr, model=self, **kernel_kwargs)
                mmd = _mmd_from_features(t, phi_d, phi_fr, biased=biased)
            else:
                gramian_d_d = kernel(t, r_dc, r_dc, model=self)
                gramian_fr_fr = kernel(t, r_fr, r_fr, model=self)
                gramian_d_fr = kernel(t, r_dc, r_fr, model=self)
                mmd = _mmd_from_gramians(t, gramian_11, gramian_22, gramian_12, biased=biased)
            
            _loss = alpha_mmd * mmd
            
            if log_likelihood:
                _nll = negative_log_likelihood(dt, mask_spikes, r_dc)
                _loss = _loss + _nll
                
            if (epoch % n_metrics) == 0:
                with torch.no_grad():
                    _metrics = metrics(self, t, mask_spikes, mask_spikes_fr) if metrics is not None else {}
                    _metrics['mmd'] = mmd.item()
                    if log_likelihood:
                        _metrics['nll'] = _nll.item()
                    metrics_list = _append_metrics(metrics_list, _metrics)
            
            _loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_value_(self.parameters(), clip)
            optim.step()
            loss.append(_loss.item())
            
            theta = self.get_params()
            self.set_params(theta.data.detach().numpy())
        
        return loss, metrics_list

    def sample_free_running(self, t, stim, n_batch_fr):
        _, _, mask_spikes_fr = self.sample(t, stim=stim, shape=(n_batch_fr,))
        likelihood_kwargs = self.likelihood_kwargs(t.numpy(), mask_spikes_fr, stim=stim)
        dt = likelihood_kwargs['dt']
        X_fr = torch.from_numpy(likelihood_kwargs['X']).float()
        r_fr = self(dt, X_fr)
#         mask_spikes_fr = torch.from_numpy(mask_spikes_fr)
        return r_fr, mask_spikes_fr
