import torch

from .base import GLM
from ..metrics import _append_metrics, _mmd_from_features, _mmd_from_gramians, negative_log_likelihood
from ..utils import get_timestep


class ModelBasedMMDGLM(GLM):
    """Implements a point process autoregressive GLM that minimizes a joint negative log-likelihood and model based MMD
    objective"""
    
    def __init__(self, bias, stim_kernel=None, hist_kernel=None):
        super().__init__(bias=bias, stim_kernel=stim_kernel, hist_kernel=hist_kernel)
        
    def fit(self, t, mask_spikes, stim=None, phi=None, kernel=None, biased=True, n_batch_fr=50, kernel_kwargs=None,
            alpha_ll=0., num_epochs=20, optim=None, clip=None, metrics=None, n_metrics=1, verbose=False):

        if phi is None and kernel is None:
            raise RuntimeError("Either the feature map phi or the kernel have to be not None")

        dt = get_timestep(t)
        kernel_kwargs = kernel_kwargs if kernel_kwargs is not None else {}
        loss, metrics_list = [], None
        _loss = torch.tensor(float('nan'))

        for epoch in range(num_epochs):
            
            if verbose:
                print('\r', 'epoch', epoch, 'of', num_epochs, '||', 
                      'loss %.4f' % _loss.item(), end='')

            optim.zero_grad()
                        
            log_lam_d = self.log_conditional_intensity(t, mask_spikes, stim=stim)
            log_lam_fr, mask_spikes_fr = self.sample(t, stim=stim, shape=(n_batch_fr,))

            if phi is not None:
                phi_d = phi(t, log_lam_d, model=self, **kernel_kwargs)
                phi_fr = phi(t, log_lam_fr, model=self, **kernel_kwargs)
                mmd = _mmd_from_features(t, phi_d, phi_fr, biased=biased)
            else:
                gramian_d_d = kernel(t, log_lam_d, log_lam_d, model=self)
                gramian_fr_fr = kernel(t, log_lam_fr, log_lam_fr, model=self)
                gramian_d_fr = kernel(t, log_lam_d, log_lam_fr, model=self)
                mmd = _mmd_from_gramians(t, gramian_d_d, gramian_fr_fr, gramian_d_fr, biased=biased)
            
            _loss = mmd
            if alpha_ll > 0:
                _nll = negative_log_likelihood(dt, mask_spikes, log_lam_d)
                _loss = _loss + alpha_ll * _nll
                
            if (epoch % n_metrics) == 0:
                with torch.no_grad():
                    _metrics = metrics(self, t, mask_spikes, mask_spikes_fr) if metrics is not None else {}
                    _metrics['mmd'] = mmd.item()
                    if alpha_ll > 0:
                        _metrics['nll'] = _nll.item()
                    metrics_list = _append_metrics(metrics_list, _metrics)
            
            _loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_value_(self.parameters(), clip)
            optim.step()
            loss.append(_loss.item())
        
        return loss, metrics_list
