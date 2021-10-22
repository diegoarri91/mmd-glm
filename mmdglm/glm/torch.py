import torch

from .base import GLM
from ..metrics import negative_log_likelihood, _append_metrics
from ..utils import get_dt


class TorchGLM(GLM, torch.nn.Module):
    
    """Point process autoregressive GLM implemented in Pytorch"""
    
    def __init__(self, bias=0, kappa=None, hist=None):
        torch.nn.Module.__init__(self)
        GLM.__init__(self, bias=bias, kappa=kappa, hist=hist)
        
        bias = torch.tensor([bias]).float()
        self.register_parameter("b", torch.nn.Parameter(bias))
        
        if self.stim is not None:
            kappa_coefs = torch.from_numpy(kappa.coefs).float()
            self.register_parameter("kappa_coefs", torch.nn.Parameter(kappa_coefs))
            
        if self.hist is not None:
            eta_coefs = torch.from_numpy(hist.coefs).float()
            self.register_parameter("eta_coefs", torch.nn.Parameter(eta_coefs))
    
    def forward(self, dt, X):
        """Returns the intensity given the data matrix"""
        theta = self.get_params()
        u = torch.einsum('tka,a->tk', X, theta)
        r = torch.exp(u)
        return r

    def fit(self, t, mask_spikes, stim=None, l2=False, alpha_l2=1e0, num_epochs=20, optim=None, metrics=None,
            n_metrics=10, verbose=False):
        
        dt = torch.tensor([get_dt(t)])
        loss, metrics_list = [], None
        
        X = torch.from_numpy(self.likelihood_kwargs(t.numpy(), mask_spikes.numpy(), stim=stim)['X']).float()
        
        _loss = torch.tensor(float('nan'))
        
        for epoch in range(num_epochs):
            
            if verbose:
                print('\r', 'epoch', epoch, 'of', num_epochs, '||', 
                      'loss %.4f' % _loss.item(), end='')

            optim.zero_grad()

            r = self(dt, X)
            _loss = negative_log_likelihood(dt, mask_spikes, r)
            if l2:
                _loss = _loss + alpha_l2 * torch.sum(self.eta_coefs**2)

            if (epoch % n_metrics) == 0:
                with torch.no_grad():
                    _metrics = metrics(self, t, mask_spikes, X) if metrics is not None else {}
                    if l2:
                        _metrics['nll'] = _nll.detach()
                    metrics_list = _append_metrics(metrics_list, _metrics)

            _loss.backward()
            optim.step()
            loss.append(_loss.item())

            theta = self.get_params()
            self.set_params(theta.detach().numpy())

        return loss, metrics_list

    def get_params(self):
        
        n_kappa = 0 if self.stim is None else self.stim.nbasis
        n_eta = 0 if self.hist is None else self.hist.nbasis
        theta = torch.zeros(1 + n_kappa + n_eta)
        
        theta[0] = self.bias
        if self.stim is not None:
            theta[1:1 + n_kappa] = self.kappa_coefs
        if self.hist is not None:
            theta[1 + n_kappa:] = self.eta_coefs
        
        return theta
