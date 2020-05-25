# import numpy as np
import torch

from .base import GLM
from ..utils import get_dt

dic_nonlinearities = {'exp': lambda x: torch.exp(x), 'log_exp': lambda x: torch.log(1 + torch.exp(x))}

class TorchGLM(GLM, torch.nn.Module):

    def __init__(self, u0=0, kappa=None, eta=None, non_linearity='exp', noise='poisson'):
        torch.nn.Module.__init__(self)
        GLM.__init__(self, u0=u0, kappa=kappa, eta=eta, non_linearity=non_linearity)
        self.non_linearity_torch = dic_nonlinearities[non_linearity]
        self.noise = noise
        
        n_kappa = 0 if self.kappa is None else self.kappa.nbasis
        n_eta = 0 if self.eta is None else self.eta.nbasis

        # TODO. implement convolutions in pytorch
        
        b = torch.tensor([u0]).double()
        self.register_parameter("b", torch.nn.Parameter(b))
        
        if self.kappa is not None:
            kappa_coefs = torch.from_numpy(kappa.coefs)
            self.register_parameter("kappa_coefs", torch.nn.Parameter(kappa_coefs))
        if self.eta is not None:
            eta_coefs = torch.from_numpy(eta.coefs)
            self.register_parameter("eta_coefs", torch.nn.Parameter(eta_coefs))
        
    def forward(self, dt, mask_spikes, X):
        
#         theta = torch.cat((self.b, self.kappa_coefs, self.eta_coefs))
        theta = self.get_params()
        
        u = torch.einsum('tka,a->tk', X, theta)
        r = self.non_linearity_torch(u)
        
        if self.noise == 'poisson':
            nll = -(torch.sum(u[mask_spikes]) - dt * torch.sum(r))
        elif self.noise == 'bernoulli':
            nll = -(torch.sum(torch.log(1 - torch.exp(-dt * r[mask_spikes]) + 1e-24) ) - \
                     dt * torch.sum(r[~mask_spikes]))
            
        return nll
    
    def get_params(self):
        n_kappa = 0 if self.kappa is None else self.kappa.nbasis
        n_eta = 0 if self.eta is None else self.eta.nbasis
        theta = torch.zeros(1 + n_kappa + n_eta)
        theta[0] = self.b
        if self.kappa is not None:
            theta[1:1 + n_kappa] = self.kappa_coefs
        if self.eta is not None:
            theta[1 + n_kappa:] = self.eta_coefs
        theta = theta.double()
        return theta

    def train(self, t, mask_spikes, stim=None, optim=None, num_epochs=20, verbose=False, metrics=None):
        
        dt = torch.tensor([get_dt(t)])
        nll, metrics_list = [], []
        
        X = torch.from_numpy(self.objective_kwargs(t, mask_spikes, stim=stim)['X']).double()
        
        _nll = torch.tensor(float('nan'))
        
        for epoch in range(num_epochs):
            
            if verbose:
                print('\r', 'epoch', epoch, 'of', num_epochs, 
                      'nll', round(_nll.item(), 4), end='')
            
            optim.zero_grad()
            _nll = self(dt, mask_spikes, X)
#             print(_nll)
#             print(kwnkwn)
            
            if metrics is not None:
                _metrics = metrics(self, t, mask_spikes, X)
                if epoch == 0:
                    metrics_list = {key:[val] for key, val in _metrics.items()}
                else:
                    for key, val in _metrics.items():
                        metrics_list[key].append(val)
                        
            _nll.backward()
            optim.step()
#             theta = torch.cat((self.b, self.kappa_coefs, self.eta_coefs))
            theta = self.get_params()
            self.set_params(theta.data.detach().numpy())
            
            nll.append(_nll.item())
            
        return nll, metrics_list
