# import numpy as np
import torch

from .base import GLM
from ..utils import get_dt

dic_nonlinearities = {'exp': lambda x: torch.exp(x), 'log_exp': lambda x: torch.log(1 + torch.exp(x))}

class TorchGLM(GLM, torch.nn.Module):

    def __init__(self, u0=0, kappa=None, eta=None, non_linearity='exp'):
        torch.nn.Module.__init__(self)
        GLM.__init__(self, u0=u0, kappa=kappa, eta=eta, non_linearity=non_linearity)
        self.non_linearity_torch = dic_nonlinearities[non_linearity]
        
        n_kappa = 0 if self.kappa is None else self.kappa.nbasis
        n_eta = 0 if self.eta is None else self.eta.nbasis
        
        theta = torch.zeros(1 + n_kappa + n_eta)
        theta[0] = torch.tensor([u0])
        
        # TODO. imlpement convolution filters in pytorch and change this
        
        if self.kappa is not None:
            theta[1:1 + n_kappa] = torch.from_numpy(kappa.coefs)
        if self.eta is not None:
            theta[1 + n_kappa:] = torch.from_numpy(eta.coefs)
            
        theta = theta.double()
        
        self.register_parameter("theta", torch.nn.Parameter(theta))
        
        
    def forward(self, t, mask_spikes, X):
        
        dt = get_dt(t)
#         self.theta.requires_grad = True
        
        n_batch = mask_spikes.shape[1]
        u = torch.einsum('tka,a->tk', X, self.theta)
        r = self.non_linearity_torch(u)
        
#         if stim is not None:
#             u_fr, r_fr, mask_spikes_fr = self.sample(t, stim=stim)
#         else:
#             
#         X_fr = torch.from_numpy(self.objective_kwargs(t, mask_spikes_fr, stim=stim)['X'])
#         u_fr = torch.einsum('tka,a->tk', X_fr, self.theta)
#         r_fr = self.non_linearity_torch(u_fr)
            
#         mask_spikes_fr = torch.from_numpy(mask_spikes_fr)
#         mask_spikes = torch.cat((mask_spikes_te, mask_spikes_fr), dim=1).double()
#         u = torch.cat((u_te, u_fr), dim=1).double()
#         r = torch.cat((r_te, r_fr), dim=1).double()
        
#         mmd = self.mmd(r, mask_spikes, y, lam_mmd, distance=distance, **mmd_kwargs)
        loss = -(torch.sum(u[mask_spikes]) - dt * torch.sum(r))
#         loss = -(torch.sum(torch.log(1 - torch.exp(-dt * r_te[mask_spikes_te]))) - dt * torch.sum(r_te[~mask_spikes_te]))

        return loss

    def train(self, t, mask_spikes, stim=None, optim=None, num_epochs=20, n_batch_fr=100, verbose=False, metrics=None):
        
#         self.metrics = metrics
        n_batch_te = mask_spikes.shape[1]
        dt = torch.tensor([get_dt(t)])
        loss, metrics_list = [], []
        
        X = torch.from_numpy(self.objective_kwargs(t, mask_spikes, stim=stim)['X']).double()
        
#         _loss = torch.tensor(np.nan)
        _loss = torch.tensor(float('nan'))
        
        for epoch in range(num_epochs):
            
            if verbose:
                print('\r', 'epoch', epoch, 'of', num_epochs, 
                      'loss', round(_loss.item(), 4), end='')
            
            optim.zero_grad()
            _loss = self(t, mask_spikes, X)
#             if hasattr(self, 'metrics'):
            if metrics is not None:
                _metrics = metrics(self, t, mask_spikes, X, n_batch_fr)
                if epoch == 0:
                    metrics_list = {key:[val] for key, val in _metrics.items()}
                else:
                    for key, val in _metrics.items():
                        metrics_list[key].append(val)
                        
            _loss.backward()
            optim.step()
            self.set_params(self.theta.data.detach().numpy())
            
            loss.append(_loss.item())
            
        return loss, metrics_list
