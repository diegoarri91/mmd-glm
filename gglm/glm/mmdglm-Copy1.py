import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import Parameter
from torch.optim import Adam, LBFGS

from .base import GLM
from ..utils import get_dt, shift_array

dic_nonlinearities = {'exp': lambda x: torch.exp(x), 'log_exp': lambda x: torch.log(1 + torch.exp(x))}

class MMDGLM(GLM, torch.nn.Module):

    def __init__(self, u0=0, kappa=None, eta=None, non_linearity='exp'):
        torch.nn.Module.__init__(self)
        GLM.__init__(self, u0=u0, kappa=kappa, eta=eta, non_linearity=non_linearity)
        self.non_linearity_torch = dic_nonlinearities[non_linearity]
        
        n_kappa = 0 if self.kappa is None else self.kappa.nbasis
        n_eta = 0 if self.eta is None else self.eta.nbasis
        theta_g = torch.zeros(1 + n_kappa + n_eta)
        theta_g[0] = torch.tensor([u0])
        if self.kappa is not None:
            theta_g[1:1 + n_kappa] = torch.from_numpy(kappa.coefs)
        if self.eta is not None:
            theta_g[1 + n_kappa:] = torch.from_numpy(eta.coefs)
        theta_g = theta_g.double()
        self.register_parameter("theta_g", Parameter(theta_g))
    
    def forward(self, t, mask_spikes, X_dc, distance, stim, lam_mmd, n_batch_fr, mmd_kwargs):
        
        dt = get_dt(t)
        self.theta_g.requires_grad = True
        
        n_batch_dc = mask_spikes.shape[1]
        u_dc = torch.einsum('tka,a->tk', X_dc, self.theta_g)
        r_dc = self.non_linearity_torch(u_dc)
        
        # TODO. I am calculating u_fr and r_fr twice because I can't backpropagate through my current sample function. change this
        if stim is not None:
            _, _, mask_spikes_fr = self.sample(t, stim=stim)
        else:
            _, _, mask_spikes_fr = self.sample(t, shape=(n_batch_fr,))
            
        X_fr = torch.from_numpy(self.objective_kwargs(t, mask_spikes_fr, stim=stim)['X'])
        u_fr = torch.einsum('tka,a->tk', X_fr, self.theta_g)
        r_fr = self.non_linearity_torch(u_fr)
        mask_spikes_fr = torch.from_numpy(mask_spikes_fr)
        
#         mask_spikes = torch.cat((mask_spikes, mask_spikes_fr), dim=1).double()
#         u = torch.cat((u_dc, u_fr), dim=1).double()
#         r = torch.cat((r_dc, r_fr), dim=1).double()
        
        mmd = self.mmd(r_dc, mask_spikes, r_fr, mask_spikes_fr, distance=distance, **mmd_kwargs)
        neg_log_likelihood = -(torch.sum(u_dc[mask_spikes]) - dt * torch.sum(r_dc))
#         neg_log_likelihood = -(torch.sum(torch.log(1 - torch.exp(-dt * r_dc[mask_spikes_dc]))) - dt * torch.sum(r_dc[~mask_spikes_dc]))

        loss = neg_log_likelihood + lam_mmd * mmd  # compute wasserstein distance      
        
        return loss, mmd
    
    def mmd(self, r_dc, mask_spikes, r_fr, mask_spikes_fr, distance=None, kernel=None, **kwargs):
        
        if distance is not None:
            d = distance(r_dc, mask_spikes, r_fr, mask_spikes_fr)
        
        # match average over trials. computing all products (for testing)
#         norm2_dc, norm2_fr, mean_dot = 0, 0, 0
#         for ii in range(n_batch_dc):
#             for jj in range(n_batch_fr):
#                 if jj > ii:
#                     norm2_dc += torch.sum(r_dc[:, ii] * r_dc[:, jj])
#                     norm2_fr += torch.sum(r_fr[:, ii] * r_fr[:, jj])
#                 mean_dot += torch.sum(r_dc[:, ii] * r_dc[:, jj])              
#         norm2_dc = norm2_dc / (n_batch_dc * (n_batch_dc - 1))       
#         norm2_fr = norm2_fr / (n_batch_fr * (n_batch_fr - 1))
#         mean_dot = mean_dot / (n_batch_dc * n_batch_fr)
#         distance = lam * (norm2_dc + norm2_fr - 2 * mean_dot)
        
#         sd = kwargs['sd']
#         distance = lam * torch.exp(-(norm2_dc + norm2_fr - 2 * mean_dot)**2 / 2 * sd**2)
        
#         import torch.nn.functional  as F

#         weight = torch.from_numpy(KernelFun.rbf(5).interpolate(np.arange(0, 31, 1) - 15))[None, 2, :]
#         norm2_dc = (torch.sum(r_sum_dc**2) - torch.sum(r_dc**2)) / (n_batch_dc * (n_batch_dc - 1))
#         norm2_fr = (torch.sum(r_sum_fr**2) - torch.sum(r_fr**2)) / (n_batch_fr * (n_batch_fr - 1))
#         mean_dot = torch.sum(r_sum_dc * r_sum_fr, 0) / (n_batch_dc * n_batch_fr)
#         aux = F.conv1d(mask, weight, padding=(weight.shape[2] - 1) // 2)
#         , , kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'
#         F.conv1d(2, , stride=1)
        
        return d
    
    def train(self, t, mask_spikes, distance, stim=None, lam_mmd=1e0, optim=None, num_epochs=20, n_batch_fr=100, verbose=False,
              mmd_kwargs=None, metrics=None):

        n_batch_dc = mask_spikes.shape[1]
    
        dt = torch.tensor([get_dt(t)])
        loss, mmd = [], []
        
        X_dc = torch.from_numpy(self.objective_kwargs(t, mask_spikes, stim=stim)['X']).double()
        
        _loss = torch.tensor([np.nan])
        mmd_kwargs = {} if mmd_kwargs is None else mmd_kwargs
        for epoch in range(num_epochs):
            if verbose:
                print('\r', 'epoch', epoch, 'of', num_epochs, 
                      'loss', np.round(_loss.item(), 4), end='')
            
            optim.zero_grad()
            
            _loss, _mmd = self(t, mask_spikes, X_dc, distance, stim=stim, lam_mmd=lam_mmd, 
                                n_batch_fr=n_batch_fr, mmd_kwargs=mmd_kwargs)
            
#             if hasattr(self, 'metrics'):
            if metrics is not None:
                _metrics = metrics(self, t, mask_spikes, X_dc, n_batch_fr)
                if epoch == 0:
                    metrics_list = {key:[val] for key, val in _metrics.items()}
                else:
                    for key, val in _metrics.items():
                        metrics_list[key].append(val)
                        
            _loss.backward()
            optim.step()
            self.set_params(self.theta_g.data.detach().numpy())
            
            loss.append(_loss.item())
            mmd.append(_mmd.item())
#             n_spikes.append(_n_spikes)
            
        self.set_params(self.theta_g.data.detach().numpy())
        
        return loss, mmd, metrics_list

