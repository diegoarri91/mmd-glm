import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import Parameter
from torch.optim import Adam, LBFGS

from sptr.sptr import SpikeTrain

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

        b = torch.tensor([u0]).double()
        self.register_parameter("b", torch.nn.Parameter(b))
        
        if self.kappa is not None:
            kappa_coefs = torch.from_numpy(kappa.coefs)
            self.register_parameter("kappa_coefs", torch.nn.Parameter(kappa_coefs))
        if self.eta is not None:
            eta_coefs = torch.from_numpy(eta.coefs)
            self.register_parameter("eta_coefs", torch.nn.Parameter(eta_coefs))
    
    def forward(self, t, mask_spikes, phi_d=None, gramian_d_d=None, phi=None, kernel=None, stim=None, n_batch_fr=None, 
                lam_mmd=1, biased=False, biased_mmd=False):
        
        dt = get_dt(t)
#         theta_g = torch.cat((self.b, self.kappa_coefs, self.eta_coefs))
        theta_g = self.get_params()
        
        # TODO. I am calculating u_fr and r_fr twice because I can't backpropagate through my current sample function. change this
        if stim is not None:
            _, _, mask_spikes_fr = self.sample(t, stim=stim)
        else:
            _, _, mask_spikes_fr = self.sample(t, shape=(n_batch_fr,))

        X_fr = torch.from_numpy(self.objective_kwargs(t, mask_spikes_fr, stim=stim)['X'])
        u_fr = torch.einsum('tka,a->tk', X_fr, theta_g)
        r_fr = self.non_linearity_torch(u_fr)
        mask_spikes_fr = torch.from_numpy(mask_spikes_fr)
        
        if phi is not None:
            phi_fr = phi(t, mask_spikes_fr)
            gramian_fr_fr = phi_fr[:, None] * phi_fr[None, :]
            gramian_d_fr = phi_d[:, None] * phi_fr[None, :]
            
        if kernel is not None:
            gramian_fr_fr = kernel(t, mask_spikes_fr, mask_spikes_fr)
            gramian_d_fr = kernel(t, mask_spikes, mask_spikes_fr)
         
        n_batch_d, n_batch_fr = gramian_d_fr.shape[0], gramian_d_fr.shape[1]
#         log_likelihood_fr = torch.sum(u_fr * mask_spikes_fr.double() - dt * r_fr, 0)
        log_likelihood_fr = torch.sum(torch.log(1 - torch.exp(-dt * r_fr) + 1e-20) * mask_spikes_fr.double() - \
                                      dt * r_fr * (1 - mask_spikes_fr.double()), 0)

        idx_d = np.triu_indices(gramian_d_d.shape[0], k=1)
        idx_d = (torch.from_numpy(idx_d[0]), torch.from_numpy(idx_d[1]))
        idx_fr = np.triu_indices(gramian_fr_fr.shape[0], k=1)
        idx_fr = (torch.from_numpy(idx_fr[0]), torch.from_numpy(idx_fr[1]))
        
        if not(biased):
            mmd_grad = torch.mean(((log_likelihood_fr[:, None] + log_likelihood_fr[None, :]) * gramian_fr_fr)[idx_fr]) \
                       -2 * torch.mean(log_likelihood_fr[None, :] * gramian_d_fr)            
        else:
            mmd_grad = 2 * torch.sum((torch.mean(phi_d, 1) - torch.mean(phi_fr, 1)) * \
                                     torch.mean(log_likelihood_fr[None, :] * phi_fr, 1))
            
#         mmd_grad2 = 2 * torch.mean((log_likelihood_fr[None, :] * gramian_fr_fr)[idx_fr]) \
#                   -2 * torch.mean( log_likelihood_fr[None, :] * gramian_d_fr)
        
        if not(biased_mmd):
            mmd = torch.mean(gramian_d_d[idx_d]) + torch.mean(gramian_fr_fr[idx_fr]) - 2 * torch.mean(gramian_d_fr)
        else:
            if kernel is None:
                mmd = torch.sum((torch.mean(phi_d, 1) - torch.mean(phi_fr, 1))**2)
            else:
                mmd = torch.mean(gramian_d_d) + torch.mean(gramian_fr_fr) - 2 * torch.mean(gramian_d_fr)
    
#         mmd = self.mmd(mask_spikes, mask_spikes_fr, kernel=kernel)
#         mmd = self.mmd_spikes(mask_spikes, mask_spikes_fr, kernel=kernel)
#         mmd_grad = self._mmd_grad(mask_spikes, u_fr, r_fr, mask_spikes_fr, kernel=kernel, **mmd_kwargs)

        loss = lam_mmd * mmd_grad 
        
        return loss, mmd
    
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

    def mmd_spikes(self, mask_spikes, mask_spikes_fr, kernel=None):

        n_batch_te, n_batch_fr = mask_spikes.shape[1], mask_spikes_fr.shape[1]
        norm2_te, norm2_fr, mean_dot = 0, 0, 0

        for ii in range(n_batch_te):
            _s_dc_ii = mask_spikes[:, ii:ii + 1]
            _s_dc = mask_spikes[:, ii + 1:]
            _s_fr_ii = mask_spikes_fr[:, ii:ii + 1]
            _s_fr = mask_spikes_fr[:, ii + 1:]

            norm2_te += torch.sum(kernel(_s_dc_ii, _s_dc), 0) # sum is over samples
            norm2_fr += torch.sum(kernel(_s_fr_ii, _s_fr), 0)
            mean_dot += torch.sum(kernel(_s_dc_ii, _s_fr), 0)

        norm2_te /= (n_batch_te * (n_batch_te - 1) / 2)
        norm2_fr /= (n_batch_fr * (n_batch_fr - 1) / 2)
        mean_dot /= (n_batch_te * n_batch_fr)
        d = norm2_te + norm2_fr - 2 * mean_dot

        return d

    def _log_likelihood(self, dt, mask_spikes, X_dc):
#         theta_g = torch.cat((self.b, self.kappa_coefs, self.eta_coefs))
        theta_g = self.get_params()
        u_dc = torch.einsum('tka,a->tk', X_dc, theta_g)
        r_dc = self.non_linearity_torch(u_dc)
#         neg_log_likelihood = -(torch.sum(u_dc[mask_spikes]) - dt * torch.sum(r_dc))
        neg_log_likelihood = -(torch.sum(torch.log(1 - torch.exp(-dt * r_dc)) * mask_spikes.double()) - \
                               dt * torch.sum(r_dc * (1 - mask_spikes.double())))
        return neg_log_likelihood
    
    def train(self, t, mask_spikes, phi=None, kernel=None, stim=None, log_likelihood=False, lam_mmd=1e0, biased=False, 
              biased_mmd=False, optim=None, num_epochs=20, n_batch_fr=100, n_batch=1, verbose=False, mmd_kwargs=None, metrics=None):

        n_batch_dc = mask_spikes.shape[1]
    
        dt = torch.tensor([get_dt(t)])
        loss, nll, mmd = [], [], []
        
        X_dc = torch.from_numpy(self.objective_kwargs(t, mask_spikes, stim=stim)['X']).double()
        
        if phi is not None:
            phi_d = phi(t, mask_spikes)
            gramian_d_d = phi_d[:, None] * phi_d[None, :]
        else:
            phi_d = None
            
        if kernel is not None:
            gramian_d_d = kernel(t, mask_spikes, mask_spikes)
        
        _loss = torch.tensor([np.nan])
        mmd_kwargs = {} if mmd_kwargs is None else mmd_kwargs

        for epoch in range(num_epochs):
            if verbose:
                print('\r', 'epoch', epoch, 'of', num_epochs, 
                      'loss', np.round(_loss.item(), 10), end='')
            
            optim.zero_grad()
            
            batch_loss = 0
            batch_mmd = 0
            for ii in range(n_batch):
            
                _loss, _mmd = self(t, mask_spikes, phi_d=phi_d, gramian_d_d=gramian_d_d, phi=phi, kernel=kernel, 
                                   stim=stim, n_batch_fr=n_batch_fr, lam_mmd=lam_mmd, biased=biased, biased_mmd=biased_mmd)
                
                batch_loss = batch_loss + _loss
                batch_mmd = batch_mmd + _mmd
            
            batch_loss = batch_loss / n_batch
            batch_mmd = batch_mmd / n_batch
            
            if log_likelihood:
                _nll = self._log_likelihood(dt, mask_spikes, X_dc)
                batch_loss = batch_loss + _nll
                nll.append(_nll.item())
                        
            batch_loss.backward()
            optim.step()
            
            if metrics is not None:
                _metrics = metrics(self, t, mask_spikes, X_dc, n_batch_fr)
                if log_likelihood:
                    _metrics['nll'] = _nll.item()
                if epoch == 0:
                    metrics_list = {key:[val] for key, val in _metrics.items()}
                else:
                    for key, val in _metrics.items():
                        metrics_list[key].append(val)
            
#             theta_g = torch.cat([self.b, self.kappa_coefs, self.eta_coefs])
            theta_g = self.get_params()
            self.set_params(theta_g.data.detach().numpy())
            
            loss.append(batch_loss.item())
            mmd.append(batch_mmd)
            
        if metrics is None:
            metrics_list = None
        
        return loss, mmd, metrics_list

