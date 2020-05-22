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
#         theta_g = torch.zeros(1 + n_kappa + n_eta)
#         theta_g[0] = torch.tensor([u0])
        b = torch.tensor([u0]).double()
        kappa_coefs = torch.zeros(n_kappa + n_eta).double()
        if self.kappa is not None:
#             theta_g[1:1 + n_kappa] = torch.from_numpy(kappa.coefs)
            kappa_coefs[:n_kappa] = torch.from_numpy(kappa.coefs)
        if self.eta is not None:
            kappa_coefs[n_kappa:] = torch.from_numpy(eta.coefs)
#         theta_g = theta_g.double()
#         self.register_parameter("theta_g", Parameter(theta_g))
        self.register_parameter("b", Parameter(b))
        self.register_parameter("kappa_coefs", Parameter(kappa_coefs))
    
    def forward(self, t, mask_spikes, phi_d=None, gramian_d_d=None, phi=None, kernel=None, stim=None, n_batch_fr=None, lam_mmd=1):
        
        dt = get_dt(t)
#         self.theta_g.requires_grad = True
        theta_g = torch.cat((self.b, self.kappa_coefs))
        
#         u_dc = torch.einsum('tka,a->tk', X_dc, self.theta_g)
#         r_dc = self.non_linearity_torch(u_dc)
#         phi_dc = kernel.convolve_continuous(t, mask_spikes.numpy())
        
        # TODO. I am calculating u_fr and r_fr twice because I can't backpropagate through my current sample function. change this
        if stim is not None:
#             print('j')
            _, _, mask_spikes_fr = self.sample(t, stim=stim)
        else:
            _, _, mask_spikes_fr = self.sample(t, shape=(n_batch_fr,))
            
#         _, _, mask_spikes_fr = self.sample(t, stim=stim, shape=(n_batch_fr, ))    
#         print(mask_spikes_fr.shape)
        X_fr = torch.from_numpy(self.objective_kwargs(t, mask_spikes_fr, stim=stim)['X'])
#         print(X_fr.shape, stim.shape)
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
            
#         phi_fr = kernel.convolve_continuous(t, mask_spikes_fr)
#         isi_fr = SpikeTrain(t, mask_spikes_fr).isi_distribution(concatenate=False)
#         isi_fr = torch.tensor([np.mean(_isi) for _isi in isi_fr]).double()
        
#         gramian_fr_fr = torch.from_numpy(np.sum(phi_fr[:, :, None] * phi_fr[:, None, :], 0))
#         gramian_d_fr = torch.from_numpy(np.sum(phi_d[:, :, None] * phi_fr[:, None, :], 0))
        
#         print(float(torch.mean(torch.sqrt(torch.diag(gramian_d_d)))), float(torch.mean(isi_fr)))
#         gramian_fr_fr = isi_fr[:, None] * isi_fr[None, :]
#         gramian_d_fr = torch.sqrt(torch.diag(gramian_d_d))[:, None] * isi_fr[None, :]
        
#         gramian_fr_fr = torch.exp(-(isi_fr[:, None] - isi_fr[None, :])**2 / 1e2)
#         gramian_d_fr = torch.exp(-(phi_d[:, None] - isi_fr[None, :])**2 / 1e2)
        
        n_batch_d, n_batch_fr = gramian_d_fr.shape[0], gramian_d_fr.shape[1]
#         log_likelihood_fr = torch.sum(u_fr * mask_spikes_fr.double() - dt * r_fr, 0)
        log_likelihood_fr = torch.sum(torch.log(1 - torch.exp(-dt * r_fr)) * mask_spikes_fr.double() - \
                                      dt * r_fr * (1 - mask_spikes_fr.double()), 0)
        
        idx_d = np.triu_indices(gramian_d_d.shape[0], k=1)
        idx_d = (torch.from_numpy(idx_d[0]), torch.from_numpy(idx_d[1]))
        idx_fr = np.triu_indices(gramian_fr_fr.shape[0], k=1)
        idx_fr = (torch.from_numpy(idx_fr[0]), torch.from_numpy(idx_fr[1]))
        
        mmd_grad = torch.mean(((log_likelihood_fr[:, None] + log_likelihood_fr[None, :]) * gramian_fr_fr)[idx_fr]) \
                   -2 * torch.mean(log_likelihood_fr[None, :] * gramian_d_fr)
#         mmd_grad = torch.mean(((log_likelihood_fr[:, None] + log_likelihood_fr[None, :]) * gramian_fr_fr)[idx_fr])
        
#         self.theta_g.requires_grad = False
        mmd = torch.mean(gramian_d_d[idx_d]) + torch.mean(gramian_fr_fr[idx_fr]) - 2 * torch.mean(gramian_d_fr)
    
#         mmd_grad = mmd_grad ** 0.5
#         mmd = mmd ** 0.5
#         print(gramian_d_d.shape, gramian_fr_fr.shape, gramian_d_fr.shape)
#         mmd_grad = torch.sum(((log_likelihood_fr[:, None] + log_likelihood_fr[None, :]) * gramian_fr_fr)[idx_fr]) / (n_batch_fr * (n_batch_fr - 1) / 2) \
#                    -2 * torch.sum(log_likelihood_fr[None, :] * gramian_d_fr) / (n_batch_d * n_batch_fr)
        
#         self.theta_g.requires_grad = False
#         _idx = (torch.zeros(idx[0].shape[0], dtype=int), idx[0], idx[1])
#         _aux = torch.einsum('tka,tk->ak', X_fr, (mask_spikes_fr.double() - r_fr))
#         _grad = torch.sum(((_aux[:, :, None] + _aux[:, None, :]) * gramian_fr_fr[None, :, :])[_idx]) / (n_batch_fr * (n_batch_fr - 1) / 2) \
#                    -2 * torch.sum(_aux[:, None, :] * gramian_dc_fr[None, :, :], dim=(1, 2)) / (n_batch_dc * n_batch_fr)
#         self.theta_g.requires_grad = True
#         print('\nmy_grad', lam_mmd * _grad)
        
#         mmd_grad = -2 * torch.sum(aux_grad[None, :] * gramian_dc_fr) / (n_batch_dc * n_batch_fr)
        
#         self.theta_g.requires_grad = False
#         neg_log_likelihood = -(torch.sum(u_dc[mask_spikes]) - dt * torch.sum(r_dc))
#         self.theta_g.requires_grad = True
#         neg_log_likelihood = -(torch.sum(torch.log(1 - torch.exp(-dt * r_dc[mask_spikes_dc]))) - dt * torch.sum(r_dc[~mask_spikes_dc]))
    
#         mmd = self.mmd(mask_spikes, mask_spikes_fr, kernel=kernel)
#         mmd = self.mmd_spikes(mask_spikes, mask_spikes_fr, kernel=kernel)
#         mmd_grad = self._mmd_grad(mask_spikes, u_fr, r_fr, mask_spikes_fr, kernel=kernel, **mmd_kwargs)

#         loss = 0 * neg_log_likelihood + lam_mmd * mmd_grad 
        loss = lam_mmd * mmd_grad 
        
        return loss, mmd

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
    
    def _mmd_grad(self, mask_spikes, u_fr, r_fr, mask_spikes_fr, kernel=None, **kwargs):

        n_batch_dc, n_batch_fr = mask_spikes.shape[1], mask_spikes_fr.shape[1]
        norm2_fr, mean_dot = 0, 0
        for ii in range(n_batch_fr - 1):
            _s_dc = mask_spikes[:, ii + 1:]
    
            _s_fr_ii = mask_spikes_fr[:, ii:ii + 1]
            _s_fr = mask_spikes_fr[:, ii + 1:]
            
            _r_fr_ii = r_fr[:, ii:ii + 1]
            _r_fr = r_fr[:, ii + 1:]
            
            _u_fr_ii = u_fr[:, ii:ii + 1]
            _u_fr = u_fr[:, ii + 1:]
            
#             print(torch.sum(_u_fr * _s_fr - _r_fr, 0).shape, torch.sum(_u_fr_ii * _s_fr_ii - _r_fr_ii, 0)[:, None].shape, kernel(_s_fr_ii, _s_fr).shape, '\n')
            norm2_fr += torch.sum((torch.sum(_u_fr_ii * _s_fr_ii - _r_fr_ii, 0) + \
                                   torch.sum(_u_fr * _s_fr - _r_fr, 0)) \
                                  * kernel(_s_fr_ii, _s_fr), 0) # sum is over samples
            mean_dot += torch.sum(torch.sum(_u_fr_ii * _s_fr_ii - _r_fr_ii, 0) * kernel(_s_fr_ii, _s_dc), 0)

        norm2_fr /= (n_batch_fr * (n_batch_fr - 1) / 2)
        mean_dot /= (n_batch_dc * n_batch_fr)
#         d = norm2_fr - 2 * mean_dot
        d = 0 - 2 * mean_dot

        return d
    
    def train(self, t, mask_spikes, phi=None, kernel=None, stim=None, lam_mmd=1e0, optim=None, num_epochs=20, n_batch_fr=100, verbose=False,
              mmd_kwargs=None, metrics=None):

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
#             isi_d = SpikeTrain(t, mask_spikes.numpy()).isi_distribution(concatenate=False)
#             isi_d = torch.tensor([np.mean(_isi) for _isi in isi_d]).double()
#             phi_d = isi_d.clone()
#         print(gramian_d_d)
        
#         phi_d = kernel.convolve_continuous(t, mask_spikes.numpy())
#         gramian_d_d = torch.from_numpy(np.sum(phi_d[:, :, None] * phi_d[:, None, :], 0))
        
        _loss = torch.tensor([np.nan])
        mmd_kwargs = {} if mmd_kwargs is None else mmd_kwargs
        for epoch in range(num_epochs):
            if verbose:
                print('\r', 'epoch', epoch, 'of', num_epochs, 
                      'loss', np.round(_loss.item(), 10), end='')
            
            optim.zero_grad()
            
            _loss, _mmd = self(t, mask_spikes, phi_d=phi_d, gramian_d_d=gramian_d_d, phi=phi, kernel=kernel, 
                               stim=stim, n_batch_fr=n_batch_fr, lam_mmd=lam_mmd)
            
#             if hasattr(self, 'metrics'):
            if metrics is not None:
                _metrics = metrics(self, t, mask_spikes, X_dc, n_batch_fr)
                if epoch == 0:
                    metrics_list = {key:[val] for key, val in _metrics.items()}
                else:
                    for key, val in _metrics.items():
                        metrics_list[key].append(val)
                        
#             print('\ngrad1', self.theta_g.grad)                        
            _loss.backward()
#             print('\ngrad2', self.theta_g.grad)
            optim.step()
            theta_g = torch.cat([self.b, self.kappa_coefs])
            self.set_params(theta_g.data.detach().numpy())
            
            loss.append(_loss.item())
#             nll.append(_nll.item())
            mmd.append(_mmd)
#             n_spikes.append(_n_spikes)
            
        if metrics is None:
            metrics_list = None
        
        return loss, mmd, metrics_list

