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
                lam_mmd=1, biased=False, biased_mmd=False, control_variates=False):
        
        dt = get_dt(t)
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
        
#         print(theta_g, mask_spikes_fr)
        
        if phi is not None:
            phi_fr = phi(t, mask_spikes_fr)
            gramian_fr_fr = phi_fr[:, None] * phi_fr[None, :]
            gramian_d_fr = phi_d[:, None] * phi_fr[None, :]
            
        if kernel is not None:
            gramian_fr_fr = kernel(t, mask_spikes_fr, mask_spikes_fr)
            gramian_d_fr = kernel(t, mask_spikes, mask_spikes_fr)
#             print(mask_spikes_fr.shape, gramian_fr_fr.shape)

        r_fr_s, r_fr_ns = r_fr[mask_spikes_fr], r_fr[~mask_spikes_fr]
        log_likelihood_fr = torch.sum(torch.log(1 - torch.exp(-dt * r_fr) + 1e-24) * mask_spikes_fr.double() - \
                                      dt * r_fr * (1 - mask_spikes_fr.double()), 0)
        exp_r_s = torch.exp(r_fr.detach() * dt)
        g_log_likelihood_fr = dt * torch.einsum('tka,tk->ka', X_fr, r_fr.detach() / (exp_r_s - 1 - 1e-24) * mask_spikes_fr.double()) \
                                - dt * torch.einsum('tka,tk->ka', X_fr, r_fr.detach() * (1 - mask_spikes_fr.double()))

#         r_fr_s, r_fr_ns = r_fr[mask_spikes_fr], r_fr[~mask_spikes_fr] # I would have to do it in an iterative way to get it for each sample
#         log_likelihood_fr = torch.sum(torch.log(1 - torch.exp(-dt * r_fr_s) + 1e-24) - \
#                                       dt * r_fr_ns, 0)

        idx_d = np.triu_indices(gramian_d_d.shape[0], k=1)
        idx_d = (torch.from_numpy(idx_d[0]), torch.from_numpy(idx_d[1]))
        idx_fr = np.triu_indices(gramian_fr_fr.shape[0], k=1)
        idx_fr = (torch.from_numpy(idx_fr[0]), torch.from_numpy(idx_fr[1]))

        if control_variates:
            mean_g_log_likelihood = torch.mean(g_log_likelihood_fr)
            mean_gramian_d_fr = torch.mean(gramian_d_fr)
            mean_gramian_fr_fr = torch.mean(gramian_fr_fr[idx_fr])
            var_g_log_likelihood_fr = torch.mean(g_log_likelihood_fr**2, dim=0) - mean_g_log_likelihood**2
            a_d_fr = (torch.mean(g_log_likelihood_fr[None, :, :] * gramian_d_fr[:, :, None], dim=1) - \
                      mean_g_log_likelihood * mean_gramian_d_fr) / var_g_log_likelihood_fr # shape=(n_data, n_pars)
#             a_fr_fr = (torch.stack([g_log_likelihood_fr[ii, :] * gramian_fr_fr[ii, ] for ii in range(n_batch_fr)], 0)
                
#                 torch.mean(g_log_likelihood_fr[None, :, :] * gramian_fr_fr[:, :, None], dim=1) - \
#                        mean_g_log_likelihood * mean_gramian_fr_fr) / var_g_log_likelihood_fr # shape=(n_batch_fr, n_pars)
            _g = gramian_fr_fr.clone()
            _g.fill_diagonal_(0)
#             a_fr_fr = (torch.sum(g_log_likelihood_fr[None, :, :] * _g[:, :, None], dim=1) / (n_batch_fr * (n_batch_fr - 1)) - \
#                        mean_g_log_likelihood * mean_gramian_fr_fr) / var_g_log_likelihood_fr # shape=(n_batch_fr, n_pars)
            a_fr_fr = (torch.sum(g_log_likelihood_fr[None, :, :] * _g[:, :, None], dim=1) / (n_batch_fr - 1) - \
                       mean_g_log_likelihood * mean_gramian_fr_fr) / var_g_log_likelihood_fr # shape=(n_batch_fr, n_pars)
        
        if not(biased):
            if control_variates:
                mmd_grad = torch.mean(((log_likelihood_fr[:, None] + log_likelihood_fr[None, :]) * gramian_fr_fr)[idx_fr]) \
                           -2 * torch.mean(log_likelihood_fr[None, :] * gramian_d_fr)
                grad_theta_g = torch.mean((g_log_likelihood_fr[:, None, :] * (gramian_fr_fr[:, :, None] - a_fr_fr[:, None, :])  \
                                         + g_log_likelihood_fr[None, :, :] * (gramian_fr_fr[:, :, None] - a_fr_fr[None, :, :]))[idx_fr], \
                                          dim=0) \
                               -2 * torch.mean(g_log_likelihood_fr[None, :, :] * (gramian_d_fr[:, :, None] - a_d_fr[:, None, :]), dim=(0, 1))
            else:
                mmd_grad = torch.mean(((log_likelihood_fr[:, None] + log_likelihood_fr[None, :]) * gramian_fr_fr)[idx_fr]) \
                           -2 * torch.mean(log_likelihood_fr[None, :] * gramian_d_fr)
                grad_theta_g = torch.mean(((g_log_likelihood_fr[:, None, :] + g_log_likelihood_fr[None, :, :]) * \
                                           gramian_fr_fr[:, :, None])[idx_fr], dim=0) \
                               -2 * torch.mean(g_log_likelihood_fr[None, :, :] * gramian_d_fr[:, :, None], dim=(0, 1))
        else:
            pass
#             mmd_grad = 2 * torch.sum((torch.mean(phi_d, 1) - torch.mean(phi_fr, 1)) * \
#                                      torch.mean(log_likelihood_fr[None, :] * phi_fr, 1))grad_theta_g
        
        if not(biased_mmd):
            mmd = torch.mean(gramian_d_d[idx_d]) + torch.mean(gramian_fr_fr[idx_fr]) - 2 * torch.mean(gramian_d_fr)
        else:
            if kernel is None:
                mmd = torch.sum((torch.mean(phi_d, 1) - torch.mean(phi_fr, 1))**2)
            else:
                mmd = torch.mean(gramian_d_d) + torch.mean(gramian_fr_fr) - 2 * torch.mean(gramian_d_fr)

        loss = lam_mmd * mmd_grad
        grad_theta_g = lam_mmd * grad_theta_g
        
        return loss, grad_theta_g, mmd
    
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
              biased_mmd=False, control_variates=False, lr=None, num_epochs=20, n_batch_fr=100, verbose=False, 
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
        
        _loss = torch.tensor([np.nan])
        theta_g = self.get_params()
        
        for epoch in range(num_epochs):
            if verbose:
                print('\r', 'epoch', epoch, 'of', num_epochs, 
                      'loss', np.round(_loss.item(), 10), end='')
            
            _loss, grad_theta_g, _mmd = self(t, mask_spikes, phi_d=phi_d, gramian_d_d=gramian_d_d, phi=phi, kernel=kernel, 
                                   stim=stim, n_batch_fr=n_batch_fr, lam_mmd=lam_mmd, biased=biased, biased_mmd=biased_mmd, 
                                   control_variates=control_variates)
            
            if log_likelihood:
                _nll = self._log_likelihood(dt, mask_spikes, X_dc)
                batch_loss = batch_loss + _nll
                nll.append(_nll.item())
                        
#             batch_loss.backward()
            theta_g = theta_g - lr[epoch] * grad_theta_g
#             optim.step()
            
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
#             theta_g = self.get_params()
            self.set_params(theta_g.data.detach().numpy())
            
            loss.append(_loss.item())
            mmd.append(_mmd)
            
        if metrics is None:
            metrics_list = None
        
        return loss, mmd, metrics_list

