import numpy as np
import torch

from .base import GLM
from ..utils import get_dt, shift_array

dic_nonlinearities = {'exp': lambda x: torch.exp(x), 'log_exp': lambda x: torch.log(1 + torch.exp(x))}


class CIMMDGLM(GLM, torch.nn.Module):

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
            
    def forward(self, t, stim=None, n_batch_fr=None):
        
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
        
        return r_fr, mask_spikes_fr, X_fr
    
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
        theta_g = self.get_params()
        u_dc = torch.einsum('tka,a->tk', X_dc, theta_g)
        r_dc = self.non_linearity_torch(u_dc)
#         neg_log_likelihood = -(torch.sum(u_dc[mask_spikes]) - dt * torch.sum(r_dc))
        neg_log_likelihood = -(torch.sum(torch.log(1 - torch.exp(-dt * r_dc) + 1e-24) * mask_spikes.double()) - \
                               dt * torch.sum(r_dc * (1 - mask_spikes.double())))
        return neg_log_likelihood
    
    def train(self, t, mask_spikes, phi=None, kernel=None, stim=None, log_likelihood=False, lam_mmd=1e0, biased=False, 
              score_term=False, optim=None, clip=None, num_epochs=20, n_batch_fr=100, verbose=False, metrics=None, n_metrics=25):

        n_d = mask_spikes.shape[1]
    
        dt = torch.tensor([get_dt(t)])
        loss, nll = [], []
        
        X_dc = torch.from_numpy(self.objective_kwargs(t, mask_spikes, stim=stim)['X']).double()
        
        if phi is None:
            idx_fr = np.triu_indices(n_batch_fr, k=1)
            idx_fr = (torch.from_numpy(idx_fr[0]), torch.from_numpy(idx_fr[1]))
            idx_d = np.triu_indices(n_d, k=1)
            idx_d = (torch.from_numpy(idx_d[0]), torch.from_numpy(idx_d[1]))
                
        _loss = torch.tensor([np.nan])

        for epoch in range(num_epochs):
            if verbose:
                print('\r', 'epoch', epoch, 'of', num_epochs, 
                      'loss', np.round(_loss.item(), 10), end='')
            
            optim.zero_grad()
            
            theta_g = self.get_params()
            u_dc = torch.einsum('tka,a->tk', X_dc, theta_g)
            r_dc = self.non_linearity_torch(u_dc)
            
            r_fr, mask_spikes_fr, X_fr = self(t, stim=stim, n_batch_fr=n_batch_fr)
            
            if score_term:
                score = torch.sum(torch.log(1 - torch.exp(-dt * r_fr) + 1e-24) * mask_spikes_fr.double(), 0) - \
                          dt * torch.sum(r_fr * (1 - mask_spikes_fr.double()), 0)

            if phi is not None:
                phi_d = phi(t, r_dc, model=self)
                phi_fr = phi(t, r_fr, model=self)
#                 print(phi_d.shape, phi_fr.shape, n_d, n_batch_fr)
                if not biased:
                    sum_phi_d = torch.sum(phi_d, 1)
                    sum_phi_fr = torch.sum(phi_fr, 1)

                    norm2_d = (torch.sum(sum_phi_d**2) - torch.sum(phi_d**2)) / (n_d * (n_d - 1))
                    norm2_fr = (torch.sum(sum_phi_fr**2) - torch.sum(phi_fr**2)) / (n_batch_fr * (n_batch_fr - 1))
                    mean_dot = torch.sum(sum_phi_d * sum_phi_fr) / (n_d * n_batch_fr)
                    
                    if score_term:
                        sum_score_phi_fr = torch.sum(score[None, :] * phi_fr.detach(), 1)
                        sum_score_ker_fr = torch.sum(sum_phi_fr.detach() * sum_score_phi_fr) - \
                                           torch.sum(score[None, :] * phi_fr.detach()**2)
                        norm2_fr = norm2_fr + 2 * sum_score_ker_fr / (n_batch_fr * (n_batch_fr - 1))
                        mean_dot = mean_dot + torch.sum(sum_phi_d.detach() * sum_score_phi_fr)/ (n_d * n_batch_fr)
                        
                    mmd_grad = norm2_d + norm2_fr - 2 * mean_dot
                else:
                    mmd_grad = torch.sum((torch.mean(phi_d, 1) - torch.mean(phi_fr, 1))**2)
#                 print(mmd_grad)
            else:
                gramian_d_d = kernel(t, r_dc, r_dc)
                gramian_fr_fr = kernel(t, r_fr, r_fr)
                gramian_d_fr = kernel(t, r_dc, r_fr)
                if not biased:
#                     mmd_grad = torch.mean(gramian_d_d[idx_d]) + \
#                                torch.mean(((score[:, None] + score[None, :]) * gramian_fr_fr.detach() + gramian_fr_fr)[idx_fr]) \
#                                 -2 * torch.mean(score[None, :] * gramian_d_fr.detach()  + gramian_d_fr)
                    mmd_grad = torch.mean(gramian_d_d[idx_d]) + torch.mean(gramian_fr_fr[idx_fr]) \
                                -2 * torch.mean(gramian_d_fr)
                else:
                    mmd_grad = torch.mean(gramian_d_d) + torch.mean(gramian_fr_fr) \
                                -2 * torch.mean(gramian_d_fr)
            
            _loss = lam_mmd * mmd_grad
            
            if log_likelihood:
                _nll = self._log_likelihood(dt, mask_spikes, X_dc)
#                 print(_loss, _nll)
                nll.append(_nll.item())
                _loss = _loss + _nll
#                 print(_loss)
                        
            if metrics is not None and (epoch % n_metrics) == 0:
                _metrics = metrics(self, t, mask_spikes, mask_spikes_fr)
                if phi is not None:
                    _metrics['mmd'] = (torch.sum((torch.mean(phi_d.detach(), 1) - torch.mean(phi_fr.detach(), 1))**2)).item()
                else:
                    _metrics['mmd'] = torch.mean(gramian_d_d.detach()[idx_d]) + torch.mean(gramian_fr_fr.detach()[idx_fr]) \
                                      - 2 * torch.mean(gramian_d_fr.detach())
                if epoch == 0:
                    metrics_list = {key:[val] for key, val in _metrics.items()}
                else:
                    for key, val in _metrics.items():
                        metrics_list[key].append(val)
        
            _loss.backward()
            
            if clip is not None:
                torch.nn.utils.clip_grad_value_(self.parameters(), clip)
            
            optim.step()
#             if scheduler is not None:
#                 scheduler.step()
            
            theta_g = self.get_params()
            self.set_params(theta_g.data.detach().numpy())
            
            loss.append(_loss.item())
            
        if metrics is None:
            metrics_list = None
        
        return loss, nll, metrics_list

