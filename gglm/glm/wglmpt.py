import numpy as np
import torch
from torch.nn import Parameter
from torch.optim import Adam

from .base import GLM
from ..utils import get_dt, shift_array


class WGLM(GLM, torch.nn.Module):

    def __init__(self, u0, eta, critic):
#         super().__init__(u0=u0, eta=eta)
#         torch.nn.Module.__init__()
#         self.__init__()
        torch.nn.Module.__init__(self)
        GLM.__init__(self, u0=u0, eta=eta)
        self.critic = critic
        theta_g = torch.cat((torch.tensor([u0]).double(), torch.from_numpy(eta.coefs).double()), dim=0)
        self.register_parameter("theta_g", Parameter(theta_g))
    
    def forward(self, t, mask_spikes_te, X_te, X_r_spk_te, y, lam_w, n_batch_fr, lr_c, num_epochs_c):
        
        dt = get_dt(t)
        self.theta_g.requires_grad = True
        self.critic.theta.requires_grad = True
        self.critic.theta2.requires_grad = True

#             u_te, r_te = self.simulate_subthreshold(t, np.zeros(tuple(mask_spikes_te.shape)), mask_spikes_te.numpy())
        n_batch_te = mask_spikes_te.shape[1]
        u_te = torch.einsum('tka,a->tk', X_te, self.theta_g)
        r_te = torch.exp(u_te)
        
        u_fr, r_fr, mask_spikes_fr = self.sample(t, shape=(n_batch_fr,))
        X_fr = torch.from_numpy(self.objective_kwargs(t, mask_spikes_fr)['X_te'])
        u_fr = torch.einsum('tka,a->tk', X_fr, self.theta_g)
        r_fr = torch.exp(u_fr)
            
        mask_spikes_fr = torch.from_numpy(mask_spikes_fr)
        mask_spikes = torch.cat((mask_spikes_te, mask_spikes_fr), dim=1)
        u = torch.cat((u_te, u_fr), dim=1)
        r = torch.cat((r_te, r_fr), dim=1)
        self.critic.get_conv_r_spikes(t, mask_spikes_fr)     
        self.critic.X_r_spk = torch.cat((X_r_spk_te, self.critic.X_r_spk), dim=1)
            
        self.theta_g.requires_grad = False

        _neg_w_distance = self.critic.train(t, mask_spikes, u, r, y, lam_w, lr=lr_c, num_epochs=num_epochs_c, verbose=False)

        self.theta_g.requires_grad = True
        self.critic.theta.requires_grad = False
        self.critic.theta2.requires_grad = False

          # zero gradient
        w_distance = self.critic.forward(t, mask_spikes, u, r, y, lam_w, neg=False)
        neg_log_likelihood = -(torch.sum(u_te[mask_spikes_te]) - dt * torch.sum(r_te))
        loss = neg_log_likelihood + w_distance  # compute wasserstein distance      
        
        return loss, w_distance, _neg_w_distance
    
    def train(self, t, mask_spikes, y, lam_w=1e0, lr_g=1e-3, lr_c=1e-2, num_epochs=20, num_epochs_c=10, n_batch_fr=100, verbose=False):
        optim = Adam(self.parameters(), lr=lr_g)  # optimizer
        mask_spikes_te = mask_spikes.clone()
        n_batch_te = mask_spikes_te.shape[1]
        dt = torch.tensor([get_dt(t)])
#         optim_critic = Adam(self.parameters(), lr=lr_c)  # optimizer
        loss = []
        w_distance = []
        neg_w_distance = []
        X_te = torch.from_numpy(self.objective_kwargs(t, mask_spikes_te)['X_te'])
        self.critic.get_conv_r_spikes(t, mask_spikes)
        X_r_spk_te = self.critic.X_r_spk.clone()
        _loss = torch.tensor([np.nan])
#         for epoch in tqdm(range(num_epochs)):
        for epoch in range(num_epochs):
            if verbose:
                print('\r', 'epoch', epoch, 'of', num_epochs, 
                      'loss', np.round(_loss.item(), 4), end='')
            
            optim.zero_grad()
            _loss, _w_distance, _neg_w_distance = self.forward(t, mask_spikes, X_te, X_r_spk_te, y, lam_w=lam_w, n_batch_fr=n_batch_fr, 
                                                                   lr_c=lr_c, num_epochs_c=num_epochs_c)
            _loss.backward()  # compute gradient
            optim.step()  # take gradient step
            loss.append(_loss.item())
            neg_w_distance += _neg_w_distance
            w_distance.append(_w_distance.item())
            
        self.set_params(self.theta_g.data.detach().numpy())
        return loss, w_distance, neg_w_distance
    
#     def forward(self, dt, u_te, r_te, mask_spikes_te):
        """Compute log-likelihood + wasserstein distance distance"""

#         a = self.critic.transform(t, mask_spikes, r, y)
        
#         theta_g = torch.from_numpy(self.get_params())
        
#         n_samples_te = u_te.shape[1]
#         t = np.arange(X_te.shape[0]) * dt

#         u_te = torch.einsum('tka,a->tk', X_te, self.theta_g)
#         r_te = torch.exp(u_te)
        
#         u_fr = torch.einsum('tka,a->tk', X_fr, self.theta_g)
#         r_fr = torch.exp(u_fr)
        
#         mask_spikes = torch.cat((mask_spikes_te, mask_spikes_fr), axis=1)
#         u = torch.cat((u_te, u_fr), axis=1)
#         r = torch.cat((r_te, r_fr), axis=1)
#         y = torch.cat((torch.ones(n_samples_te), torch.zeros(n_samples_fr)))
        
#         neg_log_likelihood = -(torch.sum(u_te[mask_spikes_te]) - dt * torch.sum(r_te))

#         return neg_log_likelihood

    def objective_kwargs(self, t, mask_spikes, stim=None, n_samples_fr=100, newton_kwargs_critic=None):

        n_kappa = 0
        n_eta = 0 if self.eta is None else self.eta.nbasis

        X_te = np.zeros(mask_spikes.shape + (1 + n_kappa + n_eta,))
        X_te[:, :, 0] = -1.

        if self.eta is not None:
            args = np.where(shift_array(mask_spikes, 1, fill_value=False))
            t_spk = (t[args[0]],) + args[1:]
            n_eta = self.eta.nbasis
            X_eta = self.eta.convolve_basis_discrete(t, t_spk, shape=mask_spikes.shape)
            X_te[:, :, n_kappa + 1:] = -X_eta

        obj_kwargs = dict(dt=get_dt(t), X_te=X_te, mask_spikes_te=mask_spikes, n_samples_fr=n_samples_fr,
                          newton_kwargs_critic=newton_kwargs_critic)

        return obj_kwargs

