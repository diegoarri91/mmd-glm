import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import Parameter
from torch.optim import Adam, LBFGS

from .base import GLM
from ..utils import get_dt, shift_array


class MMDGLM(GLM, torch.nn.Module):

    def __init__(self, u0, eta):
        torch.nn.Module.__init__(self)
        GLM.__init__(self, u0=u0, eta=eta)
        theta_g = torch.cat((torch.tensor([u0]).double(), torch.from_numpy(eta.coefs).double()), dim=0)
        self.register_parameter("theta_g", Parameter(theta_g))
    
    def forward(self, t, mask_spikes_te, X_te, y, lam_mmd, n_batch_fr, mmd_kwargs):
        
        dt = get_dt(t)
        self.theta_g.requires_grad = True
#         self.critic.set_requires_grad(True)
            
#             u_te, r_te = self.simulate_subthreshold(t, np.zeros(tuple(mask_spikes_te.shape)), mask_spikes_te.numpy())
        n_batch_te = mask_spikes_te.shape[1]
        u_te = torch.einsum('tka,a->tk', X_te, self.theta_g)
        r_te = torch.exp(u_te)
        
        u_fr, r_fr, mask_spikes_fr = self.sample(t, shape=(n_batch_fr,))
        n_spikes = np.mean(np.sum(mask_spikes_fr, 0))
        X_fr = torch.from_numpy(self.objective_kwargs(t, mask_spikes_fr)['X_te'])
        u_fr = torch.einsum('tka,a->tk', X_fr, self.theta_g)
        r_fr = torch.exp(u_fr)
            
        mask_spikes_fr = torch.from_numpy(mask_spikes_fr)
        mask_spikes = torch.cat((mask_spikes_te, mask_spikes_fr), dim=1).double()
        u = torch.cat((u_te, u_fr), dim=1).double()
        r = torch.cat((r_te, r_fr), dim=1).double()
        
        mmd = self.mmd(r, y, lam_mmd, **mmd_kwargs)
        neg_log_likelihood = -(torch.sum(u_te[mask_spikes_te]) - dt * torch.sum(r_te))
        loss = neg_log_likelihood + mmd  # compute wasserstein distance      
        
        return loss, mmd, n_spikes
    
    def mmd(self, r, y, lam, **kwargs):
        
        r_te, r_fr = r[:, y == 1], r[:, y == 0]
        n_batch_te, n_batch_fr = r_te.shape[1], r_fr.shape[1]
        r_sum_te, r_sum_fr = torch.sum(r_te, 1), torch.sum(r_fr, 1)
        
        # match average over trials. using linearity I avoid computing all products
        norm2_te = (torch.sum(r_sum_te**2) - torch.sum(r_te**2)) / (n_batch_te * (n_batch_te - 1))
        norm2_fr = (torch.sum(r_sum_fr**2) - torch.sum(r_fr**2)) / (n_batch_fr * (n_batch_fr - 1))
        mean_dot = torch.sum(r_sum_te * r_sum_fr, 0) / (n_batch_te * n_batch_fr)
        distance = lam * (norm2_te + norm2_fr - 2 * mean_dot)

        # match average over trials. BIASED estimate
#         distance = lam * torch.sum((r_sum_te / n_batch_te - r_sum_fr / n_batch_fr) ** 2)
        
        # match average over trials. computing all products (for testing)
#         norm2_te, norm2_fr, mean_dot = 0, 0, 0
#         for ii in range(n_batch_te):
#             for jj in range(n_batch_fr):
#                 if jj > ii:
#                     norm2_te += torch.sum(r_te[:, ii] * r_te[:, jj])
#                     norm2_fr += torch.sum(r_fr[:, ii] * r_fr[:, jj])
#                 mean_dot += torch.sum(r_te[:, ii] * r_te[:, jj])              
#         norm2_te = norm2_te / (n_batch_te * (n_batch_te - 1))       
#         norm2_fr = norm2_fr / (n_batch_fr * (n_batch_fr - 1))
#         mean_dot = mean_dot / (n_batch_te * n_batch_fr)
#         distance = lam * (norm2_te + norm2_fr - 2 * mean_dot)

        # match second moment. BIASED estimate
#         distance = lam * torch.sum((torch.mean(r_te**2, 1) - torch.mean(r_fr**2, 1)) ** 2)
        
        # match first and second moment. BIASED estimate
#         distance = lam * torch.sum((torch.mean(r_te, 1) - torch.mean(r_fr, 1) + \
#                                     0.5 * (torch.mean(r_te**2, 1)  - torch.mean(r_fr**2, 1))) ** 2)

        # match all moments (exponential). BIASED estimate 
#         distance = lam * torch.sum((torch.mean(torch.exp(r_te), 1) - torch.mean(torch.exp(r_fr), 1)) ** 2)
        
#         sd = kwargs['sd']
#         distance = lam * torch.exp(-(norm2_te + norm2_fr - 2 * mean_dot)**2 / 2 * sd**2)
        
#         import torch.nn.functional  as F

#         weight = torch.from_numpy(KernelFun.rbf(5).interpolate(np.arange(0, 31, 1) - 15))[None, 2, :]
#         norm2_te = (torch.sum(r_sum_te**2) - torch.sum(r_te**2)) / (n_batch_te * (n_batch_te - 1))
#         norm2_fr = (torch.sum(r_sum_fr**2) - torch.sum(r_fr**2)) / (n_batch_fr * (n_batch_fr - 1))
#         mean_dot = torch.sum(r_sum_te * r_sum_fr, 0) / (n_batch_te * n_batch_fr)
#         aux = F.conv1d(mask, weight, padding=(weight.shape[2] - 1) // 2)
#         , , kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'
#         F.conv1d(2, , stride=1)
        
        return distance
    
    def train(self, t, mask_spikes, y, lam_mmd=1e0, lr=1e-3, num_epochs=20, n_batch_fr=100, verbose=False,
              mmd_kwargs=None):
        optim = Adam(self.parameters(), lr=lr)
        mask_spikes_te = mask_spikes.clone()
        n_batch_te = mask_spikes_te.shape[1]
        dt = torch.tensor([get_dt(t)])
        loss, mmd, n_spikes = [], [], []
        X_te = torch.from_numpy(self.objective_kwargs(t, mask_spikes_te)['X_te'])
        _loss = torch.tensor([np.nan])
        mmd_kwargs = {} if mmd_kwargs is None else mmd_kwargs
        for epoch in range(num_epochs):
            if verbose:
                print('\r', 'epoch', epoch, 'of', num_epochs, 
                      'loss', np.round(_loss.item(), 4), end='')
            
            optim.zero_grad()
            _loss, _mmd, _n_spikes = self(t, mask_spikes, X_te, y, lam_mmd=lam_mmd, n_batch_fr=n_batch_fr, mmd_kwargs=mmd_kwargs)
            _loss.backward()  # compute gradient
            optim.step()
            self.set_params(self.theta_g.data.detach().numpy())
            loss.append(_loss.item())
            mmd.append(_mmd.item())
            n_spikes.append(_n_spikes)
            
        self.set_params(self.theta_g.data.detach().numpy())
        return loss, mmd, n_spikes

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

