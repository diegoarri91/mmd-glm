from functools import partial

import numpy as np
import torch
from torch.nn import Parameter
from torch.optim import Adam
# from tqdm import tqdm
# from tqdm.notebook import tqdm


from .utils import get_dt, shift_array


class CriticPyTorch(torch.nn.Module):
    
    def __init__(self, r_kernel, theta, theta2, clip, clip2):
        super().__init__()
        self.r_kernel = r_kernel
        self.register_parameter("theta", Parameter(theta))
        self.register_parameter("theta2", Parameter(theta2))
        self.clip = clip
        self.clip2 = clip2
        
#     def copy(self):
#         u_kernel = None if self.u_kernel is None else self.u_kernel.copy()
#         u_spk_kernel = None if self.u_spk_kernel is None else self.u_spk_kernel.copy()
#         r_kernel = None if self.r_kernel is None else self.r_kernel.copy()
# #         r_spk_kernel = None if self.r_spk_kernel is None else self.r_spk_kernel.copy()
# #         beta = None if self.beta is None else self.beta.copy()
#         return self.__class__(u_kernel=u_kernel, u_spk_kernel=u_spk_kernel, r_kernel=r_kernel, r_spk_kernel=r_spk_kernel, features=self.features.copy(), 
#                               beta=beta)
    
#     def get_conv_r_spikes(self, t, mask_spikes):
# #         t = t.numpy()
#         mask_spikes = mask_spikes.clone().numpy()
#         args = np.where(shift_array(mask_spikes, 1, fill_value=False))
#         t_spk = (t[args[0]],) + args[1:]
# #         self.X_r_spk = torch.from_numpy(self.r_kernel.convolve_basis_discrete(t, t_spk, shape=mask_spikes.shape))
#         self.X_r_spk = torch.zeros(mask_spikes.shape + (1 + len(self.r_kernel.coefs),))
#         self.X_r_spk[:, :, 0] = 1
#         self.X_r_spk[:, :, 1:] = torch.from_numpy(self.r_kernel.convolve_basis_discrete(t, t_spk, shape=mask_spikes.shape))
#         self.X_r_spk = self.X_r_spk.double()
        
    def get_conv_r_spikes(self, t, mask_spikes):
        mask_spikes = mask_spikes.clone().numpy()
        args = np.where(shift_array(mask_spikes, 1, fill_value=False))
        t_spk = (t[args[0]],) + args[1:]
#         self.X_r_spk = torch.from_numpy(self.r_kernel.convolve_basis_discrete(t, t_spk, shape=mask_spikes.shape))
        self.X_r_spk = torch.zeros(mask_spikes.shape + (1 + len(self.r_kernel.coefs),))
        self.X_r_spk[:, :, 0] = 1
        self.X_r_spk[:, :, 1:] = torch.from_numpy(self.r_kernel.convolve_basis_discrete(t, t_spk, shape=mask_spikes.shape))
        self.X_r_spk = self.X_r_spk.double()
    
    def transform(self, t, mask_spikes, u, r):
        
        n_theta = len(self.r_kernel.coefs) + 1
        
        for ii in range(len(self.clip)):
#             self.theta.data[ii:ii + len(self.r_kernel.coefs)] = torch.clamp(self.theta.data[ii:ii + len(self.r_kernel.coefs)], -self.clip[ii], self.clip[ii])
            self.theta.data[ii * n_theta:(ii + 1) * n_theta] = torch.clamp(self.theta.data[ii * n_theta:(ii + 1) * n_theta], -self.clip[ii], self.clip[ii])
    
        for ii in range(len(self.clip2)):
            self.theta2.data[ii] = torch.clamp(self.theta2.data[ii], -self.clip2[ii], self.clip2[ii])
        
        ii = 0
        
        a = torch.zeros(mask_spikes.shape[1])
        
        # sum r
#         a += torch.sum(self.theta2[0] * r, axis=0)
        
        # dot 
#         a += torch.sum(torch.einsum('tka,a->tk', self.X_r_spk, self.theta[ii:ii + n_theta]) * r, axis=0)
#         ii += n_theta
        
        # dot exp
#         a += torch.sum(torch.exp(torch.einsum('tka,a->tk', self.X_r_spk, self.theta[ii:ii + n_theta])) * r, axis=0)
#         a += torch.sum(mask_spikes, 0) * torch.sum(torch.exp(torch.einsum('tka,a->tk', self.X_r_spk, self.theta[ii:ii + n_theta])) * r, axis=0)
#         projection = torch.exp(torch.einsum('tka,a->tk', self.X_r_spk, self.theta[ii:ii + n_theta]))
#         projection = projection / torch.norm(projection, dim=0)
#         a += torch.sum(projection * r, axis=0)
#         ii += n_theta
        
        projection = torch.einsum('tka,a->tk', self.X_r_spk, self.theta[ii:ii + n_theta])
        projection = projection / torch.norm(projection, dim=0)
        a += torch.sum(projection * u, axis=0)
        ii += n_theta

        # res
#         a = torch.sum(torch.einsum('tka,a->tk', self.X_r_spk, self.theta) - r, axis=0)

        # squared res
#         a += torch.sum((torch.einsum('tka,a->tk', self.X_r_spk, self.theta) - r)**2, axis=0)

        # exp_res
#         a += torch.sum(torch.exp(torch.einsum('tka,a->tk', self.X_r_spk, self.theta[ii:ii + n_theta])) - r, axis=0)
#         ii += n_theta
        
        # exp_squared_res
#         mask = torch.sum(self.X_r_spk, dim=2)
#         a += torch.mean((torch.exp(torch.einsum('tka,a->tk', self.X_r_spk, self.theta[ii:ii + n_theta])) - r * mask)**2, axis=0)
#         ii += n_theta
        
        # rbf distance
#         a = torch.sum(torch.exp(-(torch.einsum('tka,a->tk', self.X_r_spk, self.theta) - r)**2), axis=0)

        # rbf distance exp convolution
#         a = torch.sum(torch.exp(-(torch.exp(torch.einsum('tka,a->tk', self.X_r_spk, self.theta)) - r)**2), axis=0)
        
        return a
    
    def forward(self, t, mask_spikes, u, r, y, lam, neg=False):
        """Compute Wasserstein distance"""
        a = self.transform(t, mask_spikes, u, r)
        if neg:
            w_distance = -lam * (torch.mean(a[y == 1]) - torch.mean(a[y == 0]))
        else:
            w_distance = lam * (torch.mean(a[y == 1]) - torch.mean(a[y == 0]))
        return w_distance
    
    def train(self, t, mask_spikes, u, r, y, lam, lr=1e-3, num_epochs=10000, stop_cond=1e-6, verbose=False):
        optim = Adam(self.parameters(), lr=lr)  # optimizer
        w_distance = []
#         for epoch in tqdm(range(num_epochs)):
        for epoch in range(num_epochs):
        
            if verbose:
                print('\r', 'epoch', epoch, 'of', num_epochs, end='')
            
#             print(self.theta)
            optim.zero_grad()  # zero gradient
            _w_distance = self.forward(t, mask_spikes, u, r, y, lam, neg=True)  # compute wasserstein distance
            _w_distance.backward(retain_graph=True)  # compute gradient
            optim.step()  # take gradient step
            
            n_theta = len(self.r_kernel.coefs) + 1
            for ii in range(len(self.clip)):
#                 self.theta.data[ii:ii + len(self.r_kernel.coefs)] = torch.clamp(self.theta.data[ii:ii + len(self.r_kernel.coefs)], -self.clip[ii], self.clip[ii])
                self.theta.data[ii * n_theta:(ii + 1) * n_theta] = torch.clamp(self.theta.data[ii * n_theta:(ii + 1) * n_theta], -self.clip[ii], self.clip[ii])
    
            for ii in range(len(self.clip2)):
                self.theta2.data[ii] = torch.clamp(self.theta2.data[ii], -self.clip2[ii], self.clip2[ii])
    
            w_distance.append(_w_distance.item())
            
#             if epoch > 0:
#                 print(w_distance[-1], w_distance[-2], np.abs(w_distance[-1] - w_distance[-2]))
                
            if epoch > 0 and np.abs(w_distance[-1] - w_distance[-2]) < stop_cond:
                epochs_no_improve += 1
            else:
                epochs_no_improve = 0
                
            if epoch > 1 and epochs_no_improve == 3:
                break
        
        ii = 1
        self.r_kernel.coefs = self.theta[ii:ii + len(self.r_kernel.coefs)].data.numpy().copy()
        return w_distance
    
    def set_requires_grad(self, requires_grad):
        self.theta.requires_grad = requires_grad
        self.theta2.requires_grad = requires_grad