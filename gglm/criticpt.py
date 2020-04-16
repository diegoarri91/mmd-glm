from functools import partial

import numpy as np
import torch
from torch.nn import Parameter
from torch.optim import Adam
# from tqdm import tqdm
# from tqdm.notebook import tqdm


from .utils import get_dt, shift_array


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class CriticPyTorch(torch.nn.Module):
    
    def __init__(self, r_kernel, theta, clip):
        super().__init__()
        self.r_kernel = r_kernel
        self.register_parameter("theta", Parameter(theta))
        self.clip = clip
        
#     def copy(self):
#         u_kernel = None if self.u_kernel is None else self.u_kernel.copy()
#         u_spk_kernel = None if self.u_spk_kernel is None else self.u_spk_kernel.copy()
#         r_kernel = None if self.r_kernel is None else self.r_kernel.copy()
# #         r_spk_kernel = None if self.r_spk_kernel is None else self.r_spk_kernel.copy()
# #         beta = None if self.beta is None else self.beta.copy()
#         return self.__class__(u_kernel=u_kernel, u_spk_kernel=u_spk_kernel, r_kernel=r_kernel, r_spk_kernel=r_spk_kernel, features=self.features.copy(), 
#                               beta=beta)
    
    def get_conv_r_spikes(self, t, mask_spikes):
        args = np.where(shift_array(mask_spikes, 1, fill_value=False))
        t_spk = (t[args[0]],) + args[1:]
        self.X_r_spk = torch.from_numpy(self.r_kernel.convolve_basis_discrete(t, t_spk, shape=mask_spikes.shape))
    
    def transform(self, t, mask_spikes, r, y):
        r = torch.from_numpy(r)
        self.theta.data = torch.clamp(self.theta.data, -self.clip, self.clip)
        
        # rbf distance
#         a = torch.sum(torch.exp(-(torch.einsum('tka,a->tk', self.X_r_spk, self.theta) - r)**2), axis=0)

        # rbf distance exp convolution
        a = torch.sum(torch.exp(-(torch.exp(torch.einsum('tka,a->tk', self.X_r_spk, self.theta)) - r)**2), axis=0)
        
        return a
    
    def forward(self, t, mask_spikes, r, y):
        """Compute Wasserstein distance"""
#         X_r_spk = torch.from_numpy(X_r_spk)
#         r = torch.from_numpy(r)
#         self.theta.data = torch.clamp(self.theta.data, -self.clip, self.clip)
        
#         X_r_spk = torch.sum(self.X_r_spk * r[..., None], axis=0)
#         a = torch.mv(X_r_spk, self.theta)
        
#         a = torch.sum(torch.einsum('tka,a->tk', self.X_r_spk, self.theta) - r, axis=0)
        
#         a = torch.sum((torch.einsum('tka,a->tk', self.X_r_spk, self.theta) - r)**2, axis=0)

        
#         a = torch.sum(torch.exp(-(torch.einsum('tka,a->tk', self.X_r_spk, self.theta) - r)**2), axis=0)

        a = self.transform(t, mask_spikes, r, y)
        
        w_distance = -(torch.mean(a[y == 1]) - torch.mean(a[y == 0]))
        return w_distance
    
    def train(self, t, mask_spikes, r, y, lr=1e-3, num_epochs=10000):
        optim = Adam(self.parameters(), lr=lr)  # optimizer
        w_distance = []
#         for epoch in tqdm(range(num_epochs)):
        for epoch in range(num_epochs):
            print('\r', 'epoch', epoch, 'of', num_epochs, end='')
            optim.zero_grad()  # zero gradient
            _w_distance = self.forward(t, mask_spikes, r, y)  # compute wasserstein distance
            _w_distance.backward()  # compute gradient
            optim.step()  # take gradient step
            self.theta.data = torch.clamp(self.theta.data, -self.clip, self.clip)
            w_distance.append(_w_distance.item())
        self.r_kernel.coefs = self.theta.data.numpy().copy()
        return w_distance
    
