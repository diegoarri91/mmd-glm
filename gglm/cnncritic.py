from functools import partial

import numpy as np
import torch
# from torch.nn import Parameter
from torch.autograd import Variable
from torch.optim import Adam
from torch.nn.utils import weight_norm
# from tqdm.notebook import tqdm


from .utils import get_dt, shift_array


class CNNCritic(torch.nn.Module):
    
    def __init__(self, n, n_in, two_layers=False, avg_pooling=True, kernel_size=21, cnn_bias=True, weight_normalization=False, cnn_output_window=100, clip=1e2):
        super().__init__()
        self.cnn_output_window = cnn_output_window
        if two_layers:
            input_linear_factor = 1
            self.cnn_layers = torch.nn.Sequential(\
                                                  torch.nn.Conv1d(n_in, n_in, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, 
                                                                  bias=cnn_bias, groups=n_in), 
                                                  torch.nn.Conv1d(n_in, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, 
                                                                  bias=cnn_bias, groups=1))
        else:
            input_linear_factor = n_in
            self.cnn_layers = torch.nn.Sequential(\
                                                  torch.nn.Conv1d(n_in, n_in, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, 
                                                                  bias=cnn_bias, groups=n_in))
        if avg_pooling:
            self.pooling = torch.nn.AvgPool1d(n // cnn_output_window)
        else:
            kernel_size = n // cnn_output_window
            self.pooling = torch.nn.Conv1d(1, 1, kernel_size=kernel_size, stride=kernel_size, padding=0)
        self.linear_layers = torch.nn.Sequential(torch.nn.Linear(input_linear_factor * cnn_output_window, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1))
        self = self.double()
        
        if weight_normalization:
#             if not(cnn_bias):
#                 with torch.no_grad():
# #                         self.cnn_layers[0].weight[0, 0, :].fill_(1 / kernel_size**0.5)
# #                     self.cnn_layers[0].weight[1, 0, :].fill_(-1 / kernel_size**0.5)
#                     self.cnn_layers[0].weight[0, 0, :].uniform_(0, 1 / kernel_size**0.5)
#                     self.cnn_layers[0].weight[1, 0, :].uniform_(-1 / kernel_size**0.5, 0)
            self.cnn_layers[0] = weight_norm(self.cnn_layers[0], name='weight', dim=0)
            self.cnn_layers[0].weight_g.data = torch.ones(self.cnn_layers[0].weight_g.data.shape)
            self.cnn_layers[0].weight_g.requires_grad = False

        self.cnn_layers = self.cnn_layers.double()
        self.linear_layers = self.linear_layers.double()
        self.clip = clip
        self.weight_normalization = weight_normalization

    def forward(self, x):
        """Compute Wasserstein distance"""
        # (n_batch, 2, len(t))
        
        x = self.cnn_layers(x)
#         x[:, 1, :].data = x[:, 1, :].data / torch.norm(x[:, 1, :].data, dim=1)[:, None]
        x = self.pooling(x)
        x = x.view(x.shape[0], -1)
        x = self.linear_layers(x)
        
        return x
    
    def w_distance(self, x, y, lam, neg=False):
        if neg:
            w_distance = -lam * (torch.mean(x[y == 1]) - torch.mean(x[y == 0]))
        else:
            w_distance = lam * (torch.mean(x[y == 1]) - torch.mean(x[y == 0]))
        return w_distance
    
    def train(self, x, y, lam, lr=1e-3, num_epochs=10000, stop_cond=1e-6, verbose=False):
        
        optim = Adam(self.parameters(), lr=lr)  # optimizer
        neg_w_distance = []
        
#         x = torch.stack((mask_spikes.T, u.T), dim=1)
        
        for epoch in range(num_epochs):
        
            if verbose:
                print('\r', 'epoch', epoch, 'of', num_epochs, end='')
            
            optim.zero_grad()  # zero gradient
            output = self(x)
            _neg_w_distance = self.w_distance(output, y, lam, neg=True)
            _neg_w_distance.backward(retain_graph=True)  # compute gradient
            optim.step()  # take gradient step
            
#             n_theta = len(self.r_kernel.coefs) + 1
        
            if self.weight_normalization:
                self.cnn_layers[0].weight_v.data /= torch.norm(self.cnn_layers[0].weight_v.data, dim=2)[..., None]
#                 print(self.cnn_layers[0].weight_v.data)
#                 print(self.cnn_layers[0].weight_g.data)
                
            for par in self.parameters():
                par.data = torch.clamp(par.data, -self.clip, self.clip)
    
            neg_w_distance.append(_neg_w_distance.item())
                
            if epoch > 0 and np.abs(neg_w_distance[-1] - neg_w_distance[-2]) < stop_cond:
                epochs_no_improve += 1
            else:
                epochs_no_improve = 0
                
            if epoch > 1 and epochs_no_improve == 3:
                break
        
        ii = 1
        return neg_w_distance
    

    def set_requires_grad(self, requires_grad):
        if requires_grad:
            if self.weight_normalization:
                self.cnn_layers[0].weight_v.requires_grad = True
                self.cnn_layers[0].weight_g.requires_grad = False
            for par in self.linear_layers.parameters():
                par.requires_grad = True
        else:
            for par in self.parameters():
                par.requires_grad = False
                