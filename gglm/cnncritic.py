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
    
    def __init__(self, n, kernel_size=21, cnn_bias=True, cnn_output_window=100, clip=1e2):
        super().__init__()
        self.cnn_output_window = cnn_output_window
        self.cnn_layers = torch.nn.Sequential(\
                                              torch.nn.Conv1d(2, 2, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, 
                                                              bias=cnn_bias, groups=2), 
                                              torch.nn.ReLU())
        self.pooling = torch.nn.AvgPool1d(n // cnn_output_window)
        self.linear_layers = torch.nn.Sequential(torch.nn.Linear(2 * cnn_output_window, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1))
        
        self.cnn_layers[0] = weight_norm(self.cnn_layers[0], name='weight', dim=0)
        
        self.cnn_layers = self.cnn_layers.double()
        self.linear_layers = self.linear_layers.double()
#         self.sequential = nn.Sequential(self.conv1, torch.nn.ReLU(), self.linear1)
#         self.register_parameter("theta", Parameter(theta))
#         self.register_parameter("theta2", Parameter(theta2))
        self.clip = clip
#         self.clip2 = clip2

    def forward(self, x):
        """Compute Wasserstein distance"""
        # (n_batch, 2, len(t))
        
        x = self.cnn_layers(x)
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
    
    def train(self, t, mask_spikes, u, r, y, lam, lr=1e-3, num_epochs=10000, stop_cond=1e-6, verbose=False):
        
        optim = Adam(self.parameters(), lr=lr)  # optimizer
#         print(list(self.parameters())[0])
        neg_w_distance = []
        
        x = Variable(torch.stack((mask_spikes.T, u.T), dim=1))
#         print(x)
        for epoch in range(num_epochs):
        
            if verbose:
                print('\r', 'epoch', epoch, 'of', num_epochs, end='')
            
            optim.zero_grad()  # zero gradient
            output = self(x)
            _neg_w_distance = self.w_distance(output, y, lam, neg=True)
            _neg_w_distance.backward(retain_graph=True)  # compute gradient
            optim.step()  # take gradient step
            
#             n_theta = len(self.r_kernel.coefs) + 1
        
#             self.cnn_layers[0].weight.data /= torch.norm(self.cnn_layers[0].weight.data, dim=2)[..., None]
#             for par in self.parameters():
#                 par.data = torch.clamp(par.data, -self.clip, self.clip)
    
            neg_w_distance.append(_neg_w_distance.item())
                
            if epoch > 0 and np.abs(neg_w_distance[-1] - neg_w_distance[-2]) < stop_cond:
                epochs_no_improve += 1
            else:
                epochs_no_improve = 0
                
            if epoch > 1 and epochs_no_improve == 3:
                break
        
        ii = 1
        return neg_w_distance
    
