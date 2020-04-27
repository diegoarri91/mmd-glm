from functools import partial

import numpy as np
import torch
# from torch.nn import Parameter
# from torch.optim import Adam
# from tqdm import tqdm
# from tqdm.notebook import tqdm


from .utils import get_dt, shift_array


class CNNCritic(torch.nn.Module):
    
    def __init__(self, n, kernel_size=21):
        super().__init__()
        self.cnn_layers = torch.nn.Sequential(\
                                              torch.nn.Conv1d(2, 2, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, 
                                                              groups=2), 
                                              torch.nn.ReLU(), 
                                              torch.nn.AvgPool1d(n // 100))
                                              
        self.linear_layers = torch.nn.Sequential(torch.nn.Linear(100, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1))
#         self.sequential = nn.Sequential(self.conv1, torch.nn.ReLU(), self.linear1)
#         self.register_parameter("theta", Parameter(theta))
#         self.register_parameter("theta2", Parameter(theta2))
#         self.clip = clip
#         self.clip2 = clip2

    def forward(self, t, mask_spikes, u, r):
        """Compute Wasserstein distance"""
        # (n_batch, 2, len(t))
        x = torch.stack((mask_spikes.T, u.T), dim=1)
        print(x.shape)
        
        x = self.cnn_layers(x)
        
        x = x.view(-1, 100)

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
        w_distance = []
        
        for epoch in range(num_epochs):
        
            if verbose:
                print('\r', 'epoch', epoch, 'of', num_epochs, end='')
            
            optim.zero_grad()  # zero gradient
            output = self.forward(t, mask_spikes, u, r)
            _w_distance = self.w_distance(x, y, lam, neg=True)
            _w_distance.backward(retain_graph=True)  # compute gradient
            optim.step()  # take gradient step
            
#             n_theta = len(self.r_kernel.coefs) + 1
            
            self.cnn_layers[0].bias = torch.clamp(self.cnn_layers[0].bias, -self.clip, self.clip)
    
#             for ii in range(len(self.clip2)):
#                 self.theta2.data[ii] = torch.clamp(self.theta2.data[ii], -self.clip2[ii], self.clip2[ii])
    
            w_distance.append(_w_distance.item())
                
            if epoch > 0 and np.abs(w_distance[-1] - w_distance[-2]) < stop_cond:
                epochs_no_improve += 1
            else:
                epochs_no_improve = 0
                
            if epoch > 1 and epochs_no_improve == 3:
                break
        
        ii = 1
        return w_distance
    
