import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
import torch

from ..utils import get_arg_support, get_dt, searchsorted


class Kernel:

    def __init__(self):
        pass

    def interpolate(self, t):
        pass
    
    def interpolate_basis(self, t):
        pass

    def convolve_continuous(self, t, x):
        """Implements the convolution of a time series with the kernel using scipy fftconvolve.

        Args:
            t (array): time points
            x (array): time series to be convolved
            mode (str): 

        Returns:
            array: convolved time series
        """
        
        dt = get_dt(t)
        arg_support0, arg_supportf = get_arg_support(dt, self.support)

        t_support = np.arange(arg_support0, arg_supportf, 1) * dt
        kernel_values = self.interpolate(t_support)
        
        shape = (kernel_values.shape[0], ) + tuple([1] * (x.ndim - 1))
        kernel_values = kernel_values.reshape(shape)

        convolution = np.zeros(x.shape)
        
        full_convolution = fftconvolve(kernel_values, x, mode='full', axes=0)

        if arg_support0 >= 0:
            convolution[arg_support0:, ...] = full_convolution[:len(t) - arg_support0, ...]
        elif arg_support0 < 0 and arg_supportf >= 0: # or to arg_support0 < 0 and len(t) - arg_support0 <= len(t) + arg_supportf - arg_support0:
            convolution = full_convolution[-arg_support0:len(t) - arg_support0, ...]
        else: # or arg0 < 0 and len(t) - arg0 > len(t) + arg_supportf - arg0:
            convolution[:len(t) + arg_supportf, ...] = full_convolution[-arg_supportf:, ...]
                
        convolution *= dt
        
        return torch.from_numpy(convolution)

    def convolve_discrete(self, t, s, A=None, shape=None, renewal=False):
        """Implements the convolution of discrete events in time with the kernel

        Args:
            t (array): time points
            s (array): time events
            mode (str): 

        Returns:
            array: convolved time series
        """
        
        if type(s) is not tuple:
            s = (s,)
            
        if A is None:
            A = (1. for ii in range(s[0].size))

        if shape is None:
            shape = tuple([max(s[dim]) + 1 for dim in range(1, len(s))])

        arg_s = searchsorted(t, s[0])
        arg_s = np.atleast_1d(arg_s)

        convolution = np.zeros((len(t), ) + shape)

        for ii, (arg, A) in enumerate(zip(arg_s, A)):

            index = tuple([slice(arg, None)] + [s[dim][ii] for dim in range(1, len(s))])
            if not(renewal):
                convolution[index] += A * self.interpolate(t[arg:] - t[arg])
            else:
                convolution[index] = A * self.interpolate(t[arg:] - t[arg])
                
        return torch.from_numpy(convolution)
   
    def fit(self, t, input, output, mask=None):
        """Fits the kernel to data using least squares"""

        if mask is None:
            mask = np.ones(input.shape, dtype=bool)

        X = self.convolve_basis_continuous(t, input)
        X = X[mask, :]
        output = output[mask]

        self.coefs = np.linalg.lstsq(X, output, rcond=None)[0]

    def correlate_continuous(self, t, x):
        """Implements the correlation of a time series with the kernel"""
        return self.convolve_continuous(t, x[::-1])[::-1]
        
    def plot(self, t=None, ax=None, offset=0, invert_t=False, invert_values=False, gain=False, **kwargs):
        """Plots the kernel"""

        if t is None:
            t = np.arange(self.support[0], self.support[1] + self.dt, self.dt)

        if ax is None:
            fig, ax = plt.subplots()

        y = self.interpolate(t) + offset
        if invert_t:
            t = -t
        if invert_values:
            y = -y
        if gain:
            y = np.exp(y)
        ax.plot(t, y, **kwargs)

        return ax
