import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

from ..utils import get_dt, searchsorted


class Kernel:

    def __init__(self, prior=None, prior_pars=None):
        self.prior = prior
        self.prior_pars = np.array(prior_pars)
        self.fix_values = False
        self.values = None

    def interpolate(self, t):
        pass

    # def get_KernelValues(self, t):
    #     kernel_values = self.interpolate(t)
    #     return KernelValues(values=kernel_values, support=self.support)

    def plot(self, t=None, ax=None, invert_t=False, invert_values=False, exp_values=False,  **kwargs):

        if t is None:
            dt = .1
            t = np.arange(self.support[0], self.support[1] + dt, dt)

        if ax is None:
            figsize = kwargs.get('figsize', (8, 5) )
            fig, ax = plt.subplots(figsize = figsize)

        y = self.interpolate(t)
        if invert_t:
            t = -t
        if invert_values:
            y = -y
        if exp_values:
            y = np.exp(y)
        ax.plot(t, y, **kwargs)

        return ax
    
    def plot_lin_log(self, t=None, axs=None, **kwargs):

        if t is None:
            dt = .1
            t = np.arange(self.support[0], self.support[1] + dt, dt)

        if axs is None:
            figsize = kwargs.get('figsize', (12, 5) );
            fig, axs = plt.subplots(figsize = figsize, ncols = 3);
            
        y = self.interpolate(t)
        axs[0].plot(t, y)
        axs[1].plot(t, y)
        axs[1].set_yscale('log')
        axs[2].plot(t, y)
        axs[2].set_xscale('log'); axs[2].set_yscale('log')
        
        return axs

    def set_values(self, dt):
        arg0 = int(self.support[0] / dt)
        argf = int(np.ceil(self.support[1] / dt))
        t_support = np.arange(arg0, argf + 1, 1) * dt
        self.values = self.interpolate(t_support)
        return self

    # def set_values(self, dt, ndim):
    #     arg0 = int(self.support[0] / dt)
    #     argf = int(np.ceil(self.support[1] / dt))
    #
    #     t_support = np.arange(arg0, argf + 1, 1) * dt
    #     t_shape = (len(t_support), ) + tuple([1] * (ndim-1))
    #     self.values = self.interpolate(t_support).reshape(t_shape)
    
    def convolve_continuous(self, t, I, mode='fft'):
        
        # Given a 1d-array t and an nd-array I with I.shape=(len(t),...) returns convolution,
        # the convolution of the kernel with axis 0 of I for all other axis values
        # so that convolution.shape = I.shape
        
        dt = get_dt(t)
        arg0 = int(self.support[0] / dt)
        argf = int(np.ceil(self.support[1] / dt))

        if isinstance(self, KernelValues):
            kernel_values = self.values
        else:
            t_support = np.arange(arg0, argf + 1, 1) * dt
            kernel_values = self.interpolate(t_support)
        
        shape = (kernel_values.shape[0], ) + tuple([1] * (I.ndim-1))
        kernel_values = kernel_values.reshape(shape)

        convolution = np.zeros(I.shape)
        
        if mode == 'fft':
        
            full_convolution = fftconvolve(kernel_values, I, mode='full', axes=0)
            # ME COSTO UN HUEVO LOGRAR LAS DOS LINEAS A CONTINUACION Y PARECE FUNCIONAR. NO TOCAR
#            print(arg0, argf)

            if arg0 < 0 and argf > 0 and arg0 + argf - 1 >= 0:
                convolution[arg0 + argf - 1:, ...] = full_convolution[argf - 1:len(t) - arg0, ...]
                # convolution[arg0 + argf - 1:, ...] = full_convolution[argf - 1:len(t) - arg0, ...]
            elif arg0 >= 0 and argf > 0:
                convolution[arg0:, ...] = full_convolution[:len(t) - arg0, ...]
            elif arg0 < 0:
                #convolution[:len(t) + argf - 1, ...] = full_convolution[-argf + 1:len(t), ...]
                convolution[:len(t) + arg0 + 1, ...] = full_convolution[-arg0:len(t) + 1, ...]
            else:
                return None
            # NO TOCAR A MENOS QUE ESTE MUY SEGURO
                
        convolution *= dt
        
        return convolution

    def correlate_continuous(self, t, I, mode='fft'):
        return self.convolve_continuous(t, I[::-1], mode=mode)[::-1]

    def deconvolve_continuous(self, t, I, v, mask=None):

        if mask is None:
            mask = np.ones(I.shape, dtype=bool)

        X = self.convolve_basis_continuous(t, I)
        X = X[mask, :]
        v = v[mask]

        self.coefs = np.linalg.lstsq(X, v, rcond=None)[0]

    def convolve_discrete(self, t, s, A=None, shape=None):
        
        # Given a 1d-array t and a tuple of 1d-arrays s=(tjs, shape) containing timings in the
        # first 1darray of the tuple returns the convolution of the kernels with the timings
        # the convolution of the kernel with the timings. conv.ndim = s.ndim and
        # conv.shape = (len(t), max of the array(accross each dimension))
        # A is used as convolution weights. A=(A) with len(A)=len(s[0]).
        # Assumes kernel is only defined on t >= 0
        
        if type(s) is not tuple:
            s = (s,)
            
        if A is None:
            A = (1. for ii in range(s[0].size)) # Instead of creating the whole list/array in memory I use a generator

        if shape is None:
            # max(s[dim]) determines the size of each dimension
            shape = tuple([max(s[dim]) + 1 for dim in range(1, len(s))])

        arg_s = searchsorted(t, s[0])
        arg_s = np.atleast_1d(arg_s)

        convolution = np.zeros((len(t), ) + shape)

        for ii, (arg, A) in enumerate(zip(arg_s, A)):

            index = tuple([slice(arg, None)] + [s[dim][ii] for dim in range(1, len(s))])
            convolution[index] += A * self.interpolate(t[arg:] - t[arg])
                
        return convolution

class KernelValues(Kernel):

    def __init__(self, values=None, support=None):
        self.values = values
        self.support = np.array(support)
