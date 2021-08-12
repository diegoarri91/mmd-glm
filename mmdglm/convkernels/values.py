import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

from .base import Kernel
from ..utils import get_dt, get_arg_support, searchsorted


class KernelBasisValues(Kernel):

    def __init__(self, dt, basis_values, support=None, coefs=None):
        self.dt = dt
        self.basis_values = basis_values
        self.nbasis = basis_values.shape[1]
        self.support = np.array(support) if support is not None else np.array([0, basis_values.shape[0] * dt])
        self.coefs = np.array(coefs) if coefs is not None else np.ones(self.nbasis)
        assert self.basis_values.shape[0] == int((self.support[1] - self.support[0]) / self.dt)

    def interpolate(self, t):
        """Interpolates the values of the kernel to the given times"""
        assert len(t)==1 or np.isclose(self.dt, get_dt(t))

        t = np.atleast_1d(t)
        res = np.zeros(len(t))

        arg0, argf = get_arg_support(self.dt, self.support, t0=t[0])

        if arg0 >= 0:
            argf = min(argf, len(t))
            res[arg0:argf] = np.matmul(self.basis_values, self.coefs)[:argf - arg0]
        elif arg0 < 0 and argf > 0:
            n_times = self.basis_values.shape[0]
            res[:min(len(t), n_times + arg0)] = np.matmul(self.basis_values, self.coefs)[-arg0:min(len(t) - arg0, n_times)]
        
        return res

    def interpolate_basis(self, t):
        """Interpolates the values of the kernel basis to the given times"""
        assert len(t)==1 or np.isclose(self.dt, get_dt(t))

        t = np.atleast_1d(t)

        arg0, argf = get_arg_support(self.dt, self.support, t0=t[0])

        if arg0 >= 0:
            argf = min(argf, len(t))
            return self.basis_values[:argf - arg0]
        elif arg0 < 0 and argf > 0:
            n_times = self.basis_values.shape[0]
            return self.basis_values[-arg0:min(len(t) - arg0, n_times)]

    def convolve_basis_discrete(self, t, s, shape=None):
        """Convolves the kernel basis to the given spike times"""
        if type(s) is np.ndarray:
            s = (s,)

        arg_s = searchsorted(t, s[0])
        arg_s = np.atleast_1d(arg_s)
        arg0, argf = searchsorted(t, self.support)

        if shape is None:
            shape = tuple([len(t)] + [max(s[dim]) + 1 for dim in range(1, len(s))] + [self.nbasis])
        else:
            shape = shape + (self.nbasis, )

        X = np.zeros(shape)
        
        for ii, arg in enumerate(arg_s):
            indices = tuple([slice(arg, min(arg + argf, len(t)))] + [s[dim][ii] for dim in range(1, len(s))] + [slice(0, self.nbasis)])
            X[indices] += self.basis_values[:min(arg + argf, len(t)) - arg, :]

        return X

    @classmethod
    def orthogonalized_raised_cosines(cls, dt, last_time_peak, n, b, a=1e0, coefs=None):
        """Returns a kernel using a base of orthogonalized raised cosines"""
        range_locs = np.log(np.array([0, last_time_peak]) + b)
        delta = (range_locs[1] - range_locs[0]) / (n - 1)  # delta = 1 / (n - 1) * np.log(1 + last_peak / b)
        locs = np.linspace(range_locs[0], range_locs[1], n)

        last_time = np.exp(range_locs[1] + 2 * delta / a) - b
        t = np.arange(0, last_time, dt)
        support = [t[0], t[-1] + dt]

        raised_cosines = (1 + np.cos(np.maximum(-np.pi, np.minimum(
            a * (np.log(t[:, None] + b) - locs[None, :]) * np.pi / delta / 2, np.pi)))) / 2
        raised_cosines = raised_cosines / np.sqrt(np.sum(raised_cosines ** 2, 0))
        u, s, v = np.linalg.svd(raised_cosines)
        basis = u[:, :n]

        return cls(basis_values=basis, support=support, dt=dt, coefs=coefs)

    def copy(self):
        kernel = KernelBasisValues(self.dt, self.basis_values.copy(), self.support.copy(), coefs=self.coefs.copy())
        return kernel

    @classmethod
    def gaussian(cls, dt, tau):
        """Returns a gaussian kernel centered at 0"""
        support = np.array([-5 * tau, 5 * tau])
        t = np.arange(support[0], support[1], dt)
        A = 1 / np.sqrt(2 * np.pi * tau ** 2)
        basis_values = np.exp( -(t / (2 * tau))**2 )
        return cls(dt, basis_values.reshape(-1, 1), support=support, coefs=[A])
