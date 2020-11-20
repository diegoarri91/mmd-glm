import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

from .utils import get_dt, searchsorted
from .base import Kernel


class KernelBasisValues(Kernel):

    def __init__(self, basis_values, support, dt, coefs=None, prior=None, prior_pars=None):
        super().__init__(prior=prior, prior_pars=prior_pars)
        self.dt = dt
        self.basis_values = basis_values
        self.nbasis = basis_values.shape[1]
        self.coefs = np.array(coefs) if coefs is not None else np.ones(self.nbasis)
        self.support = np.array(support)

    def copy(self):
        kernel = KernelBasisValues(self.basis_values.copy(), self.support.copy(), self.dt, coefs=self.coefs.copy(), 
                                   prior=self.prior, prior_pars=self.prior_pars.copy())
        return kernel

    def interpolate(self, t):

#         assert np.isclose(self.dt, get_dt(t))

        t = np.atleast_1d(t)
        res = np.zeros(len(t))

        arg0 = int(self.support[0] / self.dt)
        argf = int(np.ceil(self.support[1] / self.dt))

        # TODO. Check conditions
        if arg0 >= 0 and argf <= len(t):
            res[arg0:argf] = np.matmul(self.basis_values, self.coefs)
        elif arg0 == 0 and argf > len(t):
            res = np.matmul(self.basis_values, self.coefs)[:len(t)]
        else:
            res = None
        
        return res

    def interpolate_basis(self, t):
        return self.basis_values

    def convolve_basis_discrete(self, t, s, shape=None):

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
