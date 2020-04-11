import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

from ..utils import get_dt, searchsorted
from .base import Kernel


class KernelFun(Kernel):

    def __init__(self, fun, basis_kwargs, shared_kwargs=None, support=None, coefs=None, prior=None, prior_pars=None):
        super().__init__(prior=prior, prior_pars=prior_pars)
        self.fun = fun
        self.basis_kwargs = basis_kwargs
        self.shared_kwargs = shared_kwargs if shared_kwargs is not None else {}
        self.support = np.array(support)
        self.nbasis = len(list(self.basis_kwargs.values())[0])
        self.coefs = np.array(coefs) if coefs is not None else np.ones(self.nbasis)

    def copy(self):
        kernel = KernelFun(self.fun, basis_kwargs=self.basis_kwargs.copy(),
                              shared_kwargs=self.shared_kwargs.copy(), support=self.support.copy(),
                              coefs=self.coefs.copy(), prior=self.prior, prior_pars=self.prior_pars.copy())
        return kernel

    def area(self, dt):
        return np.sum(self.interpolate(np.arange(self.support[0], self.support[1] + dt, dt))) * dt

    def interpolate(self, t):
        # kwargs = {self.key_par: self.vals_par[None, :], **self.shared_kwargs}
        kwargs = {**{key:vals[None, :] for key, vals in self.basis_kwargs.items()}, **self.shared_kwargs}
        return np.sum(self.coefs[None, :] * self.fun(t[:, None], **kwargs), 1)

    def interpolate_basis(self, t):
        # kwargs = {self.key_par: self.vals_par[None, :], **self.shared_kwargs}
        kwargs = {**{key: vals[None, :] for key, vals in self.basis_kwargs.items()}, **self.shared_kwargs}
        return self.fun(t[:, None], **kwargs)

    def convolve_basis_continuous(self, t, I):
        """# Given a 1d-array t and an nd-array I with I.shape=(len(t),...) returns X_te,
        # the convolution matrix of each rectangular function of the base with axis 0 of I for all other axis values
        # so that X_te.shape = (I.shape, nbasis)
        # Discrete convolution can be achieved by using an I with 1/dt on the correct timing values
        Assumes sorted t"""

        dt = get_dt(t)
        arg0, argf = searchsorted(t, self.support)
        X = np.zeros(I.shape + (self.nbasis, ))

        basis_shape = tuple([argf] + [1 for ii in range(I.ndim - 1)] + [self.nbasis])
        # basis = np.zeros(basis_shape)
        # kwargs = {self.key_par: self.vals_par[None, :], **self.shared_kwargs}
        kwargs = {**{key: vals[None, :] for key, vals in self.basis_kwargs.items()}, **self.shared_kwargs}
        basis = self.fun(t[:argf, None], **kwargs).reshape(basis_shape)

        X = fftconvolve(basis, I[..., None], axes=0)
        X = X[:len(t), ...] * dt

        return X

    def convolve_basis_discrete(self, t, s, shape=None):

        if type(s) is np.ndarray:
            s = (s,)

        arg_s = searchsorted(t, s[0])
        arg_s = np.atleast_1d(arg_s)
        arg0, argf = searchsorted(t, self.support)
        # print(argf)

        if shape is None:
            shape = tuple([len(t)] + [max(s[dim]) + 1 for dim in range(1, len(s))] + [self.nbasis])
        else:
            shape = shape + (self.nbasis, )

        X = np.zeros(shape)

        kwargs = {**{key: vals[None, :] for key, vals in self.basis_kwargs.items()}, **self.shared_kwargs}

        for ii, arg in enumerate(arg_s):
            indices = tuple([slice(arg, None)] + [s[dim][ii] for dim in range(1, len(s))] + [slice(0, self.nbasis)])
            X[indices] += self.fun(t[arg:, None] - t[arg], **kwargs).reshape((len(t[arg:]), self.nbasis))

        return X

    @classmethod
    def single_exponential(cls, tau, A=1, support=None):
        support = support if support is not None else [0, 10 * tau]
        return cls(fun=lambda t, tau: np.exp(-t / tau), basis_kwargs=dict(tau=np.array([tau])), support=support,
                   coefs=np.array([A]))

class KernelFun2(Kernel):

    def __init__(self, fun=None, pars=None, support=None):
        self.fun = fun
        self.pars = pars
        self.support = np.array(support)
        self.values = None

    def interpolate(self, t):
        return self.fun(t, *self.pars)

    def area(self, dt):
        return np.sum(self.interpolate(np.arange(self.support[0], self.support[1] + dt, dt))) * dt

    @classmethod
    def exponential(cls, tau, A, support=None):
        if support is None:
            support = [0, 10 * tau]
        return cls(fun=lambda t, tau, A: A * np.exp(-t / tau), pars=[tau, A], support=support)

    @classmethod
    def gaussian(cls, tau, A):
        return cls(fun=lambda t, tau, A: A * np.exp(-(t / tau)**2.), pars=[tau, A], support=[-5 * tau, 5 * tau + .1])

    @classmethod
    def gaussian_delta(cls, delta):
        return cls.gaussian(np.sqrt(2.) * delta, 1. / np.sqrt(2. * np.pi * delta ** 2.))
