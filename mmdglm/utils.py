from kernel.utils import torch_convolve
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve
import torch


def get_arg_support(dt, t_support, t0=0):
    r"""Given the support of a kernel and a sampling interval dt returns the corresponding 
    indexes/argument values"""
    arg_support0 = int((t_support[0] - t0) / dt)
    arg_supportf = int(np.ceil((t_support[1] - t0) / dt))
    return arg_support0, arg_supportf


def get_dt(t):
    r"""Receives a 1d-array with time values and returns the sampling interval"""
    arg_dt = 20 if len(t) >= 20 else len(t)
    dt = np.median(np.diff(t[:arg_dt]))
    return dt


def plot_spiketrain(t, mask_spikes, ax=None, **kwargs):
    r"""Plots a spike train"""
    color = kwargs.get('color', 'C0')
    marker = kwargs.get('marker', 'o')
    ms = kwargs.get('ms', mpl.rcParams['lines.markersize'])
    mew = kwargs.get('mew', 0)
    label = kwargs.get('label', None)

    no_ax = False
    if ax is None:
        figsize = kwargs.get('figsize', (6, 2))
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlabel('time (ms)')
        no_ax = True

    arg_spikes = np.where(mask_spikes)
    ax.plot(t[arg_spikes[0]], arg_spikes[1], marker=marker, lw=0, color=color, ms=ms, mew=mew)
        
    extra_xspace = (t[-1] - t[0]) * .01
    ax.set_xlim(t[0] - extra_xspace, t[-1] + extra_xspace)
    ax.tick_params('y', labelleft=False, left=False)
    if label is not None:
        ax.set_ylabel(label)
    
    return ax


# def raw_autocorrelation(mask_spikes, arg_last_lag=None):
#     r"""Computes the raw autocorrelation of a spiketrain"""
#     arg_last_lag = arg_last_lag if arg_last_lag is not None else mask_spikes.shape[0]
#     x = mask_spikes.numpy()
#     autocor = fftconvolve(x, x[::-1], mode='full', axes=0)[::-1] / x.shape[0]
#     arg_lag0 = autocor.shape[0] // 2
#     autocor = torch.from_numpy(autocor[arg_lag0:arg_lag0 + arg_last_lag])
#     return autocor


def raw_autocorrelation(mask_spikes, arg_last_lag=None):
    r"""Computes the raw autocorrelation of a spiketrain"""
    arg_last_lag = arg_last_lag if arg_last_lag is not None else mask_spikes.shape[0]
    autocor = torch_convolve(mask_spikes, mask_spikes.flip(dims=(0,)), mode='fft').flip(dims=(0,))
    arg_lag0 = autocor.shape[0] // 2
    autocor = autocor[arg_lag0:arg_lag0 + arg_last_lag] / mask_spikes.shape[0]
    return autocor
            

def searchsorted(t, s, side='left'):
    '''
    Uses np.searchsorted but handles numerical round error with care
    such that returned index satisfies
    t[i-1] < s <= t[i]
    np.searchsorted(side='right') doesn't properly handle the equality sign
    on the right side
    '''
    s = np.atleast_1d(s)
    arg = np.searchsorted(t, s, side=side)

    if len(t) > 1:
        dt = get_dt(t)
        s_ = (s - t[0]) / dt
        round_s = np.round(s_, 0)
        mask_round = np.isclose(s_, np.round(s_, 0)) & (round_s >= 0) & (round_s < len(t))
        if side == 'left':
            arg[mask_round] = np.array(round_s[mask_round], dtype=int)
        elif side == 'right':
            arg[mask_round] = np.array(round_s[mask_round], dtype=int) + 1
    else:
        s_ = s - t[0]
        mask = np.isclose(s - t[0], 0.)# & (round_s >= 0) & (round_s < len(t))
        arg[mask] = np.array(s_[mask], dtype=int)

    if len(arg) == 1:
        arg = arg[0]

    return arg


def shift_array(arr, shift, fill_value=False):
    """
    Shifts array on axis 0 filling the shifted values with fill_value
    Positive shift is to the right, negative to the left
    """
    result = np.empty_like(arr)
    if shift > 0:
        result[:shift, ...] = fill_value
        result[shift:, ...] = arr[:-shift, ...]
    elif shift < 0:
        result[shift:, ...] = fill_value
        result[:shift, ...] = arr[-shift:, ...]
    else:
        result = arr
    return result
