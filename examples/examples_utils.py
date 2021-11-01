import math

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import torch

from mmdglm.metrics import mmd_loss
from mmdglm.utils import get_timestep, plot_spiketrain, raw_autocorrelation



PALETTE = {'data': 'C0', 'ml-glm': 'C2', 'mmd-glm': 'C1'}


def compute_mean_mmd(model, t, mask_spikes, kernel, n_batch_fr, biased=False):
    mmd_ml = []
    for ii in range(20):
        _, mask_spikes_ml = model.sample(t, shape=(n_batch_fr,))
        _mmd_ml = mmd_loss(t, mask_spikes, mask_spikes_ml, kernel=kernel, biased=biased)
        mmd_ml.append(_mmd_ml)
    mmd_ml = torch.tensor(mmd_ml)
    mean_mmd_ml = torch.mean(mmd_ml)
    sd_mmd_ml = torch.std(mmd_ml)
    se_mmd_ml = sd_mmd_ml / math.sqrt(len(mmd_ml))
    return mean_mmd_ml, se_mmd_ml


def set_style():
    mpl.rcParams['figure.figsize'] = (4, 4)
    mpl.rcParams['axes.titlesize'] = 14
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['lines.markersize'] = 2
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['legend.fontsize'] = 14

    
def fig_layout(mmd=False):
    
    n_spike_trains = 3 if mmd else 2
    nrows = 4
    ncols = 6
    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(hspace=0.4, wspace=0.9)
    
    ax_st1 = plt.subplot2grid((nrows, ncols), (0, 0), colspan=3)
    ax_st2 = plt.subplot2grid((nrows, ncols), (1, 0), colspan=3, sharex=ax_st1)
    ax_st = [ax_st1, ax_st2]
    if mmd:
        ax_st.append(plt.subplot2grid((nrows, ncols), (2, 0), colspan=3, sharex=ax_st1))
    for ax in ax_st:
        ax.tick_params(axis='both', labelbottom=False, labelleft=False)
        ax.set_yticks([])
    
    ax_fr = plt.subplot2grid((nrows, ncols), (2 + mmd, 0), colspan=3, sharex=ax_st1)
    ax_fr.set_xlabel('time (ms)')
    ax_fr.set_ylabel('firing rate (Hz)')
    
    ax_hist = plt.subplot2grid((nrows, ncols), (0, 3), rowspan=2, colspan=2)
    ax_hist.set_ylabel('history filter (gain)')
    
    ax_autocor = plt.subplot2grid((nrows, ncols), (2, 3), rowspan=2, colspan=2)
    ax_autocor.set_xlabel('lag (ms)')
    ax_autocor.set_ylabel('autocorrelation')
    ax_autocor.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    
    ax_ll = plt.subplot2grid((nrows, ncols), (0, 5), rowspan=2)
    xticks = [0, 1] if mmd else [1]
    xlabels = ['mmd-glm', 'ml-glm'] if mmd else ['ml-glm']
    ax_ll.set_xticks(xticks)
    ax_ll.set_xticklabels(xlabels, rotation=45)
    ax_ll.set_ylabel('log-likelihood (bits/spk)')
    
    return ax_st, ax_fr, ax_hist, ax_autocor, ax_ll


def fig2_layout():
    fig = plt.figure(figsize=(15, 3))
    nrows, ncols = 2, 4

    axd = plt.subplot2grid((nrows, ncols), (0, 0), rowspan=1)
    axmmd_samples = plt.subplot2grid((nrows, ncols), (1, 0), rowspan=1, sharex=axd)
    axmmd = plt.subplot2grid((nrows, ncols), (0, 1), rowspan=nrows)
    axmmd_ins = inset_axes(axmmd, width=0.8, height=0.5, bbox_to_anchor=(.25, .1, .7, .4),
                           bbox_transform=axmmd.transAxes)
    axnll = plt.subplot2grid((nrows, ncols), (0, 2), rowspan=nrows)
    axhist = plt.subplot2grid((nrows, ncols), (0, 3), rowspan=nrows)

    fig.subplots_adjust(wspace=0.35)

    axd.set_title('samples')
    axd.set_ylabel('true')
    axd.set_yticks([])
    axd.tick_params(axis='x', labelbottom=False)

    axmmd_samples.set_xlabel('time (ms)')
    axmmd_samples.set_ylabel('MMD-GLM')
    axmmd_samples.set_yticks([])

    axmmd.set_xlabel('iteration')
    axmmd.set_ylabel('MMD$^2$')

    axnll.set_xlabel('iteration');
    axnll.set_ylabel('NLL');

    axhist.set_title('history filter')
    axhist.set_xlabel('time (ms)');
    axhist.set_ylabel('gain')
    axhist.plot([0, 280], [1, 1], 'k--', lw=1, alpha=0.5)

    return fig, (axd, axmmd_samples, axmmd, axmmd_ins, axnll, axhist)


def plot_filter(ax, filter, gain=False, **kwargs):
    support_range = torch.arange(filter.support[0], filter.support[1], 1)
    filter_values = filter.evaluate(support_range)
    if gain:
        filter_values = torch.exp(filter_values)
    ax.plot(support_range, filter_values.detach(), **kwargs)


def plot_filter_bias(ax, glm, gain=False, bias_offset=0, **kwargs):
    plot_filter(ax, glm.hist_kernel, gain=gain, **kwargs)
    ax.text(0.65, 0.25 - bias_offset * 0.1, 'b={:.2f}'.format(glm.bias.item()), color=kwargs.get('color', 'C0'),
            transform=ax.transAxes, fontsize=14)



def plot_fit(axs, label, mask_spikes, dt=1, psth=None, history_filter=None, autocor=None, ll=None):
    color = PALETTE[label]
    if label == 'data':
        t = torch.arange(0, len(mask_spikes), dt)
        plot_spiketrain(t, mask_spikes, axs[0][0], color=color, label=label)
    else:
        t = torch.arange(0, len(mask_spikes), dt)
        ii = 1 if label == 'mmd-glm' else -1
        plot_spiketrain(t, mask_spikes, axs[0][ii], color=color, label=label)
    
    if psth is not None:
        t_psth = torch.arange(0, len(psth), dt)
        axs[1].plot(t_psth, psth, color=color)
        
    if history_filter is not None:
        plot_filter(ax=axs[2], filter=history_filter, gain=True, color=color, label=label)
        
    if autocor is not None:
        t_autocor = torch.arange(0, len(autocor), dt)
        axs[3].plot(t_autocor[1:], autocor[1:], color=color)
        
    if ll is not None:
        ii = 0 if label == 'mmd-glm' else 1
        axs[4].bar(ii, ll, color=color)


def plot_mmd(ax, iterations, mmd, mean_mmd_ml, se_mmd_ml, first_iteration=0):
    num_epochs = iterations[-1]
    ax.plot(iterations[first_iteration:], mmd[first_iteration:], label='MMD-GLM', color=PALETTE['mmd-glm'])
    ax.plot(num_epochs, 0, '-', label='ML-GLM', color=PALETTE['ml-glm'])
    ax.plot([first_iteration + 1, num_epochs], [mean_mmd_ml - se_mmd_ml, mean_mmd_ml - se_mmd_ml], '--', color=PALETTE['ml-glm'])
    ax.plot([first_iteration + 1, num_epochs], [mean_mmd_ml + se_mmd_ml, mean_mmd_ml + se_mmd_ml], '--', color=PALETTE['ml-glm'])
    ax.fill_between([first_iteration + 1, num_epochs], mean_mmd_ml - se_mmd_ml, mean_mmd_ml + se_mmd_ml, alpha=0.5, color=PALETTE['ml-glm'])


def psth_and_autocor(t, mask_spikes, kernel_smooth=None, smooth_autocor=False, arg_last_lag=None):
    """
    Compute PSTH and autocorrelation of samples. PSTH is obtained by smoothing and averaging over trials.
    """
    dt = get_timestep(t)
    # psth = torch.mean(kernel_smooth.convolve_continuous(t, mask_spikes), 1) * 1000 # psth obtained by
    psth = torch.mean(kernel_smooth(mask_spikes, dt=dt), 1) * 1000  # psth obtained by
    autocor = raw_autocorrelation(mask_spikes, arg_last_lag=arg_last_lag)
    autocor = torch.mean(autocor, 1)
    if smooth_autocor:
        # t_autocor = torch.arange(0, len(autocor), dt)
        # autocor[1:] = kernel_smooth.convolve_continuous(t_autocor[1:], autocor[1:])
        autocor[1:] = kernel_smooth(autocor[1:], dt=dt)
    return psth, autocor
