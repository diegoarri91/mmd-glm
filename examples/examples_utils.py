import matplotlib as mpl
import matplotlib.pyplot as plt
from mmdglm.utils import get_dt, plot_spiketrain, raw_autocorrelation
import torch


palette = {'data': 'C0', 'ml-glm': 'C2', 'mmd-glm': 'C1'}


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


def plot_fit(axs, label, mask_spikes, dt=1, psth=None, history_filter=None, autocor=None, ll=None):
    color = palette[label]
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
        history_filter.plot(ax=axs[2], gain=True, color=color, label=label)
        
    if autocor is not None:
        t_autocor = torch.arange(0, len(autocor), dt)
        axs[3].plot(t_autocor[1:], autocor[1:], color=color)
        
    if ll is not None:
        ii = 0 if label == 'mmd-glm' else 1
        axs[4].bar(ii, ll, color=color)


def psth_and_autocor(t, mask_spikes, kernel_smooth=None, smooth_autocor=False, arg_last_lag=None):
    """
    Compute PSTH and autocorrelation of samples. PSTH is obtained by smoothing and averaging over trials.
    """
    dt = get_dt(t)
    # psth = torch.mean(kernel_smooth.convolve_continuous(t, mask_spikes), 1) * 1000 # psth obtained by
    psth = torch.mean(kernel_smooth(mask_spikes, dt=dt), 1) * 1000  # psth obtained by
    autocor = raw_autocorrelation(mask_spikes, arg_last_lag=arg_last_lag)
    autocor = torch.mean(autocor, 1)
    if smooth_autocor:
        # t_autocor = torch.arange(0, len(autocor), dt)
        # autocor[1:] = kernel_smooth.convolve_continuous(t_autocor[1:], autocor[1:])
        autocor[1:] = kernel_smooth(autocor[1:], dt=dt)
    return psth, autocor
