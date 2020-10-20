import h5py
import pickle
import sys
sys.path.append('/home/diego/python/generative-glm/experiments/')

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

from gglm.glm.mmdglm import MMDGLM
from gglm.glm.torchglm import TorchGLM
from gglm.metrics import bernoulli_log_likelihood_poisson_process, time_rescale_transform
from kernel.values import KernelBasisValues
import myplt
from signals import raw_autocorrelation
from sptr.sptr import SpikeTrain

from nips import *

def after_metrics(model, t, mask_spikes, mask_spikes_fr, **kwargs):
    b_grad = model.b.grad.detach().item()
    eta_coefs_grad = model.eta_coefs.grad.detach().clone().numpy()
    T_train = t[-1] - t[0] + t[1]
    fr = np.sum(mask_spikes_fr.double().numpy(), 0) / T_train * 1000
    mu_fr = np.mean(fr) 
    fr_max = np.max(fr)
    phi_d = kwargs.get('phi_d', None)
    phi_fr = kwargs.get('phi_fr', None)
    log_proba = kwargs.get('log_proba', None)
    return dict(b_grad=b_grad, eta_coefs_grad=eta_coefs_grad, mu_fr=mu_fr, fr_max=fr_max, 
                phi_d=phi_d, phi_fr=phi_fr, log_proba=log_proba)


def fun_metrics_mmd(model, t, mask_spikes, mask_spikes_fr, **kwargs):
    T_train = t[-1] - t[0] + t[1]
    fr = np.sum(mask_spikes_fr.double().numpy(), 0) / T_train * 1000
    mu_fr = np.mean(fr)
    max_fr = np.max(fr)
    min_fr = np.min(fr)
    b_grad = model.b.grad.detach().item()
    eta0_grad = model.eta_coefs.grad.detach().numpy().copy()
    phi_d = kwargs.get('phi_d', None)
    phi_fr = kwargs.get('phi_fr', None)
    log_proba = kwargs.get('log_proba', None)
#     phi_d, phi_fr, log_proba = phi_d.detach().numpy(), phi_fr.detach().numpy(), log_proba.detach().numpy()
    return dict(mu_fr=mu_fr, max_fr=max_fr, min_fr=min_fr, mu2_fr=np.mean(fr**2), b_grad=b_grad, eta0_grad=eta0_grad, 
                phi_d=phi_d, phi_fr=phi_fr, log_proba=log_proba)


def cum_isi(isi, bins=None):
    cum_isi, _ = np.histogram(isi, bins=bins)
    cum_isi = np.cumsum(cum_isi)
    return cum_isi / cum_isi[-1]


def load_data(path="./huk_p110509b_dots.h5", idx_train=None, idx_val=None, bins_isi=None):
    f = h5py.File(path, "r")

    mask_spikes = np.array(np.stack((f['spk']), axis=1), dtype=bool)
    dt = 1
    t = np.arange(0, mask_spikes.shape[0], 1)

    st = SpikeTrain(t, mask_spikes)
    mask_spikes_train = mask_spikes[:, idx_train]

    n_spk_train = np.sum(mask_spikes_train)
    fr_train = np.mean(np.sum(mask_spikes_train, 0) / (t[-1] - t[0] + t[1]) * 1000)
    fr2_train = np.mean((np.sum(mask_spikes_train, 0) / (t[-1] - t[0] + t[1]) * 1000)**2)
    nll_pois_proc_train = -bernoulli_log_likelihood_poisson_process(mask_spikes_train)
    autocov_train = np.mean(raw_autocorrelation(mask_spikes_train, biased=True), 1)
    
    st_train = SpikeTrain(t, mask_spikes_train)
    isi_train = st_train.isi_distribution()
    mean_isi_train = np.mean(isi_train)
    cum_trian = cum_isi(isi_train, bins=bins_isi)

    mask_spikes_val = mask_spikes[:, idx_val]
    n_spk_val = np.sum(mask_spikes_val)
    fr_val = np.mean(np.sum(mask_spikes_val, 0) / (t[-1] - t[0] + t[1]) * 1000)
    nll_pois_proc_val = -bernoulli_log_likelihood_poisson_process(mask_spikes_val)
    autocov_val = np.mean(raw_autocorrelation(mask_spikes_val, biased=True), 1)

    st_val = SpikeTrain(t, mask_spikes_val)
    isi_val = st_val.isi_distribution()
    mean_isi_val = np.mean(isi_val)
    cum_val = cum_isi(isi_val, bins=bins_isi)
    
#     bins_isi = np.arange(0, 400, 10)
#     fig, ax = plt.subplots()
#     ax.hist(isi_train, density=True, alpha=0.5, bins=bins_isi);
#     ax.hist(isi_val, density=True, alpha=0.5, bins=bins_isi);
    
    return dt, t, st_train, fr_train, isi_train, autocov_train, nll_pois_proc_train, n_spk_train, cum_trian, st_val, fr_val, isi_val, autocov_val, nll_pois_proc_val, n_spk_val, cum_val


def load_ml(path, dt, st_train, st_val):
    
    with open(path, "rb") as fit_file:
        dic_ml = pickle.load(fit_file)
        
    eta_ml = KernelBasisValues(dic_ml['basis'], [0, dic_ml['basis'].shape[0]], 1, coefs=dic_ml['eta_coefs'])
    glm_ml = TorchGLM(u0=dic_ml['u0_ml'], eta=eta_ml)
    nll_normed_train_ml = dic_ml['nll_normed_train_ml']
    nll_normed_val_ml = dic_ml['nll_normed_val_ml']
    bins_ks = dic_ml['bins_ks']

    r_train_dc_ml, r_val_dc_ml, r_fr_ml, mask_spikes_fr_ml = dic_ml['r_train_dc_ml'], dic_ml['r_val_dc_ml'], dic_ml['r_fr_ml'], dic_ml['mask_spikes_fr_ml']

    st_fr_ml = SpikeTrain(st_val.t, mask_spikes_fr_ml)
    isi_fr_ml = st_fr_ml.isi_distribution()
    mean_isi_fr_ml = np.mean(isi_fr_ml)
    mean_r_fr_ml = np.mean(r_fr_ml, 1)
    sum_r_fr_ml = np.sum(r_fr_ml, 1)
    autocov_ml = np.mean(raw_autocorrelation(mask_spikes_fr_ml, biased=True), 1)

    z_ml_train, ks_ml_train = time_rescale_transform(dt, st_train.mask, r_train_dc_ml)
    values, bins_ks = np.histogram(np.concatenate(z_ml_train), bins=bins_ks)
    z_cum_ml_train = np.append(0., np.cumsum(values) / np.sum(values))

    z_ml_val, ks_ml_val = time_rescale_transform(dt, st_val.mask, r_val_dc_ml)
    values, _ = np.histogram(np.concatenate(z_ml_val), bins=bins_ks)
    z_cum_ml_val = np.append(0., np.cumsum(values) / np.sum(values))
    
    return glm_ml, nll_normed_train_ml, nll_normed_val_ml, bins_ks, r_train_dc_ml, r_val_dc_ml, r_fr_ml, mask_spikes_fr_ml, st_fr_ml, isi_fr_ml, autocov_ml, z_ml_val, ks_ml_val, z_cum_ml_val


def load_l2(path, dt, st_train, st_val):
    
    with open(path, "rb") as fit_file:
        dic_l2 = pickle.load(fit_file)
        
    eta_l2 = KernelBasisValues(dic_l2['basis'], [0, dic_l2['basis'].shape[0]], 1, coefs=dic_l2['eta_coefs'])
    glm_l2 = TorchGLM(u0=dic_l2['u0_l2'], eta=eta_l2)
    nll_normed_train_l2 = dic_l2['nll_normed_train_l2']
    nll_normed_val_l2 = dic_l2['nll_normed_val_l2']
#     bins_ks = dic_l2['bins_ks']

    r_train_dc_l2, r_val_dc_l2, r_fr_l2, mask_spikes_fr_l2 = dic_l2['r_train_dc_l2'], dic_l2['r_val_dc_l2'], dic_l2['r_fr_l2'], dic_l2['mask_spikes_fr_l2']

    st_fr_l2 = SpikeTrain(st_val.t, mask_spikes_fr_l2)
    isi_fr_l2 = st_fr_l2.isi_distribution()
    mean_isi_fr_l2 = np.mean(isi_fr_l2)
    mean_r_fr_l2 = np.mean(r_fr_l2, 1)
    sum_r_fr_l2 = np.sum(r_fr_l2, 1)
    autocov_l2 = np.mean(raw_autocorrelation(mask_spikes_fr_l2, biased=True), 1)

#     z_l2_train, ks_l2_train = time_rescale_transform(dt, st_train.mask, r_train_dc_l2)
#     values, bins_ks = np.histogram(np.concatenate(z_l2_train), bins=bins_ks)
#     z_cum_l2_train = np.append(0., np.cumsum(values) / np.sum(values))

#     z_l2_val, ks_l2_val = time_rescale_transform(dt, st_val.mask, r_val_dc_l2)
#     values, _ = np.histogram(np.concatenate(z_l2_val), bins=bins_ks)
#     z_cum_l2_val = np.append(0., np.cumsum(values) / np.sum(values))
    
    return glm_l2, nll_normed_train_l2, nll_normed_val_l2, r_train_dc_l2, r_val_dc_l2, r_fr_l2, mask_spikes_fr_l2, st_fr_l2, isi_fr_l2, autocov_l2


def load_file(path, t, n_samples=8000):
    
    with open(path, "rb") as file:
        dic = pickle.load(file)
        
    loss_mmd = dic['loss_mmd']
    metrics_mmd = dic['metrics_mmd']
    mmdi = metrics_mmd['mmd']
    nll_normed_train_mmd = dic['nll_train']
    nll_normed_val_mmd = dic['nll_normed_val_mmd']
    n, last_peak = 5, 100
    eta = KernelBasisValues.orthogonalized_raised_cosines(1, last_peak, n, 5e1, a=1, coefs=dic['eta_coefs_mmd'])

    glm = MMDGLM(u0=dic['u0_mmd'], eta=eta)
    _, _, mask_spikes_fr_mmd = glm.sample(t, shape=(n_samples, ))
    autocov_mmd = np.mean(raw_autocorrelation(mask_spikes_fr_mmd, biased=True), 1)
    ker_name = dic['ker_name']
    
    return dic, ker_name, loss_mmd, mmdi, nll_normed_train_mmd, glm, autocov_mmd


def plot_layout_fig4(figsize):
    
    def broken_yaxis(ax1, ax2, d=0.15):
        ax1.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
    #     ax1.xaxis.tick_top()
        ax1.tick_params(axis='x', labeltop=False, labelbottom=False, length=0, width=0)
    #     ax1.set_xticks([])
    #     ax2.xaxis.tick_bottom()
        d = .025  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        kwargs = dict(transform=ax1.transAxes, lw=1, color='k', clip_on=False)
#         ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
        ax1.plot((-d, +d), (0, 0), **kwargs)        # top-left diagonal

        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
#         ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        ax2.plot((-d, +d), (1, 1), **kwargs)  # bottom-left diagonal
        return ax1, ax2
    
    r1, c1 = 6, 2
    r2, c2 = 3, 1
    r2a, r2b = 1, 2
    nrows, ncols = r1, c1 + 5 * c2
    
    fig = plt.figure(figsize=figsize)
    
    axeta = plt.subplot2grid((nrows, ncols), (0, 0), rowspan=nrows, colspan=c1)
    myplt.set_labels(axeta, xlabel='time (ms)', ylabel='gain', title='history filter')
    
    axll_val = plt.subplot2grid((nrows, ncols), (r2, c1), rowspan=r2, colspan=c2)#, sharey=axll_train)
    myplt.set_labels(axll_val, ylabel='LL val')
    
    axll_train = plt.subplot2grid((nrows, ncols), (0, c1), rowspan=r2, colspan=c2)
    myplt.set_labels(axll_train, ylabel='LL train')
    
#     axfra = plt.subplot2grid((nrows, ncols), (0, c1 + c2), rowspan=r2a, colspan=c2)
#     axfrb = plt.subplot2grid((nrows, ncols), (r2a, c1 + c2), rowspan=r2b, colspan=c2, sharex=axfra)
#     axfrb.set_ylabel('firing rate (Hz)')
#     broken_yaxis(axfra, axfrb)
    
#     axlabels = plt.subplot2grid((nrows, ncols), (0, c1 + 3 * c2), rowspan=r2, colspan=c2)
    axlabels = None

#     axcia = plt.subplot2grid((nrows, ncols), (r2, c1 + 3 * c2), rowspan=r2a, colspan=c2)
#     axci = (axcia, plt.subplot2grid((nrows, ncols), (r2 + r2a, c1 + 3 * c2), rowspan=r2b, colspan=c2, sharex=axcia))
#     broken_yaxis(axci[0], axci[1])
    axci = None
    
#     axcv = plt.subplot2grid((nrows, ncols), (r2, c1 + 2 * c2), rowspan=r2, colspan=c2)
    axcv1 = plt.subplot2grid((nrows, ncols), (r2, c1 + c2), rowspan=r2a, colspan=c2)
    axcv = (axcv1, plt.subplot2grid((nrows, ncols), (r2 + r2a, c1 + c2), rowspan=r2b, colspan=c2, sharex=axcv1))
    broken_yaxis(axcv[0], axcv[1])
    myplt.set_labels(axcv[1], ylabel='cv isi')
    
#     axmuisi = plt.subplot2grid((nrows, ncols), (r2, c1 + c2), rowspan=r2, colspan=c2)
    axmuisi = plt.subplot2grid((nrows, ncols), (0, c1 + c2), rowspan=r2b, colspan=c2)
    axmuisi = (axmuisi, plt.subplot2grid((nrows, ncols), (r2b, c1 + c2), rowspan=r2a, colspan=c2, sharex=axmuisi))
    axmuisi[0].set_ylabel('mean isi (ms)')
    broken_yaxis(axmuisi[0], axmuisi[1])
    
    axac = plt.subplot2grid((nrows, ncols), (r2, c1 + 2 * c2), rowspan=r2, colspan=c2)
#     axac1 = plt.subplot2grid((nrows, ncols), (r2, c1 + 2 * c2), rowspan=r2a, colspan=c2)
#     axac = (axac1, plt.subplot2grid((nrows, ncols), (r2 + r2a, c1 + 2 * c2), rowspan=r2b, colspan=c2, sharex=axac1))
#     broken_yaxis(axac[0], axac[1])
    myplt.set_labels(axac, ylabel='rmse autocor')
    
#     axbias = inset_axes(axeta, width=1.1, height=0.5, bbox_to_anchor=(0, 0, .9, .9),
#                    bbox_transform=axeta.transAxes)
    axbias = inset_axes(axeta, width=1.7, height=0.5, bbox_to_anchor=(0, 0, .9, .9),
                   bbox_transform=axeta.transAxes)
#     axbias.set_ylabel('exp(b) (Hz)', fontsize=tick_labelsize)
    axbias.set_ylabel('bias', fontsize=tick_labelsize)

#     ax11 = plt.subplot2grid((nrows, ncols), (0, c1 + 4 * c2), rowspan=r2, colspan=c2)
    ax11 = None
#     ax21 = plt.subplot2grid((nrows, ncols), (r2, c1 + 4 * c2), rowspan=r2, colspan=c2)
    
    axpdiv1 = plt.subplot2grid((nrows, ncols), (0, c1 + 2 * c2), rowspan=r2a, colspan=c2)
    axpdiv = (axpdiv1, plt.subplot2grid((nrows, ncols), (r2a, c1 + 2 * c2), rowspan=r2b, colspan=c2, sharex=axpdiv1))
    broken_yaxis(axpdiv[0], axpdiv[1])
    myplt.set_labels(axpdiv[1], ylabel='p(diverge)')
    
    ax3 = None
    ax4 = None
    
    return fig, (axeta, axbias, axll_train, axmuisi, axlabels, axci, axll_val, axmuisi, axcv, axac, ax11, axpdiv, ax3, ax4)


def plot_fit(loss_mmd=None, nll_normed_train_mmd=None, mmdi=None, glm=None, autocov_mmd=None, argf_autocorr=None, 
             axloss=None, axnlli=None, axmmdi=None, axextra=None, axd=None, axfr=None, axeta=None, axisi=None, axpsth=None, axnll=None, axmmd=None, axac=None, label=None, color=None):
    
    color = 'C1' if color is None else color
    
    if axloss is None:
        fig, (axloss, axnlli, axmmdi, axextra, axd, axfr, axeta, axisi, axpsth, axnll, axmmd, axac) = plot_layout_fit()

    if loss_mmd is not None:
        axloss.plot(loss_mmd)
        axloss.set_ylim(np.percentile(loss_mmd, [2.5, 97.5]) * np.array([0.95, 1.05]))
        
    if nll_normed_train_mmd is not None:
        axnlli.plot(nll_normed_train_mmd)

    if mmdi is not None:
        axmmdi.plot(mmdi)
        axmmdi.set_ylim(np.percentile(mmdi, [2.5, 97.5]) * np.array([0.95, 1.05]))
        axmmdi.set_yscale('log')

    if glm is not None:
        t_ker = np.arange(0, glm.eta.basis_values.shape[0], 1) * glm.eta.dt
        glm.eta.plot(t=t_ker, ax=axeta, exp_values=True, label=label, color=color)
        axeta.text(0.5, 0.75, 'b=' + str(np.round(glm.u0, 2)), color=color, transform=axeta.transAxes)

#     st_train.plot(ax=axd, ms=0.7, color=palette['d'])
    
    x_bar = np.arange(4)
    
#     st_fr_mmd.sweeps(np.arange(st_train.mask.shape[1])).plot(ax=axfr, ms=0.7, color=palette['mmd'])

#     axnll.bar(x_bar, [-nll_normed_train_mmd[-1], -nll_normed_train_ml[-1], -nll_normed_val_mmd, -nll_normed_val_ml[-1]], color=[palette['mmd'], palette['ml']] * 2)
#     axnll.set_xticks(x_bar)
#     axnll.set_xticklabels(['MMDt', 'MLt', 'MMDv', 'MLv'], rotation=60)

#     axmmd.bar([0, 1], [mmd_mmd, mmd_ml], color=[palette['mmd'], palette['ml']])
#     axmmd.set_yscale('log')
#     axmmd.set_xticks([0, 1])
#     axmmd.set_xticklabels(['MMD', 'ML'], rotation=60)

#     axisi.hist(isi_val, density=True, alpha=1, color=palette['d'], histtype='step', label='data', bins=bins_isi)
#     axisi.hist(isi_fr_ml, density=True, alpha=1, color=palette['ml'], histtype='step', label='ML-GLM', bins=bins_isi)
#     axisi.hist(isi_fr_mmd, density=True, alpha=1, color=palette['mmd'], histtype='step', label='MMD-GLM', bins=bins_isi)

    # axac.plot(autocov_ml[:argf_autocorr], color=palette['ml'], label='ML-GLM')
    if autocov_mmd is not None:
        axac.plot(autocov_mmd[1:argf_autocorr], color=color, label=label)
    
    return axloss, axnlli, axmmdi, axextra, axd, axfr, axeta, axisi, axpsth, axnll, axmmd, axac


def violinplot(ax, data, position, widths, showmeans, color):
    parts = ax.violinplot(data, positions=[position], widths=widths, showmeans=showmeans, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        pc.set_alpha(1)
    parts['cmeans'].set_color('k')
    parts['cmeans'].set_dashes('--')
    return ax
