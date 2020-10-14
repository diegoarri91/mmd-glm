import argparse
from datetime import datetime
import h5py
import itertools
import os
import pickle
import time

import matplotlib.pyplot as plt
import pandas as pd
from torch.optim import Adam

from experiments.utils.kernels import *
from experiments.utils.plot import plot_layout_fit
from gglm.glm.mmdglm import MMDGLM
from gglm.glm.torchglm import TorchGLM
from gglm.metrics import bernoulli_log_likelihood_poisson_process, MMD, time_rescale_transform
from kernel.values import KernelBasisValues
from sptr.sptr import SpikeTrain
from utils import after_metrics, fun_metrics_mmd


server_name = os.uname()[1]
palette = dict(d='C0', ml='C2', mmd='C1')

dic = {'phi_autocov': phi_autocov}

# python huk_mmd.py --phi phi_autocov --lam_mmd 1e9 --padding 250 --beta0 0.95 --beta1 0.99 --plot True

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--phi', type=str, default=None)
parser.add_argument('--kernel', type=str, default=None)
parser.add_argument('--lam_mmd', type=float, nargs='+', default=[1e0])
parser.add_argument('--biased', type=bool, default=True)
parser.add_argument('--n_batch_fr', type=int, default=[800])
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--clip', type=float, default=None)
parser.add_argument('--num_epochs', type=int, default=400)
parser.add_argument('--initialization', type=str, default='zero')

parser.add_argument('--padding', type=int, default=None)
parser.add_argument('--beta0', type=float, default=[0])
parser.add_argument('--beta1', type=float, default=[0])
parser.add_argument('--n_repetitions', type=int, default=1)

parser.add_argument('--plot', type=bool, default=False)

args = parser.parse_args()

phi = None if args.phi is None else dic[args.phi]
kernel = None if args.kernel is None else dic[args.kernel]

biased, lr, clip, num_epochs, initialization = args.biased, args.lr, args.clip, args.num_epochs, args.initialization 
padding = args.padding
plot = args.plot

if padding is not None:
    kernel_kwargs = dict(padding=args.padding)
else:
    kernel_kwargs = None

list_args = itertools.product(args.lam_mmd, args.n_batch_fr, args.beta0, args.beta1)
list_args = list_args * args.n_repetitions
    
for lam_mmd, n_batch_fr, beta0, beta1 in list_args:
    print(lam_mmd)
    #################################################################################
    # load data
    #################################################################################

    path = "./huk_p110509b_dots.h5"
    f = h5py.File(path, "r")

    mask_spikes = np.array(np.stack((f['spk']), axis=1), dtype=bool)
    t = np.arange(0, mask_spikes.shape[0], 1)
    dt = 1

    files_folder = '/home/diego/storage/projects/generative-glm/experiments/figure4/'

    idx_train = np.arange(100, 200, 2)
    idx_val = idx_train + 1

    st = SpikeTrain(t, mask_spikes)
    mask_spikes_train = mask_spikes[:, idx_train]

    n_spk_train = np.sum(mask_spikes_train)
    fr_train = np.mean(np.sum(mask_spikes_train, 0) / (t[-1] - t[0] + t[1]) * 1000)
    fr2_train = np.mean((np.sum(mask_spikes_train, 0) / (t[-1] - t[0] + t[1]) * 1000)**2)
    nll_pois_proc_train = -bernoulli_log_likelihood_poisson_process(mask_spikes_train)
    autocov_train = np.mean(raw_autocorrelation(mask_spikes_train, biased=False), 1)
    st_train = SpikeTrain(t, mask_spikes_train)
    isi_train = st_train.isi_distribution()
    mean_isi_train = np.mean(isi_train)

    mask_spikes_val = mask_spikes[:, idx_val]
    n_spk_val = np.sum(mask_spikes_val)
    fr_val = np.mean(np.sum(mask_spikes_val, 0) / (t[-1] - t[0] + t[1]) * 1000)
    nll_pois_proc_val = -bernoulli_log_likelihood_poisson_process(mask_spikes_val)
    autocov_val = np.mean(raw_autocorrelation(mask_spikes_val, biased=False), 1)

    st_val = SpikeTrain(t, mask_spikes_val)
    isi_val = st_val.isi_distribution()
    mean_isi_val = np.mean(isi_val)

    bins_isi = np.arange(0, 400, 10)

    argf_autocorr = 200

    #################################################################################
    # load ml
    #################################################################################

    files_folder = '/home/diego/storage/projects/generative-glm/experiments/figure4/'
    ml_file = 'mln5.pk'
    with open(files_folder + ml_file, "rb") as fit_file:
        dic_ml = pickle.load(fit_file)
    eta_ml = KernelBasisValues(dic_ml['basis'], [0, dic_ml['basis'].shape[0]], 1, coefs=dic_ml['eta_coefs'])
    glm_ml = TorchGLM(u0=dic_ml['u0_ml'], eta=eta_ml)
    nll_normed_train_ml = dic_ml['nll_normed_train_ml']
    nll_normed_val_ml = dic_ml['nll_normed_val_ml']
    bins_ks = dic_ml['bins_ks']
    t_ker = np.arange(0, eta_ml.basis_values.shape[0], 1) * dt

    r_train_dc_ml, r_val_dc_ml, r_fr_ml, mask_spikes_fr_ml = dic_ml['r_train_dc_ml'], dic_ml['r_val_dc_ml'], dic_ml['r_fr_ml'], dic_ml['mask_spikes_fr_ml']

    n, last_peak = 5, 100

    st_fr_ml = SpikeTrain(st_val.t, mask_spikes_fr_ml)
    isi_fr_ml = st_fr_ml.isi_distribution()
    mean_isi_fr_ml = np.mean(isi_fr_ml)
    mean_r_fr_ml = np.mean(r_fr_ml, 1)
    sum_r_fr_ml = np.sum(r_fr_ml, 1)
    autocov_ml = np.mean(raw_autocorrelation(mask_spikes_fr_ml, biased=False), 1)

    z_ml_train, ks_ml_train = time_rescale_transform(dt, st_train.mask, r_train_dc_ml)
    values, bins_ks = np.histogram(np.concatenate(z_ml_train), bins=bins_ks)
    z_cum_ml_train = np.append(0., np.cumsum(values) / np.sum(values))

    z_ml_val, ks_ml_val = time_rescale_transform(dt, st_val.mask, r_val_dc_ml)
    values, _ = np.histogram(np.concatenate(z_ml_val), bins=bins_ks)
    z_cum_ml_val = np.append(0., np.cumsum(values) / np.sum(values))

    #################################################################################
    # fit mmd
    #################################################################################

    n_samples = 8000

    n_metrics = 1
    log_likelihood = True
    control_variates = False

    dtime = datetime.now()
    dtime = str(dtime.year) + '/' + str(dtime.month) + '/' + str(dtime.day) + '-' + str(dtime.hour) + ':' + str(dtime.minute) + ':' + str(dtime.second)

    time0 = time.time()
    u00 = glm_ml.u0
    eta0 = glm_ml.eta.copy()

    if initialization == 'zero':
        eta0.coefs = eta0.coefs * 0

    mmdglm = MMDGLM(u0=u00, eta=eta0)
    optim = Adam(mmdglm.parameters(), lr=lr, betas=(beta0, beta1), amsgrad=False)

    loss_mmd, nll_train, metrics_mmd = mmdglm.train(t, torch.from_numpy(mask_spikes_train), phi=phi, kernel=kernel, 
                                                    log_likelihood=log_likelihood,
                                                   n_batch_fr=n_batch_fr, lam_mmd=lam_mmd, biased=biased, kernel_kwargs=kernel_kwargs, optim=optim, clip=clip, 
                                                    num_epochs=num_epochs, verbose=True, metrics=fun_metrics_mmd, n_metrics=n_metrics, control_variates=control_variates)

    loss_mmd, nll_train = np.array(loss_mmd), np.array(nll_train)
    nll_normed_train_mmd = (nll_train - nll_pois_proc_train) / np.log(2) / n_spk_train
    metrics_mmd['mmd'] = np.array(metrics_mmd['mmd'])
    iterations_mmd = np.arange(1, num_epochs + 1, 1)

    _, r_dc_mmd_train = mmdglm.sample_conditioned(st_train.t, st_train.mask)
    z_mmd_train, ks_mmd_train = time_rescale_transform(dt, st_train.mask, r_dc_mmd_train)
    values, bins_ks = np.histogram(np.concatenate(z_mmd_train), bins=bins_ks)
    z_cum_mmd_val = np.append(0., np.cumsum(values) / np.sum(values))

    _, r_dc_mmd_val = mmdglm.sample_conditioned(st_val.t, st_val.mask)
    _, r_fr_mmd, mask_spikes_fr_mmd = mmdglm.sample(st_val.t, shape=(n_samples,))
    st_fr_mmd = SpikeTrain(st_train.t, mask_spikes_fr_mmd)
    fr_mmd = np.sum(mask_spikes_fr_mmd, 0) / (dt * mask_spikes_fr_mmd.shape[0]) * 1000

    isi_fr_mmd = st_fr_mmd.isi_distribution()

    mean_isi_fr_mmd = np.mean(isi_fr_mmd)

    autocov_mmd = np.mean(raw_autocorrelation(mask_spikes_fr_mmd, biased=False), 1)
    rmse_autocov = np.sqrt(np.mean((autocov_mmd - autocov_train)**2))
    nll_val_mmd = -(np.sum(np.log(1 - np.exp(-dt * r_dc_mmd_val[st_val.mask]) + 1e-24) ) - \
                dt * np.sum(r_dc_mmd_val[~st_val.mask]))
    nll_normed_val_mmd = (nll_val_mmd - nll_pois_proc_val) / np.log(2) / n_spk_val
    z_mmd_val, ks_mmd_val = time_rescale_transform(dt, st_val.mask, r_dc_mmd_val)
    values, bins_ks = np.histogram(np.concatenate(z_mmd_val), bins=bins_ks)
    z_cum_mmd_val = np.append(0., np.cumsum(values) / np.sum(values))

    time1 = time.time()
    etime = (time1 - time0) / 60
    print('\n', 'took', time1 - time0, 'seconds', (time1 - time0) / 60, 'minutes')

    mmd_mmd = MMD(t, torch.from_numpy(st_train.mask), torch.from_numpy(mask_spikes_fr_mmd), phi=phi, kernel=kernel, biased=biased).item()
    mmd_ml = MMD(t, torch.from_numpy(st_train.mask), torch.from_numpy(mask_spikes_fr_ml), phi=phi, kernel=kernel, biased=biased).item()

    #################################################################################
    # after mmd statistics
    #################################################################################

    mmdglm_after = MMDGLM(u0=mmdglm.u0, eta=mmdglm.eta.copy())

    _loss_mmd, _nll_train, metrics_after = mmdglm_after.train(t, torch.from_numpy(mask_spikes_train), phi=phi, kernel=kernel, 
                                                    log_likelihood=False, n_batch_fr=n_batch_fr, lam_mmd=lam_mmd, biased=biased, 
                                                    kernel_kwargs=kernel_kwargs, optim=torch.optim.SGD(mmdglm_after.parameters(), lr=0),
                                                   num_epochs=50, verbose=True, metrics=after_metrics, n_metrics=1, control_variates=False)

    mmd_mean = np.mean(metrics_after['mmd'])
    mmd_sd = np.std(metrics_after['mmd'])
    params_grad = np.concatenate((np.array(metrics_after['b_grad'])[:, None], np.array(metrics_after['eta_coefs_grad'])), axis=1)
    params_grad_mean = np.mean(params_grad, 0)
    params_grad_cov = np.cov(params_grad.T)

    fr_mean_mean, fr_mean_sd = np.mean(metrics_after['mu_fr']), np.std(metrics_after['mu_fr'])
    fr_max_mean, fr_max_sd = np.mean(metrics_after['fr_max']), np.std(metrics_after['fr_max'])


    for key, val in metrics_mmd.items():
        metrics_mmd[key] = np.array(val)

    if phi is not None:
        ker_name = phi.__name__
    else:
        ker_name = kernel.__name__

    if nll_train is None or len(nll_train)==0:
        nll_train = np.zeros(len(iterations_mmd))
        nll_normed_train_mmd = np.zeros(len(iterations_mmd))

    if plot:
        x_bar = np.arange(4)

        fig, (axloss, axnlli, axmmdi, axextra, axd, axfr, axeta, axisi, axpsth, axnll, axmmd, axac) = plot_layout_fit()

        axloss.plot(loss_mmd)
        axloss.set_ylim(np.percentile(loss_mmd, [2.5, 97.5]) * np.array([0.95, 1.05]))

        axnlli.plot(nll_normed_train_mmd)

        mmdi = loss_mmd - nll_train
        axmmdi.plot(mmdi)
        axmmdi.set_ylim(np.percentile(mmdi, [2.5, 97.5]) * np.array([0.95, 1.05]))

        axextra.plot(metrics_mmd['mmd'])
        axextra.set_ylim(np.percentile(metrics_mmd['mmd'], [2.5, 97.5]) * np.array([0.95, 1.05]))

        glm_ml.eta.plot(t=t_ker, ax=axeta, exp_values=True, label='ML-GLM', color=palette['ml'])
        mmdglm.eta.plot(t=t_ker, ax=axeta, exp_values=True, label='MMD-GLM', color=palette['mmd'])
        axeta.text(0.5, 0.85, 'b=' + str(np.round(glm_ml.u0, 2)), color=palette['ml'], transform=axeta.transAxes)
        axeta.text(0.5, 0.75, 'b=' + str(np.round(mmdglm.u0, 2)), color=palette['mmd'], transform=axeta.transAxes)

        st_train.plot(ax=axd, ms=0.7, color=palette['d'])

        st_fr_mmd.sweeps(np.arange(st_train.mask.shape[1])).plot(ax=axfr, ms=0.7, color=palette['mmd'])

        axnll.bar(x_bar, [-nll_normed_train_mmd[-1], -nll_normed_train_ml[-1], -nll_normed_val_mmd, -nll_normed_val_ml[-1]], color=[palette['mmd'], palette['ml']] * 2)
        axnll.set_xticks(x_bar)
        axnll.set_xticklabels(['MMDt', 'MLt', 'MMDv', 'MLv'], rotation=60)

        axmmd.bar([0, 1], [mmd_mmd, mmd_ml], color=[palette['mmd'], palette['ml']])
        axmmd.set_yscale('log')
        axmmd.set_xticks([0, 1])
        axmmd.set_xticklabels(['MMD', 'ML'], rotation=60)

        axisi.hist(isi_val, density=True, alpha=1, color=palette['d'], histtype='step', label='data', bins=bins_isi)
        axisi.hist(isi_fr_ml, density=True, alpha=1, color=palette['ml'], histtype='step', label='ML-GLM', bins=bins_isi)
        axisi.hist(isi_fr_mmd, density=True, alpha=1, color=palette['mmd'], histtype='step', label='MMD-GLM', bins=bins_isi)

        axac.plot(autocov_val[1:argf_autocorr], color=palette['d'], label='data')
        # axac.plot(autocov_ml[:argf_autocorr], color=palette['ml'], label='ML-GLM')
        axac.plot(autocov_mmd[1:argf_autocorr], color=palette['mmd'], label='MMD-GLM')

        fig.savefig('./tmp_fit.pdf')


    list_files = os.listdir('/home/diego/storage/projects/generative-glm/experiments/figure4/')

    file_name = 'mmd_' + ker_name + '_' + 'lammmd' + str(lam_mmd) + 'biased' + str(biased) +  \
               '_epochs' + str(num_epochs) + '_' + 'lr' + str(lr) + '_' + server_name
    ii = 1
    while file_name + '_' + str(ii) + '.pk' in list_files:
        ii += 1
    file_name = file_name + '_' + str(ii) + '.pk'

    dic = dict(server=os.uname()[1], dtime=dtime, idx_train=idx_train, idx_val=idx_val, basis=None, 
               u0_ml=glm_ml.u0, eta_coefs_ml=glm_ml.eta.coefs, ker_name=ker_name, ml_file=ml_file, 
               lam_mmd=lam_mmd, biased=biased, lr=lr, clip=clip, n_batch_fr=n_batch_fr, loss_mmd=loss_mmd, nll_train=nll_normed_train_mmd, 
               nll_normed_val_mmd=nll_normed_val_mmd, metrics_mmd=metrics_mmd, u0_mmd=mmdglm.u0, eta_coefs_mmd=mmdglm.eta.coefs, autocov_mmd=autocov_mmd, vals_isi=vals_isi)

    path = '/home/diego/storage/projects/generative-glm/experiments/figure4/' + file_name
    with open(path, "wb") as fit_file:
        pickle.dump(dic, fit_file)

    ker_kwargs = None if kernel_kwargs is None else list(kernel_kwargs.values())
    dic_df = dict(file_name=file_name, server=os.uname()[1], dtime=dtime, idx_train=idx_train, idx_val=idx_val, mmd_ker='score_function', ker_name=ker_name, n_basis=n, 
                  last_peak=last_peak, num_epochs=num_epochs, log_likelihood=log_likelihood, control_variates=control_variates, 
                    lam_mmd=lam_mmd, biased=biased, lr=lr, beta0=beta0, beta1=beta1, clip=clip, n_batch_fr=n_batch_fr, etime=etime, loss_mmd=loss_mmd[-1], 
                    nll_train=nll_train[-1], nll_normed_train=nll_normed_train_mmd[-1], 
                  nll_normed_val_mmd=nll_normed_val_mmd, fr_mmd_m=np.mean(fr_mmd), fr_mmd_max=np.max(fr_mmd), mmd_mmd=mmd_mmd, 
                  u0=mmdglm.u0, eta_coefs=mmdglm.eta.coefs, mmd_mean=mmd_mean, mmd_sd=mmd_sd, params_grad_mean=params_grad_mean, params_grad_cov=params_grad_cov, 
                    fr_mean_mean=fr_mean_mean, fr_mean_sd=fr_mean_sd, fr_max_mean=fr_max_mean, fr_max_sd=fr_max_sd, initialization=initialization, 
                  kernel_kwargs=ker_kwargs)

    _df = pd.DataFrame(pd.Series(dic_df)).T

    df = pd.read_json('./mmd_summary_new.json')
    df = df.append(_df, ignore_index=True)
    df.to_json('./mmd_summary_new.json', double_precision=15)