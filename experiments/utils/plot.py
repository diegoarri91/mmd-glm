import matplotlib.pyplot as plt


def plot_errorbar(y, yerr, color, ax, ms=1, mew=1, **kwargs):
    for ii, (_y, _yerr, c) in enumerate(zip(y, yerr, color)):
        ax.plot(ii, _y, color=c, marker='_', ms=ms, mew=mew)
        ax.errorbar(ii, _y, yerr=_yerr, color=c, fmt='_', ms=0, **kwargs)


def plot_layout_fit(figsize=(16, 12)):
    r1 = 2
    r2 = int(r1 / 2)
    c3 = 1
    c1 = 2 * c3
    c2 = 2 * c1
    nrows = 3 * r1
    ncols = 4 * c1
    fig = plt.figure(figsize=figsize)
    
    axloss = plt.subplot2grid((nrows, ncols), (0, 0), rowspan=r1, colspan=c1)
    axloss.set_xlabel('iterations')
    axloss.set_ylabel('loss')
    
    axnlli = plt.subplot2grid((nrows, ncols), (0, c1), rowspan=r1, colspan=c1)
    axnlli.set_xlabel('iterations')
    axnlli.set_ylabel('nll')
    
    axmmdi = plt.subplot2grid((nrows, ncols), (0, 2 * c1), rowspan=r1, colspan=c1)
    axmmdi.set_xlabel('iterations')
    axmmdi.set_ylabel('mmd')
    
    axextra = plt.subplot2grid((nrows, ncols), (0, 3 * c1), rowspan=r1, colspan=c1)
    
    axd = plt.subplot2grid((nrows, ncols), (r1, 0), rowspan=r2, colspan=c2)
    axd.spines['right'].set_visible(True)
    axd.spines['top'].set_visible(True)
    axd.tick_params(axis='both', labelbottom=False, labelleft=False)
    axd.set_ylabel('data')
    
    axfr = plt.subplot2grid((nrows, ncols), (r1 + r2, 0), rowspan=r2, colspan=c2, sharex=axd)
    axfr.spines['right'].set_visible(True)
    axfr.spines['top'].set_visible(True)
    axfr.tick_params(axis='both', labelbottom=False, labelleft=False)
    axfr.set_ylabel('model')
    
    axeta = plt.subplot2grid((nrows, ncols), (r1, c2), rowspan=r1, colspan=c1)
    axeta.set_xlabel('time (ms)')
    axeta.set_ylabel('gain')
    
    axisi = plt.subplot2grid((nrows, ncols), (r1, c2 + c1), rowspan=r1, colspan=c1)
    axisi.set_xlabel('time (ms)')
    axisi.set_ylabel('pdf (isi)')
    
    axpsth = plt.subplot2grid((nrows, ncols), (2 * r1, 0), rowspan=r1, colspan=c2, sharex=axd)
    axpsth.spines['right'].set_visible(True)
    axpsth.spines['top'].set_visible(True)
    axpsth.set_ylabel('rate')
    axpsth.set_xlabel('time (ms)')
    
    axnll = plt.subplot2grid((nrows, ncols), (2 * r1, c2), rowspan=r1, colspan=c3)
    axnll.set_ylabel('nll')
    
    axmmd = plt.subplot2grid((nrows, ncols), (2 * r1, c2 + c3), rowspan=r1, colspan=c3)
    axmmd.set_ylabel('mmd')
    
    axac = plt.subplot2grid((nrows, ncols), (2 * r1, c2 + c1), rowspan=r1, colspan=c1)
    axac.set_xlabel('time (ms)')
    axac.set_ylabel('autocorrelation')
    axac.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    
    fig.subplots_adjust(hspace=0.7, wspace=0.8)
    
    return fig, (axloss, axnlli, axmmdi, axextra, axd, axfr, axeta, axisi, axpsth, axnll, axmmd, axac)
