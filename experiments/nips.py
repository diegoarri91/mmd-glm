from cycler import cycler

tick_labelsize = 16
f2 = 20

paper = {'figure.figsize': (3.5, 3.),  'axes.titlesize': 20, 'axes.labelsize': 20, 'lines.linewidth': 1.5, 'lines.markersize': 5,
         'xtick.labelsize': tick_labelsize, 'xtick.major.size': 2.5, 'xtick.major.pad': 2, 
         'ytick.labelsize': tick_labelsize, 'ytick.major.size': 2.5, 'ytick.major.pad': 2,
         'errorbar.capsize': 5, 'axes.labelpad': 0,
         'legend.fontsize': tick_labelsize, 'axes.spines.top': False, 'axes.spines.right': False,
         'boxplot.flierprops.markersize': 2}

#          'axes.prop_cycle': cycler(color=['dodgerblue', '#CD4545', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
#                                                             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

width, capsize, space = 15, 3, 0.5
strip_ms = 2.5
scatter_ms = 12
ms_raster = 0.95