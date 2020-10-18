import numpy as np


def cum_isi(isi, bins=None):
    cum_isi, _ = np.histogram(isi, bins=bins)
    cum_isi = np.cumsum(cum_isi)
    return cum_isi / cum_isi[-1]
