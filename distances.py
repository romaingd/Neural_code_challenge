import numpy as np


def kolmogorov_smirnov(x, y):
    '''
    Inspired from scipy.stats.ks2amp. Assumes that x and y are sorted.
    '''
    n1 = x.shape[0]
    n2 = y.shape[0]
    data_all = np.concatenate([x, y])
    cdf1 = np.searchsorted(x, data_all, side='right') / (1.0*n1)
    cdf2 = np.searchsorted(y, data_all, side='right') / (1.0*n2)
    d = np.max(np.absolute(cdf1 - cdf2))
    return(d)