import numpy as np


def kolmogorov_smirnov(x, y):
    '''
    Inspired from scipy.stats.ks2amp. Assumes that x and y are sorted.
    No need to normalize by the length of the samples (equal)
    '''
    data_all = np.concatenate([x, y])
    n = x.shape[0]
    cdf1 = np.searchsorted(x, data_all, side='right') / (1.0*n)
    cdf2 = np.searchsorted(y, data_all, side='right') / (1.0*n)
    d = np.max(np.absolute(cdf1 - cdf2))
    return(d)


def custom_abs(x):
    '''
    Custom adaptation of the absolute value for KS optimization
    '''
    return(np.abs(x + (x>0)))


def kolmogorov_smirnov_opt(x, y, idx=np.arange(1, 51)):
    '''
    Optimized (~30%) approximation of KS distance for same length (50) samples.
    Inspired from scipy.stats.ks2amp. Assumes that x and y are sorted.
    Works only for this challenge. In particular, it's not a distance.
    '''
    d = np.max(custom_abs(np.searchsorted(x, y, side='right') - idx))
    return(d)