from __future__ import division
import multiprocessing
import numpy as np
from scipy.signal import resample
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x

__authors__ = "Alex Bujan"

def resample_wrap(args):
    chan, new_time = args
    return resample(chan, new_time)

def downsample_ecog(X, new, old, parallel=True):
    """
    Down-samples the ECoG signal from the original sampling frequency (old)
    to a new frequency (new)

    Parameters
    ----------
    X : array
        Input data, dimensions (n_channels, n_timePoints)
    new : float
        New sampling frequency
    old : float
        Original sampling frequency

    Returns
    -------
    Xds : array
        Downsampled data, dimensions (n_channels, n_timePoints_new)
    """
    n_channels, time = X.shape
    new_time = int(np.ceil(time * new / old))

    if parallel:
        pool = multiprocessing.Pool()
        result = pool.map(resample_wrap, [(c, new_time) for c in X])
        pool.close()
        return np.vstack([r[np.newaxis, :] for r in result])
    else:
        Xds = np.ones((n_channels, new_time))

        for i, chan in enumerate(tqdm(X, desc='Downsampling')):
            Xds[i] = resample_wrap((chan, new_time))

        return Xds
