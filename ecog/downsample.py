from __future__ import division
import numpy as np
from scipy.signal import resample
try:
    from tqdm import tqdm
except:
    tqdn = None

__authors__ = "Alex Bujan"

def downsample_ecog(X, new, old):
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

    Xds = np.ones((X.shape[0], np.ceil(X.shape[1] * new / old)))

    if tqdm is None:
        it = X
    else:
        it = tqdm(X, desc='Downsampling')

    for i, chan in enumerate(it):
        Xds[i] = resample(chan, np.ceil(len(chan) * new / old))

    return Xds
