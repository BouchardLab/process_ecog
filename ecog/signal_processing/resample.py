from __future__ import division
import numpy as np
from scipy.signal import resample

__authors__ = "Alex Bujan"


def resample_ecog(X, new_freq, old_freq, axis=-1):
    """
    Resamples the ECoG signal from the original
    sampling frequency to a new frequency.

    Parameters
    ----------
    X : array
        Input data, dimensions (n_channels, n_timePoints)
    new_freq : float
        New sampling frequency
    old_freq : float
        Original sampling frequency

    Returns
    -------
    Xds : array
        Downsampled data, dimensions (n_channels, n_timePoints_new)
    """
    time = X.shape[axis]
    new_time = int(np.ceil(time * new_freq / old_freq))

    Xds = resample(X, new_time, axis=axis)

    return Xds
