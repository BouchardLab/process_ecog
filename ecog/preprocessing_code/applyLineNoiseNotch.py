from __future__ import division
import numpy as np
from scipy.signal import firwin2,freqz,filtfilt


def applyLineNoiseNotch(X,sampling_rate=400.):
    """Apply Notch filter at 60, 120 and 180 Hz
    
    Parameters
    ----------

    X   : array
            ECoG data, dimensions (n_channels, n_timePoints)

    sampling_rate : float, optional
            Number of samples per second

    Returns
    -------
    
    X   : array
            Denoised ECoG data, dimensions (n_channels, n_timePoints)

    Notes
    -----

    This script is a Python translation of Ben Dichter's original Matlab
    script.

    Author: Alex Bujan
    """
    nyquist_freq = sampling_rate/2
    f0 = np.array([0.,59.,59.5,60.5,61.,nyquist_freq])
    for i in xrange(3):
        fil = firwin2(1000+1,(f0+60*i)/nyquist_freq,[1,1,0,0,1,1])
        X   = filtfilt(fil,1,X)
    return X

