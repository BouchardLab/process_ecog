from __future__ import division
import numpy as np
from scipy.signal import firwin2, filtfilt
try:
    from tqdm import tqdm
except:
    tqdn = None

__authors__ = "Alex Bujan"

def apply_linenoise_notch(X, rate):
    """
    Apply Notch filter at 60 Hz and its harmonics
    
    Parameters
    ----------
    X : array
        Input data, dimensions (n_channels, n_timePoints)
    rate : float
        Number of samples per second

    Returns
    -------
    X : array
        Denoised data, dimensions (n_channels, n_timePoints)
    """

    nyquist = rate/2
    noise_hz   = 60.
    notches = np.arange(noise_hz, nyquist, noise_hz)
    if tqdn is None:
        it = tqdm(notches, 'applying notch filters')
    else:
        it = notches

    for notch in it:
        filt = firwin2(1000+1,
                       np.array([0, notch-1, notch-.5,
                                 notch+.5, notch+1, nyquist]) / nyquist,
                       [1,1,0,0,1,1])
        X = filtfilt(filt, 1, X)
    return X

