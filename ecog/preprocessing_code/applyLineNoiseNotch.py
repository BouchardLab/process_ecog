from __future__ import division
import numpy as np
from scipy.signal import firwin2,filtfilt

__authors__ = "Alex Bujan"

def applyLineNoiseNotch(X,rate):
    """Apply Notch filter at 60 Hz and its harmonics
    
    Parameters
    ----------

    X   : array
            ECoG data, dimensions (n_channels, n_timePoints)

    rate : float
            Number of samples per second

    Returns
    -------
    
    X   : array
            Denoised ECoG data, dimensions (n_channels, n_timePoints)

    """
    nyquist = rate/2
    notch   = 60.
    while notch<nyquist:
        filt = firwin2(1000+1,np.array([0,notch-1,notch-.5,\
                       notch+.5,notch+1,nyquist])/nyquist,\
                       [1,1,0,0,1,1])
        X   = filtfilt(filt,1,X)
        notch+=60.
    return X

