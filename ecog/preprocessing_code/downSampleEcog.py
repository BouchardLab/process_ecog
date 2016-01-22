from __future__ import division
import numpy as np
from scikits.samplerate import resample

def downsampleEcog(X,nf,of):
    """Down-samples the ECoG signal from the original sampling frequency (of)
        to a new frequency (nf)
    
    Parameters
    ----------

    X   : array
            ECoG data, dimensions (n_channels, n_timePoints)

    nf  : float
            New sampling frequency

    of  : float
            Original sampling frequency

    Returns
    -------
    
    Xds : array
            Downsampled ECoG data, same dimensions as input data

    Notes
    -----

    This script uses the python package samplerate found here:
    https://pypi.python.org/pypi/scikits.samplerate 
    
    This packege provides a better replacement of Matlab's resample function
    than the Scipy. To install this package follow the instructions in 
    samplerate_installation.txt
    
    """
    m = np.round(nf/400)

    n = np.round(of/3.0518e3)

    Xds = np.zeros(X.shape)

    for i in xrange(X.shape[0]):
        Xds[i] = resample(X[i],(m*2**11)/(n*5**6),'sinc_best')

    return Xds

