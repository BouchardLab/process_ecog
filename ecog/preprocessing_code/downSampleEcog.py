from __future__ import division
import numpy as np
from scikits.samplerate import resample

__authors__ = "Alex Bujan"

def downsampleEcog(X,new,old):
    """Down-samples the ECoG signal from the original sampling frequency (of)
        to a new frequency (nf)
    
    Parameters
    ----------

    X   : array
            ECoG data, dimensions (n_channels, n_timePoints)

    new : float
            New sampling frequency

    old : float
            Original sampling frequency

    Returns
    -------
    
    Xds : array
            Downsampled ECoG data, same dimensions as input data

    Notes
    -----

    This script uses the python package samplerate found here:
    https://pypi.python.org/pypi/scikits.samplerate 

    This package provides a better replacement of Matlab's resample 
    function than Scipy. To install this package follow the instructions in 
    samplerate_installation.txt

    """
    for i in xrange(X.shape[0]):
        X_tmp = resample(X[i],new/old,'sinc_best')
        if i==0:
            Xds = np.zeros((X.shape[0],len(X_tmp)))
        Xds[i] = X_tmp.copy()
    return Xds

