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

    This package provides a better replacement of Matlab's resample 
    function than Scipy. To install this package follow the instructions in 
    samplerate_installation.txt

    This script is a Python translation of Ben Dichter's original Matlab
    script.

    Author: Alex Bujan
    """
    nf = np.round(nf/400)*2**11

    of = np.round(of/3.0518e3)*5**6

    Xds = np.zeros(X.shape)

    for i in xrange(X.shape[0]):
        Xds[i] = resample(X[i],nf/of,'sinc_best')

    return Xds

