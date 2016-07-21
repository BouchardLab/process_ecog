from __future__ import division
import numpy as np
#from scikits.samplerate import resample  # I'm having trouble installing this library
from scipy.signal import resample as scipy_resample
from tqdm import tqdm

__authors__ = "Alex Bujan"

def downsampleEcog(X, new, old):
    """Down-samples the ECoG signal from the original sampling frequency (of)
        to a new frequency (nf)

    Parameters
    ----------

    X   : array
            Input data, dimensions (n_channels, n_timePoints)

    new : float
            New sampling frequency

    old : float
            Original sampling frequency

    Returns
    -------

    Xds : array
            Downsampled data, dimensions (n_channels, n_timePoints)

    Notes
    -----

    This script uses the python package samplerate found here:
    https://pypi.python.org/pypi/scikits.samplerate

    This package provides a better replacement of Matlab's resample
    function than Scipy. To install this package follow the instructions in
    samplerate_installation.txt

    """
    Xds = np.ones((X.shape[0], np.ceil(X.shape[1] * new / old)))

    for i, chan in enumerate(tqdm(X, desc='Downsampling')):
        Xds[i] = scipy_resample(chan, np.ceil(len(chan) * new / old))
        #X_tmp = resample(X[i],new/old,'sinc_best')

    #Xds = scipy_resample(X, np.round(X.shape[1]*new/old), axis=1)

    return Xds
