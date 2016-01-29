from __future__ import division
import numpy as np
from scipy.io import loadmat

__authors__ = "Alex Bujan"

def deleteBadTimeSegments(X,sampling_rate,matFile):
    """Deletes time sections of data containing artifacts.
    
    Parameters
    ----------

    X   : array
            Bad data, dimensions (n_channels, n_timePoints)

    sampling_rate : float
            Number of samples per second

    matFile : string
            Path to a mat file with the info about badSegmts

    Returns
    -------

    X   : array
            Artifact-free data

    """
    badSegmts = loadmat(matFile)['badTimeSegments']
    T = np.arange(X.shape[1])/sampling_rate
    bp = np.searchsorted(T,badSegmts)
    ids = []
    for i in xrange(len(bp)):
        ids.append(np.arange(bp[i,0],bp[i,1]))
    ids = np.concatenate(ids)
    X = np.delete(X,ids,1)
    return X
