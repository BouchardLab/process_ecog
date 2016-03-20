#!/usr/bin/env python
from __future__ import division
import numpy as np
from computePCA import computePCA


def demodulation(X):
    """Demodulates X. X is complex-valued.
    
    Parameters
    ----------

    X : array, shape (n_features,n_samples)
        input data

    Returns
    -------

    X : array, shape (n_components,n_samples)
        demodulated signal

    """

    pc1 = computePCA(X,n_components=1,whiten=True)[0]

    return X*np.exp(-np.angle(pc1)*1j)



