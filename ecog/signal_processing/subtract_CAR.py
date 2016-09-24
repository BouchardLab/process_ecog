from __future__ import division
import numpy as np

def subtract_CAR(X, b_size=16):
    """
    Compute and subtract common average reference in 16 channel blocks.
    """

    channels, time_points = X.shape
    s = channels // b_size
    r = channels % b_size

    X_1 = X[:channels-r]
    X_2 = X[channels-r:]

    X_1  = X_1.reshape((s, b_size, time_points))
    X_1 -= np.nanmean(X_1, axis=1, keepdims=True)
    X_2 -= np.nanmean(X_2, axis=0, keepdims=True)

    return np.vstack([X_1.reshape((s*b_size, time_points)), X_2])
