from __future__ import division
import numpy as np

def subtract_CAR(X, b_size=16):

    channels, timepoints = X.shape
    s = channels // b_size
    r = channels % b_size

    X_1 = X[:channels-r]
    X_2 = X[channels-r:]

    X_1  = X_1.reshape((s, b_size,m))
    X_1 -= X_1.mean(1, keepdims=True)
    X_2 -= X_2.mean(0, keepdims=True)

    return np.vstack([X_1.reshape((s*b_size, m)), X_2])
