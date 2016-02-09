from __future__ import division
import numpy as np

def subtractCAR(X,b_size=16)

    n,m = X.shape
    s = n//b_size
    r = n%b_size

    X_1 = X[:n-r]
    X_2 = X[n-r:]

    X_1  = X_1.reshape((s,b_size,m))
    X_1 -= X_1.mean(1,keepdims=True)
    X_2 -= X_2.mean(0,keepdims=True)

    return np.vstack([X_1.reshape((s*b_size,m)),X_2])
