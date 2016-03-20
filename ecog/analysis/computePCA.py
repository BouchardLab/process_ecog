#!/usr/bin/env python
from __future__ import division
import pdb,os,time,warnings
import numpy as np
from math import log
from numpy.linalg import *
from numpy.random import rand


def computePCA(X,n_components=None,whiten=True):
    """Computes the principal components of X
    
    Parameters
    ----------

    X : array, shape (n_features,n_samples)

    whiten : boolean, optional
        If True, perform an initial whitening of the data.
        If False, the data is assumed to be already white.

    n_components : int, optional
        Number of components to extract. If None, 
        n_components = n_features.

    Returns
    -------

    K : array, shape (n_components, n_features)
        If whiten is 'True', K is the pre-whitening matrix 
        projecting the data onto the principal components. 
        If whiten is 'False', K is 'None'.
    """

    n,m  = X.shape
    if n_components!=None:
        n = n_components
    X-=X.mean(1,keepdims=True)
    Ux,Sx = eig(np.cov(X))
    if whiten:
        K = np.sqrt(inv(np.diag(Ux))).dot(Sx.conj().T)[:n]
    else:
        K = Sx.conj().T[:n]
    X = K.dot(X)
    return X,K,Ux


