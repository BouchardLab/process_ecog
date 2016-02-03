from __future__ import division
import numpy as np

try:
    from pyfftw.interfaces.numpy_fft import fft,ifft,fftfreq
except:
    from numpy.fft import fft,ifft,fftfreq

__authors__ = "Alex Bujan"


def applyHilbertTransform(X,rate,center,sd):
    """Apply bandpass filtering with Hilbert transform using a Gaussian kernel
    
    Parameters
    ----------
    X   : array
            Input data, dimensions (n_channels, n_timePoints)

    rate : float
            Number of samples per second

    center : float
            Center frequency of the Gaussian kernel
    
    sd  : float,
            Standard deviation of the Gaussian kernel

    Returns
    -------

    Xc   : array
            Bandpassed analytical signal (dtype: complex)

    """
    #frequencies
    T = X.shape[-1]
    freq = fftfreq(T,1/rate)
    #heaviside kernel
    h = np.zeros(len(freq))
    h[freq>0]=2.
    h[0]=1.
    #bandpass transfer function
    k  = np.exp((-(np.abs(freq)-center)**2)/(2*(sd**2)))
    #compute analytical signal
    Xc = ifft(fft(X)*h*k)
    return Xc
