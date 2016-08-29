from __future__ import division
import numpy as np
from pyfftw.interfaces.numpy_fft import fft, ifft, fftfreq

__authors__ = "Alex Bujan"


def apply_hilbert_transform(X, rate, center=None, sd=None, kernel=None):
    """
    Apply bandpass filtering with Hilbert transform using a Gaussian kernel
    or prespecified kernel.
    
    Parameters
    ----------
    X : array
        Input data, dimensions (n_channels, n_timePoints)
    rate : float
        Number of samples per second
    center : float
        Center frequency of the Gaussian kernel
    sd : float,
        Standard deviation of the Gaussian kernel

    Returns
    -------
    Xc : array
        Bandpassed analytical signal (dtype: complex)
    """

    if kernel is None:
        assert center is not None
        assert sd is not None
    else:
        assert center is None
        assert sd is None

    # frequencies
    time_samples = X.shape[-1]
    freq = fftfreq(time_samples, 1./rate)
    # heaviside kernel
    h = np.zeros(len(freq))
    h[freq>0]=2.
    h[0]=1.
    if kernel is None:
        # bandpass transfer function
        k  = np.exp((-(np.abs(freq)-center)**2)/(2*(sd**2)))
    else:
        k = kernel

    # compute analytical signal
    Xc = ifft(fft(X)*h*k)

    return Xc
