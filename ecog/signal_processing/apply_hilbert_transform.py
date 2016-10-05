from __future__ import division
import numpy as np
from numpy.fft import fftfreq

try:
    from accelerate.mkl.fftpack import fft, ifft
except ImportError:
    try:
        from pyfftw.interfaces.numpy_fft import fft, ifft
    except ImportError:
        from numpy.fft import fft, ifft


__authors__ = "Alex Bujan, Jesse Livezey"


def gaussian(X, rate, center, sd):
    n_channels, time = X.shape
    freq = fftfreq(time, 1./rate)

    k  = np.exp((-(np.abs(freq)-center)**2)/(2*(sd**2)))

    return k

def hamming(X, rate, min_freq, max_freq):
    n_channels, time = X.shape
    freq = fftfreq(time, 1./rate)

    pos_in_window = np.logical_and(freq >= min_freq, freq <= max_freq)
    neg_in_window = np.logical_and(freq <= -min_freq, freq >= -max_freq)

    k = np.zeros(len(freq))
    window_size = np.count_nonzero(pos_in_window)
    window = np.hamming(window_size)
    k[pos_in_window] = window
    window_size = np.count_nonzero(neg_in_window)
    window = np.hamming(window_size)
    k[neg_in_window] = window

    return k

def transform(args):
    X, h, kernel = args
    # Compute analytical signal
    if kernel is None:
        Xc = ifft(fft(X)*h)
    else:
        Xc = ifft(fft(X)*h*kernel)
    return Xc

def apply_hilbert_transform(X, rate, kernel=None, parallel=True):
    """
    Apply bandpass filtering with Hilbert transform using
    a prespecified kernel.
    
    Parameters
    ----------
    X : array
        Input data, dimensions (n_channels, n_time_points)
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

    n_channels, time = X.shape
    freq = fftfreq(time, 1./rate)

    # heaviside kernel
    h = np.zeros(len(freq))
    h[freq > 0]=2.
    h[0]=1.
    h = h[np.newaxis, :]
    if kernel is not None:
        kernel = kernel[np.newaxis, :]

    return transform((X, h, kernel))
