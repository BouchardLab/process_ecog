from __future__ import division
import numpy as np
from pyfftw.interfaces.numpy_fft import fft, ifft, fftfreq

__authors__ = "Alex Bujan, Jesse Livezey"


def gaussian(X, rate, center, sd):
    time_samples = X.shape[-1]
    freq = fftfreq(time_samples, 1./rate)

    k  = np.exp((-(np.abs(freq)-center)**2)/(2*(sd**2)))

    return k

def hamming(X, rate, min_freq, max_freq):
    time_samples = X.shape[-1]
    freq = fftfreq(time_samples, 1./rate)

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


def apply_hilbert_transform(X, rate, kernel=None):
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

    time_samples = X.shape[-1]
    freq = fftfreq(time_samples, 1./rate)

    # heaviside kernel
    h = np.zeros(len(freq))
    h[freq > 0]=2.
    h[0]=1.

    # compute analytical signal
    if kernel is None:
        Xc = ifft(fft(X)*h)
    else:
        Xc = ifft(fft(X)*h*kernel)

    return Xc
