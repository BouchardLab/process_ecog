from __future__ import division
import multiprocessing
import numpy as np
from scipy.signal import firwin2, filtfilt

try:
    from accelerate.mkl.fftpack import rfft, irfft, rfftfreq
except ImportError:
    try:
        from pyfftw.interfaces.numpy_fft import fft, ifft, fftfreq
    except ImportError:
        from numpy.fft import fft, ifft, fftfreq


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x

__authors__ = "Alex Bujan"


def apply_notches(args, fft=True):
    X, notches, rate = args
    if fft:
        fs = rfftfreq(X.shape[-1], 1./rate)
        delta = 1.
    else:
        nyquist = rate/2.
        n_taps = 1001
        gain = [1, 1, 0, 0, 1, 1]
    for notch in tqdm(notches, 'applying notch filters'):
        if fft:
            window_mask = np.logical_and(fs > notch-delta, fs < notch+delta)
            window_size = window_mask.sum()
            window = np.hamming(window_size)
            fd = rfft(X)
            fd[window_mask] = fd[window_mask] * (1.-window)
            X = irfft(fd)
        else:
            freq = np.array([0, notch-1, notch-.5,
                             notch+.5, notch+1, nyquist]) / nyquist
            filt = firwin2(n_taps, freq, gain)
            X = filtfilt(filt, np.array([1]), X)
    return X

def apply_linenoise_notch(X, rate, parallel=True):
    """
    Apply Notch filter at 60 Hz and its harmonics
    
    Parameters
    ----------
    X : array
        Input data, dimensions (n_channels, n_timePoints)
    rate : float
        Number of samples per second

    Returns
    -------
    X : array
        Denoised data, dimensions (n_channels, n_timePoints)
    """

    nyquist = rate/2
    noise_hz   = 60.
    notches = np.arange(noise_hz, nyquist, noise_hz)
    n_channels, time = X.shape

    if parallel:
        pool = multiprocessing.Pool()
        result = pool.map(apply_notches, [(c, notches, rate) for c in X])
        pool.close()
        return np.vstack([r[np.newaxis, :] for r in result])
    else:
        return apply_notches((X, notches, rate))
