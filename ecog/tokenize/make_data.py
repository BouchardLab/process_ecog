from __future__ import print_function, division
import csv, glob, h5py, os, time

import numpy as np
from scipy.io import loadmat

from ecog import utils
from ..signal_processing import resample, zscore, load_silence_time
from ..utils import (bands, load_bad_electrodes, load_bad_times)

from pynwb import NWBHDF5IO

__author__ = 'David Conant, Jesse Livezey'


def run_extract_windows(block_path, event_times, align_window,
                        data_type, zscore_mode='file', fband=None, phase=False):
    """
    Extract requested data type.

    Parameters
    ----------
    block_path : str
        Path to block.
    event_times : list of float
        Event alignment times.
    align_window : ndarray
        Window around event alignment.
    data_type : str
        Data type (e.g. 'HG', 'form').
    zscore_mode : str
        Method for zscoring data.
    """
    print(block_path)

    with NWBHDF5IO(block_path, 'r') as io:
        nwb = io.read()
        bad_electrodes = load_bad_electrodes(nwb)
        bad_times = load_bad_times(nwb)
        silence_time = load_silence_time(nwb)

    chang_lab_cfs = bands.chang_lab['cfs']
    min_freqs = bands.neuro['min_freqs']
    max_freqs = bands.neuro['max_freqs']
    HG_freq = bands.neuro['HG_freq']

    D = dict()
    final_fs = dict()
    bls = dict()

    if data_type == 'AA_avg':
        nbands = bands.neuro['bands']
        HG_freq = bands.neuro['HG_freq']

        for b, minf, maxf in zip(nbands, min_freqs, max_freqs):
            target_fs = (HG_freq * (minf + maxf) /
                         (max_freqs[-1] + min_freqs[-1]))
            start = time.time()
            idxs = np.nonzero((chang_lab_cfs > minf) *
                              (chang_lab_cfs < maxf))[0].tolist()
            X, fs = load_AA_band(block_path, idxs, target_fs)

            print('load', block_path, b, target_fs, time.time()-start)
            start = time.time()
            X, m, s, bl = zscore(X, sampling_freq=fs, bad_times=bad_times,
                             align_window=align_window, mode=zscore_mode,
                             event_times=event_times,
                             silence_time=silence_time)
            X = X.mean(axis=0)
            print('zscore', block_path, target_fs, time.time()-start)
            D[b] = extract_windows(X, target_fs, event_times, align_window,
                                   bad_times=bad_times,
                                   bad_electrodes=bad_electrodes)
            final_fs[b] = target_fs
            bls[b] = bl
    elif data_type == 'AA_ff':
        nbands = np.arange(40)
        target_fs = HG_freq

        for ii in nbands:
            start = time.time()
            idxs = [ii]
            X, fs = load_AA_band(block_path, idxs, target_fs, phase)

            print('load', block_path, ii, target_fs, time.time()-start)
            start = time.time()
            X, m, s, bl = zscore(X, sampling_freq=fs, bad_times=bad_times,
                             align_window=align_window, mode=zscore_mode,
                             event_times=event_times,
                             silence_time=silence_time)
            X = X.mean(axis=0)
            print('zscore', block_path, target_fs, time.time()-start)
            D[ii] = extract_windows(X, target_fs, event_times, align_window,
                                    bad_times=bad_times,
                                    bad_electrodes=bad_electrodes)
            final_fs[ii] = target_fs
            bls[ii] = bl
    elif data_type == 'AA':
        nbands = [fband]
        target_fs = (HG_freq * 2. * chang_lab_cfs[fband] /
                     (max_freqs[-1] + min_freqs[-1]))
        start = time.time()
        X, fs = load_AA_band(block_path, fband, target_fs)
        print('load', block_path, fband, target_fs, time.time()-start)
        start = time.time()
        X, m, s, bl = zscore(X, sampling_freq=target_fs, bad_times=bad_times,
                         align_window=align_window, mode=zscore_mode,
                         event_times=event_times,
                         silence_time=silence_time)
        X = X.mean(axis=0)
        print('zscore', block_path, fband, target_fs, time.time()-start)
        D[fband] = extract_windows(X, target_fs, event_times, align_window,
                                   bad_times=bad_times,
                                   bad_electrodes=bad_electrodes)
        final_fs[fband] = target_fs
        bls[fband] = bl
    else:
        raise ValueError

    return nbands, D, final_fs, bls


def extract_windows(data, sampling_freq, event_times, align_window=None,
                    bad_times=None, bad_electrodes=None):
    """
    Extracts windows of aligned data. Assumes constant sampling frequency.
    Assumes last two dimensions of data are electrodes and time.

    Parameters
    ----------
    data : ndarray (n_channels, n_time)
        Timeseries data.
    sampling_freq : float
        Sampling frequency of data.
    event_time : list of floats
        Time (in seconds) of events.
    align_window : ndarray
        Window around event alignment.
    bad_times : ndarray (n_windows, 2)
        Start and stop points of bad times segments.
    bad_electrodes : ndarray (n_electrodes,)
        Indices of bad electrodes.

    Returns
    -------
    D : ndarray (n_events, *n_bands*, n_elects, n_time)
        Event data aligned to event times.
    """
    dtype = data.dtype
    assert data.ndim == 2, 'Data dim is wrong'

    if align_window is None:
        align_window = np.array([-1., 1.])
    else:
        align_window = np.array(align_window)
        assert align_window[0] <= 0., 'align window start'
        assert align_window[1] >= 0., 'align window end'
        assert align_window[0] < align_window[1], 'align window order'

    if bad_times is None:
        bad_times = np.array([])
    else:
        bad_times = np.array(bad_times)
    if bad_electrodes is None:
        bad_electrodes = np.array([])
    else:
        bad_electrodes = np.array(bad_electrodes)

    n_channels = data.shape[0]
    n_data = len(event_times)
    window_length = int(np.ceil(np.diff(align_window) * sampling_freq))
    window_start = int(np.floor(align_window[0] * sampling_freq))
    D = utils.nans((n_data, n_channels, window_length,), dtype=dtype)

    def time_idx(time):
        return int(np.around(time * sampling_freq))

    for ii, etime in enumerate(event_times):
        sl = slice(time_idx(etime) + window_start,
                   time_idx(etime) + window_start + window_length)
        event_data = data[..., sl]
        if event_data.shape == D.shape[1:]:
            D[ii] = event_data
        else:
            raise ValueError('Event shape mismatch ' +
                             '{}, {}'.format(event_data.shape, D.shape))

    if bad_times.size:
        bad_trials = []
        for ii, etime in enumerate(event_times):
            if np.any(utils.is_overlap(align_window + etime,
                                       bad_times)):
                bad_trials.append(ii)
        if len(bad_trials) > 0:
            D[bad_trials] = np.nan

    if bad_electrodes.size:
        bad_electrodes = bad_electrodes[bad_electrodes < D.shape[-1]]
        D[:, bad_electrodes, :] = np.nan

    return D


def load_AA_band(block_path, fbands, target_fs=None, phase=False):
    """
    Reads in analytic amplitude data.

    Parameters
    ----------
    block_path : str
        Path to block.
    fbands : int or list of ints
        Indices of Chang lab standard bands (0 through 40).

    Returns
    -------
    X : ndarray (n_channels, n_time)
        ECoG data.
    fs : float
        Sampling frequency of data.
    """
    folder, nwb = os.path.split(block_path)
    nwb_base = os.path.splitext(nwb)[0]
    if not isinstance(fbands, list):
        fbands = [fbands]

    Xs = []
    start1 = time.time()
    phase_str = ''
    if phase:
        phase_str = '_random_phase'
    ecog_hilb_path = os.path.join(folder,
                                  '{}_AA{}.h5'.format(nwb_base, phase_str))
    for fband in fbands:
        with h5py.File(ecog_hilb_path) as f:
            X = f['X'][fband]
            fs = f.attrs['sampling_rate']

        if target_fs is not None:
            if not np.allclose(target_fs, fs):
                assert target_fs < fs
                start = time.time()
                X = resample(X, target_fs, fs)
                print('downsample', block_path, target_fs, fs, time.time()-start)

            fs = target_fs
        Xs.append(X.astype('float32'))
    print('load inner', time.time()-start1)
    X = np.stack(Xs).astype('float32')

    return X, fs


def load_formant(block_path):
    fname = glob.glob(os.path.join(block_path, 'Analog',
                                   '*.ifc_out.txt'))
    F = []
    with open(fname[0]) as tsv:
        for column in zip(*[line for line in
                            csv.reader(tsv, dialect="excel-tab")]):
            F.append(column)
    F = np.array(F)
    return F
