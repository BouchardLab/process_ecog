__author__ = 'David Conant, Jesse Livezey'

import os, glob, csv, h5py
import numpy as np
import scipy as sp

import scipy.stats as stats
from scipy.io import loadmat

from scikits.samplerate import resample

import HTK, transcripts, utils


def run_makeD(blockpath, event_times, align_window, data_type, zscore='events',
              all_event_times=None):
    """
    Extract requested data type.

    Parameters
    ----------
    blockpath : str
        Path to block.
    event_times : list of float
        Event alignment times.
    align_window : ndarray
        Window around event alignment.
    data_type : str
        Data type (e.g. 'HG', 'form').
    zscore : str
        Method for zscoring data.
    """

    def HG():
        bad_electrodes = load_bad_electrodes(blockpath) -1
        bad_times = load_bad_times(blockpath)
        hg, fs_hg = load_HG(blockpath)

        hg = hg[:256]

        if zscore == 'whole':
            hg = stats.zscore(hg, axis=1)
        elif zscore == 'data':
            tt_data = np.arange(hg.shape[1]) / fs_hg
            data_start = all_event_times.min() + align_window[0]
            data_stop = all_event_times.max() + align_window[1]
            data_time = utils.isin(tt_data, np.array([data_start, data_stop]))
            for bt in bad_times:
                data_time = data_time & ~utils.isin(tt_data, bt)
            data = hg[:, data_time]
            means = data.mean(axis=1, keepdims=True)
            stds = data.std(axis=1, keepdims=True)
            hg = (hg - means)/stds
        elif zscore == 'events':
            tt_data = np.arange(hg.shape[1]) / fs_hg
            data_time = np.zeros_like(tt_data).astype(bool)
            for et in all_event_times:
                data_time = data_time | utils.isin(tt_data, et + align_window)
            for bt in bad_times:
                data_time = data_time & ~utils.isin(tt_data, bt)
            data = hg[:, data_time]
            means = data.mean(axis=1, keepdims=True)
            stds = data.std(axis=1, keepdims=True)
            hg = (hg - means) / stds
        elif ((zscore is None) or (zscore.lower() == 'none')):
            pass
        else:
            raise ValueError('zscore type {} not recognized.'.format(zscore))


        D = makeD(hg, fs_hg, event_times, align_window,
                  bad_times=bad_times, bad_electrodes=bad_electrodes)

        return D

    def form():
        F = load_form(blockpath)
        D = makeD(F, 100, times, align_window, bad_times=np.array([]), bad_electrodes=np.array([]))

        return D


    def AS():
        bad_electrodes = load_bad_electrodes(blockpath) -1
        bad_times = load_bad_times(blockpath)
        s, fs = load_AS(blockpath)

        if zscore == 'whole':
            s = stats.zscore(s, axis=1)
        elif zscore == 'data':
            tt_data = np.arange(s.shape[1]) / fs
            data_start = all_event_times.min() + align_window[0]
            data_stop = all_event_times.max() + align_window[1]
            data_time = utils.isin(tt_data, np.array([data_start, data_stop]))
            for bt in bad_times:
                data_time = data_time & ~utils.isin(tt_data, bt)
            data = s[:, data_time]
            means = data.mean(axis=1, keepdims=True)
            stds = data.std(axis=1, keepdims=True)
            s = (s - means)/stds
        elif zscore == 'events':
            tt_data = np.arange(s.shape[1]) / fs
            data_time = np.zeros_like(tt_data).astype(bool)
            for et in all_event_times:
                data_time = data_time | utils.isin(tt_data, et + align_window)
            for bt in bad_times:
                data_time = data_time & ~utils.isin(tt_data, bt)
            data = s[:, data_time]
            means = data.mean(axis=1, keepdims=True)
            stds = data.std(axis=1, keepdims=True)
            s = (s - means) / stds
        elif ((zscore is None) or (zscore.lower() == 'none')):
            pass
        else:
            raise ValueError('zscore type {} not recognized.'.format(zscore))


        D = makeD(s, fs, event_times, align_window,
                  bad_times=bad_times, bad_electrodes=bad_electrodes)

        return D


    options = {'HG' : HG,
               'form' : form,
               'AS' : AS}

    D = options[data_type]()

    return D

def makeD(data, fs_data, event_times, align_window=None, bad_times=None, bad_electrodes=None):
    """
    Extracts windows of aligned data. Assumes constant sampling frequency

    Parameters
    ----------
    data : ndarray (n_channels, n_time)
        Timeseries data.
    fs_data : float
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
    D : ndarray (n_events, n_time, n_elects)
        Event data aligned to event times.
    """
    if align_window is None:
        align_window = np.array([-1., 1.])
    else:
        align_window = np.array(align_window)
        assert align_window[0] <= 0.
        assert align_window[1] >= 0.
        assert align_window[0] < align_window[1]

    window_length = int(np.ceil(np.diff(align_window) * fs_data))
    window_start = int(np.floor(align_window[0] * fs_data))
    D = utils.nans((len(event_times), window_length, data.shape[0]))
    tt_data = np.arange(data.shape[1]) / fs_data

    def time_idx(time):
        return int(np.around(time * fs_data))

    for ievent, time in enumerate(event_times):
        event_data = data[:, time_idx(time) + window_start:time_idx(time) + window_start + window_length].T
        assert event_data.shape[0] == D.shape[1]
        D[ievent] = event_data

    if bad_times.any():
        good_trials = [ii for ii, time in enumerate(event_times)
                       if not np.any(np.logical_and(bad_times,
                           np.any(utils.is_overlap(align_window + time, bad_times))))]
        D = D[good_trials]

    if len(bad_electrodes):
        bad_electrodes = bad_electrodes[bad_electrodes < D.shape[2]]
        D[:, :, bad_electrodes] = np.nan

    return D

def load_AS(blockpath,part='real',fbid=0):
    """
    Reads in HTK data.

    Parameters
    ----------
    blockpath : str
        Path to block.

    Returns
    -------
    hg : ndarray (n_channels, n_time)
        High gamma data.
    fs_hg : float
        Sampling frequency of data.
    """

    if part=='real':
        htk_path = os.path.join(blockpath, 'HilbReal_4to200_40band')
    elif part=='imag':
        htk_path = os.path.join(blockpath, 'HilbImag_4to200_40band')

    HTKout = HTK_hilb.readHTKs(htk_path)

    s = HTKout['data'][fbid]

    # Frequency in Hz
    fs = HTKout['sampling_rate']/1e4

    return (s, fs)


def load_HG(blockpath):
    """
    Reads in HTK data.

    Parameters
    ----------
    blockpath : str
        Path to block.

    Returns
    -------
    hg : ndarray (n_channels, n_time)
        High gamma data.
    fs_hg : float
        Sampling frequency of data.
    """

    htk_path = os.path.join(blockpath, 'HilbAA_70to150_8band')
    HTKout = HTK.readHTKs(htk_path)
    hg = HTKout['data']
    # Frequency in Hz
    fs_hg = HTKout['sampling_rate']/10000

    return (hg, fs_hg)

def loadForm(blockpath):
    fname = glob.glob(os.path.join(blockpath + 'Analog', '*.ifc_out.txt'))
    F = []
    with open(fname[0]) as tsv:
        for column in zip(*[line for line in csv.reader(tsv, dialect="excel-tab")]):
            F.append(column)
    F = np.array(F)
    return F

def load_anatomy(subj_dir):

    anatomy_filename = glob.glob(os.path.join(subj_dir, 'Anatomy', '*_anatomy.mat'))[0]
    elect_labels_filename = glob.glob(os.path.join(subj_dir, 'elec_labels.mat'))

    electrode_labels = dict()
    if anatomy_filename:
        try:
            anatomy = sp.io.loadmat(anatomy_filename)['anatomy']
            names = anatomy.dtype.names
            for n, labels in zip(names, anatomy[0][0]):
                electrode_labels[n] = np.array(labels[0])
        except ValueError:
            with h5py.File(anatomy_filename) as f:
                for n in f['anatomy'].keys():
                    electrode_labels[n] = f['anatomy'][n].value
    elif elect_labels_filename:
        raise NotImplementedError
        a = sp.io.loadmat(os.path.join(subj_dir, 'elec_labels.mat'))
        electrode_labels = np.array([ elem[0] for elem in a['labels'][0]])
    else:
        raise ValueError('Could not find anatomy file.')

    return electrode_labels

def load_bad_electrodes(blockpath):
    """
    Load bad electrodes.

    Parameters
    ----------
    blockpath : str
        Path to block to load bad electrodes from.

    Returns
    -------
    bad_electrodes : ndarray
        Indices of bad electrodes.
    """

    bad_electrodes = []
    with open(os.path.join(blockpath, 'Artifacts', 'badChannels.txt'),'rt') as f:
        lines = f.readlines()
        for bes in lines:
            for be in bes:
                if be != '':
                    try:
                        be = int(be)
                        bad_electrodes.append(be)
                    except ValueError:
                        pass

    bad_electrodes = np.array([be-1 for be in bad_electrodes])

    return bad_electrodes

def load_bad_times(blockpath):
    """
    Load bad time segments.

    Parameters
    ----------
    blockpath : str
        Path to block to load bad electrodes from.

    Returns
    -------
    bad_times : ndarray
        Pairs of start and stop times for bad segments.
    """

    bad_times = []
    try:
        lab_time_conversion = transcripts.lab_time_conversion
        with open(os.path.join(blockpath, 'Artifacts', 'bad_time_segments.lab'),'rt') as f:
            lines = f.readlines()
            for line in lines:
                if 'e' in line:
                    start, stop, _, _, _ = line.split(' ')
                    bad_times.append([float(start)/lab_time_conversion, float(stop)/lab_time_conversion])
    except IOError:
        return np.array([])

    bad_times = np.array(bad_times)

    return bad_times

def resample_data(X, ratio=.502):
    """
    Resample datapoints and channels.

    Parameteres
    -----------
    X : ndarray (n_batch, n_time, n_channels)
        Data array to resamples.
    ratio : float
        Ratio for data resampling. Defaults does 516 -> 258.

    Returns
    -------
    resampled_X : ndarray
        Resampled data array.
    """
    n_batch, n_time, n_channels = X.shape
    n_time, = resample(np.zeros(n_time), ratio, 'sinc_best').shape
    resampled_X = utils.nans((n_batch, n_time, n_channels))
    
    for ii in range(n_batch):
        for jj in range(n_channels):
            if np.isnan(X[ii, :, jj]).sum() == 0:
                resampled_X[ii, :, jj] = resample(X[ii, :, jj], ratio, 'sinc_best')
            else:
                pass

    return resampled_X
