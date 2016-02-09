#!/usr/bin/env python
__author__ = 'David Conant, Jesse Livezey'

import argparse, h5py, multiprocessing re, os, glob, csv
import numpy as np
import scipy as sp
import scipy.stats as stats
from scipy.io import loadmat
import pandas as pd

from utils import HTK, transcripts


def htk_to_hdf5(path, blocks, output_folder=None, task='CV',
                align_window=None, align_pos = 0,
                data_type='HG', zscore='whole'):
    """
    Process task data into segments with labels.

    Parameters
    ----------
    path : str
        Path to subject folder.
    blocks : list of ints
        Blocks from which data is gathered.
    task : str
        Type of tokens to be extracted.
    align_window : list of two ints
        Time window in seconds around each token.
    align_pos : ints
        Align to start of this phoneme.
    data_type : str
        Type of data to use.

    Returns
    -------
    D : dict
        Dictionary containing tokens as keys and data as array.
    anat : str
        Anatomy of data channels
    start_times : dict
        Dictionary of start times per token.
    stop_times : dict
        Dictionary of stop times per token.
    """

    if output_folder is None:
        output_folder = path

    if task == 'CV':
        tokens = sorted(['baa', 'bee', 'boo', 'daa', 'dee', 'doo', 'faa', 'fee', 'foo',
                         'gaa', 'gee', 'goo', 'haa', 'hee', 'hoo', 'kaa', 'kee', 'koo',
                         'laa', 'lee', 'loo', 'maa', 'mee', 'moo', 'naa', 'nee', 'noo',
                         'paa', 'pee', 'poo', 'raa', 'ree', 'roo', 'saa', 'shaa', 'shee',
                         'shoo', 'see', 'soo', 'taa', 'thaa', 'thee', 'thoo', 'tee',
                         'too', 'vaa', 'vee', 'voo', 'waa', 'wee', 'woo','yaa','yee',
                         'yoo', 'zaa', 'zee', 'zoo'])
    else:
        raise ValueError("task must of one of ['CV']: "+str(task)+'.')

    if data_type not in ['HG']:
        raise ValueError("data_type must be one of ['HG']: "+str(data_type)+'.')

    if align_window is None:
        align_window = np.array([-1., 1.])
    else:
        align_window = np.array(align_window)
        assert align_window[0] <= 0.
        assert align_window[1] >= 0.
        assert align_window[0] < align_window[1]

    def block_str(blocks):
        rval = 'blocks_'
        for block in blocks:
            rval += str(block) + '_'
        return rval

    def align_window_str(align_window):
        rval = 'align_window_' + str(align_window[0]) + '_to_'+ str(align_window[1])
        return rval

    folder, subject = os.path.split(os.path.normpath(path))
    fname = os.path.join(output_folder, 'hdf5', (subject + '_' + block_str(blocks)
                                       + task + '_' + data_type + '_'
                                       + align_window_str(align_window) + '.h5'))

    D = dict((token, np.array([])) for token in tokens)
    stop_times = dict((token, np.array([])) for token in tokens)
    start_times = dict((token, np.array([])) for token in tokens)

    pool = multiprocessing.Pool()
    args = [(subject, block, path, tokens) for block in blocks]
    results = pool.map(process_block, args)
    for Bstart, Bstop, BD in results:
        for token in tokens:
            start_times[token] = (np.vstack((start_times[token], Bstart)) if
                                  start_times[token].size else Bstart)
            stop_times[token] = (np.vstack((stop_times[token], Bstop)) if
                                 stop_times[token].size else Bstop)
            D[token] = (np.vstack((D[token], BD[token])) if
                        D[token].size else BD[token])

    print('Saving to: '+fname)
    save_hdf5(fname, D, tokens)

    anat = load_anatomy(path)
    return (D, anat, start_times, stop_times)

def process_block(subject, block, path, tokens):
    """
    Process a single block.

    Parameters
    ----------
    subject : str
    block : int
    path : str
    tokens : list of str
    """

    print 'Processing block ' + str(block)
    blockname = subject + '_B' + str(block)
    blockpath = os.path.join(path, blockname)
    # Convert parseout to dataframe
    parseout = transcripts.parse(blockpath, blockname)
    df = make_df(parseout, block, subject, align_pos)

    D = dict((token, np.array([])) for token in tokens)
    stop_times = dict((token, np.array([])) for token in tokens)
    start_times = dict((token, np.array([])) for token in tokens)

    for ind, token in enumerate(tokens):
        match = [token in t for t in df['label']]
        event_times = df['align'][match & (df['mode'] == 'speak')]
        start = event_times.values + align_window[0]
        stop = event_times.values + align_window[1]

        start_times[token] = (np.vstack((start_times[token], start.astype(float))) if
                              start_times[token].size else start.astype(float))
        stop_times[token] = (np.vstack((stop_times[token], stop.astype(float))) if
                             stop_times[token].size else stop.astype(float))
        D[token] = (np.vstack((D[token], run_makeD(blockpath, event_times,
                                                   align_window, data_type=data_type))) if
                    D[token].size else run_makeD(blockpath, event_times, align_window, data_type=data_type))

    return (start_times, stop_times, D)

def save_hdf5(fname, D, tokens):
    """
    Save processed data to hdf5.

    Parameters
    ----------
    fname : str
        Path to save output.
    D : dict
        Dictionary containing data for each token.
    tokens : list of str
        Tokens to save from D.
    """
    tokens = sorted(tokens)
    labels = np.array(range(len(tokens)))
    X = None
    y = None
    for label, token in zip(labels, tokens):
        X_t = np.transpose(D[token], axes=(2, 0, 1))
        if X is None:
            X = X_t
        else:
            X = np.append(X, X_t, axis=0)
        if y is None:
            y = label * np.ones(X_t.shape[0], dtype=int)
        else:
            y = np.append(y, label * np.ones(X_t.shape[0], dtype=int))
    folder, f = os.path.split(fname)

    try:
        os.mkdir(folder)
    except OSError:
        pass

    with h5py.File(fname, 'w') as f:
        f.create_dataset('X', data=X.astype('float32'))
        f.create_dataset('y', data=y)
        f.create_dataset('tokens', data=tokens)


def make_df(parseout, block, subject, align_pos, tier='word'):
    """
    Organize event data.

    Parameters
    ----------
    parseout : dict
        Dictionary from parsed transcript.
    block : int
        Block ID.
    subject : str
        Subject ID.
    align_pos : int
        Subelement in event to align to.
    tier : str
        Type of event to extract.
    """

    keys = sorted(parseout.keys())
    datamat = [parseout[key] for key in keys]
    df = pd.DataFrame(np.vstack(datamat).T, columns=keys)
    first_phones_per_word = df[df['tier'] == tier]['contains'].apply(lambda x: x[align_pos])
    df_events = df[df['tier'] == tier]

    # Get rid of superfluous columns
    df_events = df_events[['label','start', 'stop']]
    df_events['align'] = df['start'].iloc[first_phones_per_word].astype(float).values
    assert np.all(df_events['align'] >= df_events['start'])
    assert np.all(df_events['align'] <= df_events['stop'])

    # Pull mode from label and get rid of number
    df_events['mode'] = ['speak' if l[-1] == '2' else 'listen' for l in df_events['label']]
    df_events['label'] = df_events['label'].apply(lambda x: x[:-1])

    df_events['label'] = df_events['label'].astype('category')
    df_events['mode'] = df_events['mode'].astype('category')
    df_events['block'] = block
    df_events['subject'] = subject

    return df_events


def run_makeD(blockpath, event_times, align_window, data_type, zscore='whole'):
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
            tt_data = np.arange(hg.shape[1])/fs_hg
            data_start = event_times.min() + align_window[0]
            data_stop = event_times.max() + align_window[1]
            data = hg[:, isin(tt_data, np.array([data_start, data_stop]))]
            means = data.mean(axis=1, keepdims=True)
            stds = data.std(axis=1, keepdims=True)
            hg = (hg - means)/stds


        D = makeD(hg, fs_hg, event_times, align_window,
                  bad_times=bad_times, bad_electrodes=bad_electrodes)

        return D

    def form():
        F = load_form(blockpath)
        D = makeD(F, 100, times, align_window, bad_times=np.array([]), bad_electrodes=np.array([]))

        return D

    options = {'HG' : HG,
               'form' : form}

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

    D = nans((len(event_times), np.ceil(np.diff(align_window) * fs_data), data.shape[0]))
    tt_data = np.arange(data.shape[1]) / fs_data

    for ievent, time in enumerate(event_times):
        event_data = data[:, isin(tt_data, align_window + time)].T
        D[ievent] = event_data

    if bad_times.any():
        good_trials = [not np.any(np.logical_and(bad_times,
            np.any(is_overlap(align_window + time, bad_times))))
            for time in event_times]
        D = D[good_trials]

    if len(bad_electrodes):
        bad_electrodes = bad_electrodes[bad_electrodes < D.shape[3]]
        D[:, :, bad_electrodes] = np.nan

    return D

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
    """"

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
    anatomy_filename = glob.glob(os.path.join(subj_dir, '*_anat.mat'))
    elect_labels_filename = glob.glob(os.path.join(subj_dir, 'elec_labels.mat'))

    if anatomy_filename:
        anatomy = sp.io.loadmat(anatomy_filename[0])
        electrode_labels = np.array([item[0][0] if len(item[0])
                                     else '' for item in anatomy['electrodes'][0]])

    elif elect_labels_filename:
        a = sp.io.loadmat(os.path.join(subj_dir, 'elec_labels.mat'))
        electrode_labels = np.array([ elem[0] for elem in a['labels'][0]])

    else:
        electrode_labels = ''

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
        bes = csv.reader(f, delimiter=' ')
        for be in bes:
            if be != '':
                bad_electrodes.append(be)

    bad_electrodes = np.array([int(be)-1 for be in bad_electrodes])

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

    bad_time = np.array(bad_times)

    return bad_electrodes

def nans(shape, dtype=float):
    """
    Create np.array of nans

    :param shape: tuple, dimensions of array
    :param dtype:
    :return:
    """
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a

def is_overlap(time_window, time_windows_array):
    """
    Does time_window overlap with the time windows in times_window_array.
    Used for bad time segments.

    Parameters
    ----------
    time_window : ndarray (2,)
       Single window of time to compare.
    time_windows_array : ndarray (n, 2)
       Windows of time to compare against.

    Returns
    -------
    Boolean overlap comparing time_window to each window in time_windows_array.

    """
    def overlap(tw1, tw2):
        return not ((tw1[1] < tw2[0]) | (tw1[0] > tw2[1]))

    return [overlap(time_window,this_time_window) for this_time_window in time_windows_array]

def isin(tt, tbounds):
    """
    util: Is time inside time window(s)?

    :param tt:      1 x n np.array        time counter
    :param tbounds: k, 2  np.array   time windows

    :return:        1 x n bool          logical indicating if time is in any of the windows
    """

    #check if tbounds in np.array and if not fix it
    tbounds = np.array(tbounds)

    tf = np.zeros(tt.shape, dtype = 'bool')

    if tbounds.ndim is 1:
        tf = (tt > tbounds[0]) & (tt < tbounds[1])
    else:
        for i in range(tbounds.shape[0]):
            tf = (tf | (tt > tbounds[i,0]) & (tt < tbounds[i,1]))
    return tf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess ECoG Data')
    parser.add_argument('path', help='path to subject folder')
    parser.add_argument('blocks', nargs='+', type=int)
    parser.add_argument('-o', '--output_folder', default=None)
    parser.add_argument('-t', '--task', default='CV')
    parser.add_argument('-w', '--align_window', nargs=2, type=float, default=[-.5, .79])
    parser.add_argument('-p', '--align_pos', type=int, default=1)
    parser.add_argument('-d', '--data_type', default='HG')
    parser.add_argument('-b', '--zscore', default='whole')
    args = parser.parse_args()
    htk_to_hdf5(args.path, args.blocks, args.output_folder, args.task,
                args.align_window, args.align_pos, args.data_type, args.zscore)
