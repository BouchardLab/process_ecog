import os

import numpy as np

from scipy.io import loadmat

from ecog import utils


def load_baseline(block_path, data, tt_data):
    baseline_file = os.path.join(block_path, 'baselineTime.mat')
    d = loadmat(baseline_file)
    baseline_time = d['baselineTime'].astype(float)[0]
    data_time = utils.is_in(tt_data, baseline_time)
    return data[..., data_time]


def zscore(data, axis=-1, mode=None, sampling_freq=None, bad_times=None,
           align_window=None, all_event_times=None, **kwargs):

    if mode is None:
        mode = 'file'

    if mode == 'whole':
        baseline = data
    elif mode == 'between_data':
        tt_data = np.arange(data.shape[axis]) / sampling_freq
        data_start = all_event_times.min() + align_window[0]
        data_stop = all_event_times.max() + align_window[1]
        data_time = utils.is_in(tt_data, np.array([data_start, data_stop]))
        for bt in bad_times:
            data_time = data_time & ~utils.is_in(tt_data, bt)
        for et in all_event_times:
            data_time = data_time & ~utils.is_in(tt_data, et + align_window)
        baseline = data[..., data_time]
    elif mode == 'data':
        tt_data = np.arange(data.shape[-1]) / sampling_freq
        data_start = all_event_times.min() + align_window[0]
        data_stop = all_event_times.max() + align_window[1]
        data_time = utils.is_in(tt_data, np.array([data_start, data_stop]))
        for bt in bad_times:
            data_time = data_time & ~utils.is_in(tt_data, bt)
        baseline = data[..., data_time]
    elif mode == 'events':
        tt_data = np.arange(data.shape[-1]) / sampling_freq
        data_time = np.zeros_like(tt_data, dtype=bool)
        for et in all_event_times:
            data_time = data_time | utils.is_in(tt_data, et + align_window)
        for bt in bad_times:
            data_time = data_time & ~utils.is_in(tt_data, bt)
        baseline = data[..., data_time]
    elif mode == 'file':
        tt_data = np.arange(data.shape[axis]) / sampling_freq
        baseline = load_baseline(kwargs['block_path'], data, tt_data)
    elif mode == 'ratio':
        tt_data = np.arange(data.shape[axis]) / sampling_freq
        baseline = load_baseline(kwargs['block_path'], data, tt_data)
        means = baseline.mean(axis=axis, keepdims=True)
        stds = baseline.std(axis=axis, keepdims=True)
        data = data / means
        return data, means, stds, baseline
    elif mode == 'none':
        tt_data = np.arange(data.shape[axis]) / sampling_freq
        baseline = load_baseline(kwargs['block_path'], data, tt_data)
        means = baseline.mean(axis=axis, keepdims=True)
        stds = baseline.std(axis=axis, keepdims=True)
        return data, means, stds, baseline
    else:
        raise ValueError('zscore_mode type {} not recognized.'.format(mode))

    means = baseline.mean(axis=axis, keepdims=True)
    stds = baseline.std(axis=axis, keepdims=True)
    data = (data - means) / stds

    return data, means, stds, baseline
