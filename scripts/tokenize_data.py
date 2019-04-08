#!/usr/bin/env python
from __future__ import print_function, division
import argparse, h5py, multiprocessing, os
import numpy as np

from ecog.tokenize import transcripts, make_data
from ecog.utils import bands, load_anatomy

import nwbext_ecog
from pynwb import NWBHDF5IO


__author__ = 'David Conant, Jesse Livezey'


def main():
    parser = argparse.ArgumentParser(description='Tokenize ECoG Data')
    parser.add_argument('path', help='path to root folder')
    parser.add_argument('subject', type=str, help="Subject code")
    parser.add_argument('blocks', nargs='+', type=str)
    parser.add_argument('-t', '--task', default='CV')
    parser.add_argument('-w', '--align_window', nargs=2, type=float,
                        default=[-.5, .79])
    parser.add_argument('-p', '--align_pos', type=int, default=1)
    parser.add_argument('-d', '--data_type', default='AA_avg')
    parser.add_argument('-b', '--zscore', default='silence')
    parser.add_argument('-m', '--mp', action='store_true', default=False)
    parser.add_argument('-e', '--phase', action='store_true', default=False)
    parser.add_argument('-f', '--fband', type=int, default=None)
    args = parser.parse_args()
    subject_path = os.path.join(args.path, args.subject)
    tokenize(subject_path, args.blocks, args.task,
             args.align_window, args.align_pos, args.data_type, args.zscore,
             args.fband, args.mp, args.phase)


def tokenize(subject_path, blocks, task='CV',
             align_window=None, align_pos=1,
             data_type='AA_avg', zscore='silence',
             fband=None, mp=True, phase=False):
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
    align_pos : int
        Align to start of this phoneme.
    data_type : str
        Type of data to use.

    Returns
    -------
    D : dict
        Dictionary containing tokens as keys and data as array.
    anat : str
        Anatomy of data channels
    """

    output_folder = subject_path

    tasks = ['CV']
    if task == 'CV':
        tokens = sorted(['baa', 'bee', 'boo', 'daa', 'dee', 'doo', 'faa',
                         'fee', 'foo', 'gaa', 'gee', 'goo', 'haa', 'hee',
                         'hoo', 'kaa', 'kee', 'koo', 'laa', 'lee', 'loo',
                         'maa', 'mee', 'moo', 'naa', 'nee', 'noo', 'paa',
                         'pee', 'poo', 'raa', 'ree', 'roo', 'saa', 'shaa',
                         'shee', 'shoo', 'see', 'soo', 'taa', 'thaa', 'thee',
                         'thoo', 'tee', 'too', 'vaa', 'vee', 'voo', 'waa',
                         'wee', 'woo', 'yaa', 'yee', 'yoo', 'zaa', 'zee',
                         'zoo'])
    else:
        raise ValueError('Task must of one of {}: {}'.format(tasks, task))

    data_types = ['AA', 'AA_avg', 'AA_ff']
    if data_type not in data_types:
        raise ValueError('Data_type must be one of {}: {}'.format(data_types,
                                                                  data_type))

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
        rval = 'align_window_{}_to_{}'.format(align_window[0], align_window[1])
        return rval

    folder, subject = os.path.split(os.path.normpath(subject_path))

    phase_str = ''
    if phase:
        phase_str = '_random_phase'

    if fband is None:
        fname = os.path.join(output_folder,
                             ('{}_{}{}_{}_{}_{}{}.h5'.format(subject,
                                                            block_str(blocks),
                                                            task,
                                                            data_type,
                                                            align_window_str(align_window),
                                                            zscore,
                                                            phase_str)))
    else:
        fname = os.path.join(output_folder,
                             (subject + '_' + block_str(blocks) +
                              task + '_' + data_type + '_' +
                              str(fband) + '_' +
                              align_window_str(align_window) + '_' +
                              zscore + '.h5'))

    blocks = [int(block) for block in blocks]

    block_path = os.path.join(subject_path, '{}_B{}.nwb'.format(subject, blocks[0]))
    with NWBHDF5IO(block_path, 'r') as io:
        nwb = io.read()
        anat = load_anatomy(nwb)
    for bi in blocks[1:]:
        with NWBHDF5IO(block_path, 'r') as io:
            nwb = io.read()
            anat_b = load_anatomy(nwb)
        if not np.all(np.equal(anat_b, anat)):
            raise ValueError('Block {} has different anatomy.'.format(bi))

    args = [(subject, block, folder, tokens, align_pos,
             align_window, data_type, zscore, fband, phase) for block in blocks]
    print('Numbers of blocks to be processed: {}'.format(len(blocks)))

    if mp and len(blocks) > 1:
        pool = multiprocessing.Pool(len(blocks))
        print('Processing blocks in parallel ' +
              'with {} processes...'.format(pool._processes))
        results = list(pool.map(process_block, args))
    else:
        print('Processing blocks serially ...')
        results = list(map(process_block, args))

    band_ids = results[0][0]
    block_fs = results[0][4]
    for r in results[1:]:
        b_ids = r[0]
        assert len(band_ids) == len(b_ids)
        assert set(band_ids) == set(b_ids)
        b_fs = r[4]
        assert len(block_fs) == len(b_fs)
        assert set(block_fs) == set(b_fs)

    labels = dict((b_id, np.array([], dtype=int)) for b_id in band_ids)
    block_numbers = dict((b_id, np.array([], dtype=int)) for b_id in band_ids)

    n_trials = 0
    shapes = dict()
    dtype = results[0][1][band_ids[0]].dtype

    for _, d, l, n, _, bl in results:
        n_trials += d[band_ids[0]].shape[0]
        for b_id in band_ids:
            shapes[b_id] = d[b_id].shape[1:]

    data = dict((b_id, np.zeros((n_trials,) + shapes[b_id],
                                dtype=dtype)) for b_id in band_ids)

    bls = dict()
    idx = 0
    for _, d, l, n, _, bl in results:
        bls_b = dict()
        for b_id in band_ids:
            bls_b[b_id] = bl[b_id]
            data[b_id][idx:idx + d[b_id].shape[0]] = d[b_id]

            if labels[b_id].size == 0:
                labels[b_id] = l
            else:
                labels[b_id] = np.hstack((labels[b_id], l))

            if block_numbers[b_id].size == 0:
                block_numbers[b_id] = n
            else:
                block_numbers[b_id] = np.hstack((block_numbers[b_id], n))
        idx += d[b_id].shape[0]
        bls[n[0]] = bls_b

    test_label = labels[band_ids[0]]
    test_block = block_numbers[band_ids[0]]
    for b in band_ids[1:]:
        assert np.all([ti == li for ti, li in zip(test_label, labels[b])])
        assert np.allclose(test_block, block_numbers[b])
    labels = test_label
    block_numbers = test_block

    print('Saving to: {}'.format(fname))
    save_hdf5(fname, data, labels, tokens, block_numbers, block_fs,
              anat, data_type, bls)


def process_block(args):
    """
    Process a single block.

    Parameters
    ----------
    subject : str
    block : int
    path : str
    tokens : list of str
    """
    (subject, block, path, tokens, align_pos, align_window,
     data_type, zscore, fband, phase) = args

    blockname = '{}_B{}.nwb'.format(subject, block)
    print('Processing subject {}'.format(subject))
    print('-----------------------')
    block_path = os.path.join(path, subject, blockname)
    with NWBHDF5IO(block_path, 'r') as io:
        nwb = io.read()
        event_times, event_labels = transcripts.get_speak_event(nwb, align_pos)

    idx = np.argsort(event_times)
    event_times = event_times[idx]
    event_labels = event_labels[idx]
    event_labels = np.array([tokens.index(li) for li in event_labels],
                            dtype=int)

    rval = make_data.run_extract_windows(block_path, event_times,
                                         align_window, data_type,
                                         zscore, fband, phase)
    band_ids, data, fs, bl = rval
    print(band_ids)

    for k, v in data.items():
        assert v.shape[0] == event_labels.shape[0], ('shapes', k, v.shape,
                                                     event_labels.shape)
    bn = np.full(list(data.values())[0].shape[0], block, dtype=int)

    return band_ids, data, event_labels, bn, fs, bl


def save_hdf5(fname, data, labels, tokens, block_numbers, block_fs,
              anat, data_type, baselines):
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
    folder, f = os.path.split(fname)

    try:
        os.mkdir(folder)
    except OSError:
        pass
    fname_tmp = fname + '.tmp'
    band_ids = sorted(data.keys())
    block_fs = np.array([block_fs[b] for b in band_ids], dtype=float)
    with h5py.File(fname_tmp, 'w') as f:
        if data_type in ['AA_avg', 'AA_ff', 'AA']:
            for b, d in data.items():
                dset = f.create_dataset('X{}'.format(b), data=d)
                dset.dims[0].label = 'batch'
                dset.dims[1].label = 'electrode'
                dset.dims[2].label = 'time'
            for n, block_bls in baselines.items():
                for b, bl in block_bls.items():
                    dset = f.create_dataset('bl_block_{}_band_{}'.format(n, b),
                                            data=bl)
            if data_type == 'AA_avg':
                min_freqs = bands.neuro['min_freqs']
                max_freqs = bands.neuro['max_freqs']
                f.create_dataset('min_freqs', data=np.array(min_freqs))
                f.create_dataset('max_freqs', data=np.array(max_freqs))
        else:
            raise ValueError

        f.create_dataset('y', data=labels)
        f.create_dataset('block', data=block_numbers)
        f.create_dataset('tokens', data=[t.encode('utf8') for t in tokens])
        f.create_dataset('sampling_freqs', data=np.array(block_fs))
        f.create_dataset('anatomy', data=[t.encode('utf8') for t in anat])

    os.rename(fname_tmp, fname)


if __name__ == "__main__":
    main()
