from __future__ import print_function, division

import argparse, h5py, time, os
import numpy as np
from scipy.io import loadmat

try:
    from tqdm import tqdm
except:
    def tqdm(x, *args, **kwargs):
        return x

from ecog.signal_processing import resample
from ecog.signal_processing import subtract_CAR
from ecog.signal_processing import linenoise_notch
from ecog.signal_processing import hilbert_transform
from ecog.signal_processing import gaussian, hamming
from ecog.utils import HTK, load_bad_electrodes, bands

srf = HTK.SAMPLING_RATE_FACTOR


__authors__ = "Alex Bujan (adapted from Kris Bouchard)"


def main():

    parser = argparse.ArgumentParser(description='Preprocessing ecog data.')
    parser.add_argument('path', type=str, help="Path to the data")
    parser.add_argument('subject', type=str, help="Subject code")
    parser.add_argument('blocks', type=int, nargs='+',
                        help="Block number eg: '1'")
    parser.add_argument('-e', '--phase', default=False, action='store_true',
                        help="Apply random phases to the hilbert transform.")
    args = parser.parse_args()
    print(args)

    for block in args.blocks:
        block_path = os.path.join(args.path, args.subject,
                                  '{}_B{}'.format(args.subject, block))
        transform(block_path, phase=args.phase)


def transform(block_path, suffix=None, phase=False, total_channels=256,
              seed=20180928):
    """
    Takes raw LFP data and does the standard hilb algorithm:
    1) CAR
    2) notch filters
    3) Hilbert transform on different bands
    ...

    Saves to os.path.join(block_path, subject + '_B' + block + '_Hilb.h5')

    Parameters
    ----------
    block_path
    rate
    cfs: filter center frequencies. If None, use Chang lab defaults
    sds: filer standard deviations. If None, use Chang lab defaults

    takes about 20 minutes to run on 1 10-min block
    """

    rng = np.random.RandomState(seed)
    rate = 400.

    cfs = bands.chang_lab['cfs']
    sds = bands.chang_lab['cfs']

    subj_path, block_name = os.path.split(block_path)

    start = time.time()
    h5_ecog_path = os.path.join(block_path, 'ecog400', 'ecog.h5')
    h5_ecog_tmp_path = os.path.join(block_path, 'ecog400', 'ecog_tmp.h5')
    mat_ecog_path = os.path.join(block_path, 'ecog400', 'ecog.mat')
    try:
        raise IOError
        # HDF5 format
        with h5py.File(h5_ecog_path, 'r') as f:
            X = f['ecogDS']['data'].value
            fs = f['ecogDS']['sampFreq'].value
        print('Load time for h5 {}: {} seconds'.format(block_name,
                                                       time.time() - start))
        print('rates {}: {} {}'.format(block_name, rate, fs))
        if not np.allclose(rate, fs):
            assert rate < fs
            X = resample(X, rate, fs)
    except IOError:
        try:
            # Load raw HTK files
            rd_path = os.path.join(block_path, 'RawHTK')
            HTKoutR = HTK.read_HTKs(rd_path)
            X = HTKoutR['data'] * 1e6
            fs = HTKoutR['sampling_rate'] / srf
            print('Load time for htk {}: {} seconds'.format(block_name, time.time() - start))

        except OSError:
            try:
                # HDF5 .mat format
                with h5py.File(mat_ecog_path, 'r') as f:
                    X = f['ecogDS']['data'][:].T
                    fs = f['ecogDS']['sampFreq'][0]
                    print('Load time for h5.mat {}:' +
                          ' {} seconds'.format(block_name, time.time() - start))
            except IOError:
                # Old .mat format
                X = None
                fs = None
                data = loadmat(mat_ecog_path)['ecogDS']
                for ii, dtn in enumerate(data.dtype.names):
                    if dtn == 'data':
                        X = data.item()[ii]
                    elif dtn == 'sampFreq':
                        fs = data.item()[ii][0]
                assert X is not None
                assert fs is not None
                print('Load time for mat {}: {} seconds'.format(block_name, time.time() - start))

        try:
            os.mkdir(os.path.join(block_path, 'ecog400'))
        except OSError:
            pass
        if not np.allclose(rate, fs):
            assert rate < fs
            start1 = time.time()
            X = resample(X, rate, fs)
            print('Downsample time for {}: {}, {}, {}'.format(block_name,
                                                              time.time() - start1,
                                                              rate, fs))
        if np.allclose(rate, 400.):
            start = time.time()
            with h5py.File(h5_ecog_tmp_path, 'w') as f:
                g = f.create_group('ecogDS')
                g.create_dataset('data', data=X)
                g.create_dataset('sampFreq', data=rate)
            os.rename(h5_ecog_tmp_path, h5_ecog_path)
            print('Save time for {}400: {} seconds'.format(block_name,
                                                           time.time()-start))

    if X.shape[0] != total_channels:
        raise ValueError(block_name, X.shape, total_channels)

    bad_elects = load_bad_electrodes(block_path)
    if len(bad_elects) > 0:
        X[bad_elects] = np.nan

    # Subtract CAR
    start = time.time()
    X = subtract_CAR(X)
    print('CAR subtract time for {}: {} seconds'.format(block_name,
                                                        time.time()-start))

    # Apply Notch filters
    start = time.time()
    X = linenoise_notch(X, rate)
    print('Notch filter time for {}: {} seconds'.format(block_name,
                                                        time.time()-start))

    # Apply Hilbert transform and store
    if suffix is None:
        suffix_str = ''
    else:
        suffix_str = '_{}'.format(suffix)
    if phase:
        suffix_str = suffix_str + '_random_phase'
    fname = '{}_Hilb{}.h5'.format(block_name, suffix_str)

    hilb_path = os.path.join(block_path, fname)
    tmp_path = os.path.join(block_path, '{}_tmp.h5'.format(fname))

    with h5py.File(tmp_path, 'w') as f:
        note = 'applying Hilbert transform'
        dset = f.create_dataset('X', (len(cfs),
                                X.shape[0], X.shape[1]),
                                np.complex)
        theta = None
        if phase:
            theta = rng.rand(*X.shape) * 2. * np.pi
            theta = np.sin(theta) + 1j * np.cos(theta)
        for ii, (cf, sd) in enumerate(tqdm(zip(cfs, sds),
                                           note,
                                           total=len(cfs))):
            kernel = gaussian(X, rate, cf, sd)
            dset[ii] = hilbert_transform(X, rate, kernel, phase=theta)

        dset.dims[0].label = 'filter'
        dset.dims[1].label = 'channel'
        dset.dims[2].label = 'time'
        for val, name in ((cfs, 'filter_center'),
                          (sds, 'filter_sigma')):
            if name not in f.keys():
                f[name] = val
            dset.dims.create_scale(f[name], name)
            dset.dims[0].attach_scale(f[name])

        f.attrs['sampling_rate'] = rate
    os.rename(tmp_path, hilb_path)

    print('{} finished'.format(block_name))
    print('saved: {}'.format(hilb_path))


if __name__ == '__main__':
    main()
