from __future__ import print_function, division

import argparse, h5py, time, os
import numpy as np
from scipy.io import loadmat

try:
    from tqdm import tqdm
except:
    def tqdm(x, *args, **kwargs):
        return x

from ecog.utils import HTK
from ecog.signal_processing import resample
import ecog.signal_processing.subtract_CAR as scar
from ecog.signal_processing import linenoise_notch
from ecog.signal_processing import hilbert_transform
from ecog.signal_processing.apply_hilbert_transform import gaussian, hamming
from ecog.utils import load_bad_electrodes, bands


__authors__ = "Alex Bujan (adapted from Kris Bouchard)"


def main():

    parser = argparse.ArgumentParser(description='Preprocessing ecog data.')
    parser.add_argument('path', type=str, help="Path to the data")
    parser.add_argument('subject', type=str, help="Subject code")
    parser.add_argument('blocks', type=int, nargs='+',
                        help="Block number eg: '1'")
    parser.add_argument('-r', '--rate', type=float, default=400.,
                        help='Resample data to this rate.')
    parser.add_argument('--cfs', type=float, default=None,
                        help="Center frequency of the Gaussian filter")
    parser.add_argument('--sds', type=float, default=None,
                        help="Standard deviation of the Gaussian filter")
    parser.add_argument('-n', '--neuro', default=False, action='store_true',
                        help="Use standard neuroscience frequency bands")
    parser.add_argument('--srf', type=float, default=1e4,
                        help="Sampling rate factor. Read notes in HTK.py")
    args = parser.parse_args()

    for block in args.blocks:
        block_path = os.path.join(args.path, args.subject,
                                  '{}_B{}'.format(args.subject, block))
        transform(block_path, rate=args.rate,
                  cfs=args.cfs, sds=args.sds,
                  neuro=args.neuro, srf=args.srf)


def transform(block_path, rate=400., cfs=None, sds=None, srf=1e4,
              neuro=False, suffix=None, total_channels=256):
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
    srf: htk multiple

    takes about 20 minutes to run on 1 10-min block
    """

    if neuro:
        band_names = bands.neuro['bands']
        min_freqs = bands.neuro['min_freqs']
        max_freqs = bands.neuro['max_freqs']
    else:
        bands.neuro = bands.chang_lab
        if cfs is None:
            cfs = bands.neuro['cfs']
        else:
            cfs = np.array(cfs)

        if sds is None:
            sds = bands.neuro['cfs']
        else:
            sds = np.array(sds)

    subj_path, block_name = os.path.split(block_path)

    start = time.time()
    h5_ecog_path = os.path.join(block_path, 'ecog400', 'ecog.h5')
    h5_ecog_tmp_path = os.path.join(block_path, 'ecog400', 'ecog_tmp.h5')
    mat_ecog_path = os.path.join(block_path, 'ecog400', 'ecog.mat')
    try:
        # HDF5 format
        with h5py.File(h5_ecog_path, 'r') as f:
            X = f['ecogDS']['data'].value
            fs = f['ecogDS']['sampFreq'].value
        print('Load time for h5 {}: {} seconds'.format(block_name,
                                                       time.time()-start))
        print('rates {}: {} {}'.format(block_name, rate, fs))
        if not np.allclose(rate, fs):
            assert rate < fs
            X = resample.resample_ecog(X, rate, fs)
    except IOError:
        try:
            # HDF5 .mat format
            with h5py.File(mat_ecog_path, 'r') as f:
                X = f['ecogDS']['data'][:].T
                fs = f['ecogDS']['sampFreq'][0]
                print('Load time for h5.mat {}: {} seconds'.format(block_name,
                                                               time.time()-start))
        except IOError:
            try:
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
                print('Load time for mat {}: {} seconds'.format(block_name,
                                                               time.time()-start))
            except IOError:
                # Load raw HTK files
                rd_path = os.path.join(block_path, 'RawHTK')
                HTKoutR = HTK.read_HTKs(rd_path)
                X = HTKoutR['data']
                fs = HTKoutR['sampling_rate'] / srf
                print('Load time for htk {}: {} seconds'.format(block_name,
                                                    time.time()-start))
        try:
            os.mkdir(os.path.join(block_path, 'ecog400'))
        except OSError:
            pass
        if not np.allclose(rate, fs):
            assert rate < fs
            X = resample.resample_ecog(X, rate, fs)
        if np.allclose(rate, 400.):
            start = time.time()
            with h5py.File(h5_ecog_tmp_path, 'w') as f:
                g = f.create_group('ecogDS')
                g.create_dataset('data', data=X)
                g.create_dataset('sampFreq', data=rate)
            os.rename(h5_ecog_tmp_path, h5_ecog_path)
            print('Save time for {}: {} seconds'.format(block_name,
                                                        time.time()-start))

    assert X.shape[0] == total_channels, (block_name, X.shape)

    bad_elects = load_bad_electrodes(block_path)
    if len(bad_elects) > 0:
        X[bad_elects] = np.nan

    # Subtract CAR
    start = time.time()
    X = scar.subtract_CAR(X)
    print('CAR subtract time for {}: {} seconds'.format(block_name,
                                                        time.time()-start))

    # Apply Notch filters
    start = time.time()
    X = linenoise_notch.apply_linenoise_notch(X, rate)
    print('Notch filter time for {}: {} seconds'.format(block_name,
                                                        time.time()-start))

    # Apply Hilbert transform and store
    if suffix is None:
        suffix_str = ''
    else:
        suffix_str = '_{}'.format(suffix)
    if neuro:
        hilb_path = os.path.join(block_path,
                                 '{}_neuro_Hilb{}.h5'.format(block_name,
                                                             suffix_str))
    else:
        hilb_path = os.path.join(block_path,
                                 '{}_Hilb{}.h5'.format(block_name,
                                                       suffix_str))
    tmp_path = os.path.join(block_path, '{}_tmp.h5'.format(block_name))

    with h5py.File(tmp_path, 'w') as f:
        note = 'applying Hilbert transform'
        if neuro:
            dset = f.create_dataset('X', (len(band_names),
                                    X.shape[0], X.shape[1]),
                                    np.complex, compression="gzip")
            for ii, (min_f, max_f) in enumerate(tqdm(zip(min_freqs, max_freqs),
                                                     note,
                                                     total=len(min_freqs))):
                kernel = hamming(X, rate, min_f, max_f)
                dset[ii] = hilbert_transform(X, rate, kernel)

            dset.dims[0].label = 'band'
            dset.dims[1].label = 'channel'
            dset.dims[2].label = 'time'
            for val, name in ((min_freqs, 'min frequency'),
                              (max_freqs, 'max frequency')):
                if name not in f.keys():
                    f[name] = val
                dset.dims.create_scale(f[name], name)
                dset.dims[0].attach_scale(f[name])

        else:
            dset = f.create_dataset('X', (len(cfs),
                                    X.shape[0], X.shape[1]),
                                    np.complex, compression="gzip")
            for ii, (cf, sd) in enumerate(tqdm(zip(cfs, sds),
                                               note,
                                               total=len(cfs))):
                kernel = gaussian(X, rate, min_f, max_f)
                dset[ii] = hilbert_transform(X, rate, kernel)

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


if __name__ == '__main__':
    main()
