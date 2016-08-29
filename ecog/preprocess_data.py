from __future__ import division
import os
import glob

import numpy as np
import h5py
import scipy
from scipy.io import loadmat, savemat
import argparse
try:
    from tqdm import tqdm
except:
    tqdm = lambda x, *args, **kwargs: x

import .signal_processing.HTK_hilb as htk
import .signal_processing.downsample as dse
import .signal_processing.subtract_CAR as scar
import .signal_processing.apply_linenoise_notch as notch
import .signal_processing.apply_hilbert_transform as aht
import .signal_processing.delete_bad_time_segments as dbts
from .utils import load_electrode_labels

import pdb

__authors__ = "Alex Bujan (adapted from Kris Bouchard)"


def main():
    usage = '%prog [options]'

    parser = argparse.ArgumentParser(description='Preprocessing ecog data.')

    parser.add_argument('subject', type=str, help="Subject code")

    parser.add_argument('blocks', type=int, nargs='+',
                        help="Block number eg: '1'")

    parser.add_argument('path', type=str, help="Path to the data")

    parser.add_argument('r', '--rate', type=float, default=200.,
        help="Sampling rate of the processed signal (optional)")

    parser.add_argument('--vsmc', type=bool, default=True,
                        help="Include vSMC electrodes only (optional)")

    parser.add_argument('--ct', type='float', default=None,
                        help="Center frequency of the Gaussian filter (optional)")

    parser.add_argument('--sd', type=float, default=None,
                        help="Standard deviation of the Gaussian filter (optional)")

    parser.add_argument('n', '--neuro', type=bool, default=False,
                        help="Use standard neuroscience boxcar frequency bands")

    parser.add_argument('--srf', type=float, default=1e4,
                        help="Sampling rate factor. Read notes in HTK.py (optional)")

    args = parser.parse_args()

    for block in args.blocks:
        blockpath = os.path.join(args.path, args.subject,
                '{}_{}'.format(subject, block))
        transform(blockpath, rate=args.rate, vsmc=args.vsmc,
                  ct=args.ct, sd=args.sd,
                  neuro=args.neuro, srf=args.srf)


def transform(blockpath, rate=400., vsmc=False, cts=None, sds=None, srf=1e4,
              neuro=False, suffix=''):
    """
    Takes raw LFP data and does the standard hilb algorithm:
    1) CAR
    2) notch filters
    3) Hilbert transform on different Gaussian bands
    ...

    Saves to os.path.join(blockpath, subject + '_B' + block + '_Hilb.h5')

    Parameters
    ----------
    blockpath
    rate
    vsmc
    cts: filter center frequencies. If None, use Chang lab defaults
    sds: filer standard deviations. If None, use Chang lab defaults
    srf: htk multiple

    takes about 20 minutes to run on 1 10-min block

    Returns
    -------
    Nothing. The result is too big to have in memory and is saved filter-by-filter to a h5.



    #ct=87.75,sd=3.65
    """

    ######## setup bandpass hilbert filter parameters
    if cts is None:
        fq_min = 4.0749286538265
        fq_max = 200.
        scale = 7.
        cts = 2 ** (np.arange(np.log2(fq_min) * scale, np.log2(fq_max) * scale) / scale)
    else:
        cts = np.array(cts)

    if sds is None:
        sds = 10 ** ( np.log10(.39) + .5 * (np.log10(cts)))
    ########


    s_path, blockname = os.path.split(blockpath)

    # first, look for downsampled ECoG in block folder
    ds_ecog_path = os.path.join(blockpath, 'ecog400', 'ecog.mat')
    if os.path.isfile(ds_ecog_path):
        print('loading ecog')
        with h5py.File(ds_ecog_path, 'r') as f:
            X = f['ecogDS']['data'][:].T
            fs = f['ecogDS']['sampFreq'][0]
    else:
        """
        Load raw HTK files
        """
        rd_path = os.path.join(blockpath, 'RawHTK')
        HTKoutR = htk.read_HTKs(rd_path)
        X = HTKoutR['data']

        """
        Downsample to 400 Hz
        """
        X = dse.downsample_ecog(X, rate, HTKoutR['sampling_rate'] / srf)

        os.mkdir(os.path.join(blockpath, 'ecog400'))
        savemat(ds_ecog_path, {'ecogDS':{'data': X, 'sampFreq': rate}})

    """
    Subtract CAR
    """
    X = scar.subtractCAR(X)

    """
    Select electrodes
    """
    labels = load_electrode_labels(s_path)
    if vsmc:
        elects = np.where((labels == 'precentral') | (labels == 'postcentral'))[0]
    else:
        elects = range(256)

    badElects = np.loadtxt('/%s/Artifacts/badChannels.txt'%blockpath)-1
    X[badElects.astype('int')] = np.nan


    X = X[elects]

    """
    Discard bad segments
    """
    #TODO

    """
    Apply Notch filters
    """
    X = notch.apply_linenoise_notch(X, rate)

    """
    Apply Hilbert transform and store
    """

    hilb_path = os.path.join(blockpath, blockname + '_Hilb' + suffix +'.h5')
    with h5py.File(hilb_path, 'w') as f:
        dset_real = f.create_dataset('X_real', (len(cts), X.shape[0], X.shape[1]), 'float32', compression="gzip")
        dset_imag = f.create_dataset('X_imag', (len(cts), X.shape[0], X.shape[1]), 'float32', compression="gzip")
        for i, (ct, sd) in enumerate(tqdm(zip(cts, sds), 'applying Hilbert transform', total=len(cts))):
            dat = aht.apply_hilbert_transform(X, rate, ct, sd)
            dset_real[i] = dat.real.astype('float32')
            dset_imag[i] = dat.imag.astype('float32')

        for dset in (dset_real, dset_imag):

            dset.dims[0].label = 'filter'
            for val, name in ((cts, 'filter_center'), (sds, 'filter_sigma')):
                if name not in f.keys():
                    f[name] = val
                dset.dims.create_scale(f[name], name)
                dset.dims[0].attach_scale(f[name])

            dset.dims[1].label = 'channel'
            dset.dims[2].label = 'time'
        f.attrs['sampling_rate'] = rate


if __name__=='__main__':
    main()
