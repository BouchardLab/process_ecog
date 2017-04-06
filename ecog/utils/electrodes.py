from __future__ import division
import glob, h5py, os

import numpy as np
from scipy.io import loadmat

__authors__ = "Alex Bujan (adapted from Kris Bouchard)"


__all__ = ['load_electrode_labels', 'load_bad_electrodes']


def load_electrode_labels(subj_dir):
    """
    Get anatomy. Try newest format, and then progressively earlier formats.

    Parameters
    ----------
    subj_dir : str
        Path to subject directory.

    Returns
    -------
    electrode_labels : ndarray
        Array of labels as strings.
    """
    anatomy_filename = glob.glob(os.path.join(subj_dir, 'anatomy', '*_anatomy.mat'))
    elect_labels_filename = glob.glob(os.path.join(subj_dir, 'elec_labels.mat'))
    hd_grid_file = glob.glob(os.path.join(subj_dir, 'Imaging', 'elecs', 'hd_grid.mat'))
    TDT_elecs_file = glob.glob(os.path.join(subj_dir, 'Imaging', 'elecs', 'TDT_elecs_all.mat'))

    if TDT_elecs_file:
        mat_in = loadmat(TDT_elecs_file[0])
        try:
            electrode_labels = mat_in['anatomy'][:, -1].ravel()
            print('anatomy from TDT_elecs_all')
            return np.array([a[0] for a in electrode_labels])
        except:
            pass

    if hd_grid_file:
        mat_in = loadmat(hd_grid_file[0])
        if 'anatomy' in mat_in:
            electrode_labels = np.concatenate(mat_in['anatomy'].ravel())
            print('anatomy hd_grid')

            return electrode_labels

    if anatomy_filename:
        electrode_labels = np.array([''] * 256, dtype='S6')
        try:
            anatomy = loadmat(anatomy_filename[0])
            anat = anatomy['anatomy']
            for name in anat.dtype.names:
                electrode_labels[anat[name][0, 0].flatten() - 1] = name
        except ValueError:
            with h5py.File(anatomy_filename[0], 'r') as f:
                anatomy = f['anatomy']
                for key in anatomy.keys():
                    electrode_labels[anatomy[key]-1] = key
        electrode_labels = np.array([word.decode("utf-8") for word in electrode_labels])


    elif elect_labels_filename:
        a = loadmat(elect_labels_filename[0])
        electrode_labels = np.array([elem[0] for elem in a['labels'][0]])

    else:
        electrode_labels = None
        print('No electrode labels found')

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
        Python (0-based) indices of bad electrodes.
    """

    bad_electrodes = []
    with open(os.path.join(blockpath, 'Artifacts', 'badChannels.txt'),'rt') as f:
        lines = f.readlines()
        lines = ' '.join(lines)
        els = lines.split(' ')
        els = [int(el) for el in els if el != '']

    bad_electrodes = np.array([el-1 for el in els], dtype=int)

    return bad_electrodes
