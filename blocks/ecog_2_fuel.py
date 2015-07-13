#!/usr/bin/env python
import h5py, os
import numpy as np
from pylearn2.datasets import ecog
from folds_dataset import H5PYDatasetFolds

pylearn2_folder = os.environ['PYLEARN2_DATA_PATH']
fuel_folder = os.environ['FUEL_DATA_PATH']
read_file = os.path.join(pylearn2_folder, 'ecog', 'EC2_CV_85_nobaseline_aug.h5')
write_file = os.path.join(fuel_folder, 'EC2_CV_85_nobaseline_aug_fuel.h5')

with h5py.File(read_file, 'r') as f:
    X = f['X'].value
    y = f['y'].value
    X_aug = f['X_aug'].value
    y_consonant = f['y_consonant'].value
    y_vowel = f['y_vowel'].value

with h5py.File(write_file, 'w') as f:
    X_ds = f.create_dataset('features', data=X)
    X_ds.attrs.create('DIMENSION_LABELS',
                      data=['batch', 'frequency', 'time', 'channels'])

    X_aug_ds = f.create_dataset('augmented_features', data=X_aug)
    X_aug_ds.attrs.create('DIMENSION_LABELS',
                          data=['batch', 'frequency', 'time', 'channels'])

    y_ds = f.create_dataset('targets',
            data=y.argmax(axis=1)[:,np.newaxis])
    y_ds.attrs.create('DIMENSION_LABELS',
                      data=['batch', 'index'])

    y_consonant_ds = f.create_dataset('consonant_targets',
                                      data=y_consonant.argmax(axis=1)[:,np.newaxis])
    y_consonant_ds.attrs.create('DIMENSION_LABELS',
                                data=['batch', 'index'])

    y_vowel_ds = f.create_dataset('vowel_targets',
                                  data=y_vowel.argmax(axis=1)[:,np.newaxis])
    y_vowel_ds.attrs.create('DIMENSION_LABELS',
                            data=['batch', 'index'])

    splits = []
    split_names = ('train', 'valid', 'test')
    names = ('split', 'source', 'fold', 'indices', 'available', 'comment')
    for fold in range(10):
        ds = ecog.ECoG(read_file, 'train', fold, level_classes=True)
        train_idx, valid_idx, test_idx, extra_idx = ds.indices
        for sp, inds in zip(split_names, ds.indices[:-1]):
            h_ds = f.create_dataset(sp+'_'+str(fold), data=inds)
            splits.append([sp, 'features', fold, h_ds.ref, True, ','])
            splits.append([sp, 'targets', fold, h_ds.ref, True, ','])
    f.attrs['split'] = H5PYDatasetFolds.create_split_array(splits)
