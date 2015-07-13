# -*- coding: utf-8 -*-
from folds_dataset import H5PYDatasetFolds
from fuel import utils


class ECOG(H5PYDatasetFolds):
    u"""MNIST dataset.
    Parameters
    ----------
    which_sets : tuple of str
        Which split to load. Valid values are 'train' and 'test',
        corresponding to the training set (50,000 examples) and the test
        set (10,000 examples).

    """
    filename = 'EC2_CV_85_nobaseline_aug_fuel.hdf5'

    def __init__(self, which_sets, **kwargs):
        kwargs.setdefault('load_in_memory', True)
        super(ECOG, self).__init__(
            file_or_path=utils.find_in_data_path(self.filename),
            which_sets=which_sets, **kwargs)
