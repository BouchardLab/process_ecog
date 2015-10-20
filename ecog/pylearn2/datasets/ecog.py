"""
ECoG dataset.
"""
__authors__ = "Jesse Livezey"

import numpy as N
np = N
import h5py
from theano.compat.six.moves import xrange
from pylearn2.datasets import hdf5, dense_design_matrix
from pylearn2.datasets import control
from pylearn2.datasets import cache
from pylearn2.utils import serial
from pylearn2.utils.rng import make_np_rng


_split = {'train': .8, 'valid': .1, 'test': .1}
assert np.allclose(_split['train']+_split['valid']+_split['test'],1.)

#class ECoG(hdf5.HDF5Dataset):
class ECoG(dense_design_matrix.DenseDesignMatrix):
    """
    ECoG dataset

    Parameters
    ----------
    filename : str
        Filename for data.
    which_set : str
        'train', 'valid', 'test', or 'augment'
    fold : int
        Which fold to use.
    center : bool
        If True, preprocess so that data has zero mean.
    move : float
        Fraction of data to move through for each fold.
    level_classes: bool
        Flag for making classes even over splits or just sampling randomly.
    consonant_prediction: bool
        Flag for just setting y to consonant class.
    vowel_prediction: bool
        Flag for just setting y to vowel class.
    two_headed: bool
        Flag for predicting consonant and vowel class in one network.
        Overrides consonant and vowel prediction.
    randomize_labels: bool
        Randomly permutes the labels for the examples.
        Meant for control runs.
    frac_train: float
        Percentage of training set to use during training.
    pm_aug_range: int
        Number of of time shifts to use in augmentation.
    """

    def __init__(self, filename, which_set,
                 fold=0, seed=20141210, center=False,
                 move = .1, level_classes=False,
                 consonant_prediction=False,
                 vowel_prediction=False,
                 two_headed=False,
                 randomize_labels=False,
                 frac_train=None,
                 pm_aug_range=None):

        self.args = locals()

        if which_set not in ['train', 'valid', 'test', 'augment']:
            raise ValueError(
                'Unrecognized which_set value "%s".' % (which_set,) +
                '". Valid values are ["train","valid","test"].')

        filename = serial.preprocess(filename)
        ext = filename.split('.')[-1]
        if ext == 'h5':
            with h5py.File(filename,'r') as f:
                X = f['X'].value
                y = f['y'].value
                if two_headed:
                    assert not consonant_prediction
                    assert not vowel_prediction
                    y_consonant = f['y_consonant'].value
                    y_vowel = f['y_vowel'].value
                elif consonant_prediction:
                    assert not vowel_prediction
                    y_consonant = f['y_consonant'].value
                elif vowel_prediction:
                    assert not consonant_prediction
                    y_vowel = f['y_vowel'].value
                if which_set == 'augment':
                    X_aug = f['X_aug'].value
                    assert X_aug.shape[1] == y.shape[0]
                    tile_len = X_aug.shape[0]
                    y_aug = y[np.newaxis,...]
                    y_aug = np.tile(y, (tile_len, 1, 1))
        elif ext == 'mat':
            assert not two_headed
            assert not (which_set == 'augment')
            assert not consonant_prediction
            assert not vowel_prediction
            with h5py.File(filename,'r') as f:
                X = f['out/D'].value[:, np.newaxis, ...]
                y = np.squeeze(f['out/labs'].value)-1
                raise NotImplemetedError
                if two_headed:
                    assert not consonant_prediction
                    assert not vowel_prediction
                    y_consonant = f['y_consonant'].value
                    y_vowel = f['y_vowel'].value
                elif consonant_prediction:
                    assert not vowel_prediction
                    y_consonant = f['y_consonant'].value
                elif vowel_prediction:
                    assert not consonant_prediction
                    y_vowel = f['y_vowel'].value
                if which_set == 'augment':
                    X_aug = f['X_aug'].value
                    assert X_aug.shape[1] == y.shape[0]
                    tile_len = X_aug.shape[0]
                    y_aug = y[np.newaxis,...]
                    y_aug = np.tile(y, (tile_len, 1, 1))

        if two_headed:
            y_labels = None
        elif consonant_prediction:
            y_labels = 19
        elif vowel_prediction:
            y_labels = 3
        else:
            y_labels = 57
            
        rng = np.random.RandomState(seed)

        def split_indices(indices, frac_train):
            """
            Split indices into training/validation/testing groups.
            """
            num_idx = len(indices)
            indices = np.array(indices, dtype=int)
            order = rng.permutation(num_idx)

            n_train = int(np.round(num_idx*_split['train']))
            n_valid = int(np.round(num_idx*_split['valid']))
            n_test = num_idx-n_train-n_valid

            train_start = fold*num_idx*move
            train_end = (train_start+n_train) % num_idx
            valid_start = train_end
            valid_end = (valid_start+n_valid) % num_idx
            test_start = valid_end
            test_end = (test_start+n_test) % num_idx

            if train_end > train_start:
                train_idx = order[train_start:train_end]
            else:
                train_idx = np.hstack((order[train_start:],order[:train_end]))
            if valid_end > valid_start:
                valid_idx = order[valid_start:valid_end]
            else:
                valid_idx = np.hstack((order[valid_start:],order[:valid_end]))
            if test_end > test_start:
                test_idx = order[test_start:test_end]
            else:
                test_idx = np.hstack((order[test_start:],order[:test_end]))
            if frac_train is not None:
                assert frac_train > 0.
                assert frac_train <= 1.
                n_keep = int(np.round(frac_train*len(train_idx)))
                extra_idx = train_idx[n_keep:]
                train_idx = train_idx[:n_keep]
            else:
                extra_idx = []
            return tuple([indices[idx].tolist() for idx in [train_idx, valid_idx, test_idx, extra_idx]])

        def check_indices(tr, va, te, ex):
            """
            Check that all indices were included and the training/validation/testing
            splits are independent.
            """
            tr = set(tr)
            va = set(va)
            te = set(te)
            ex = set(ex)
            union = tr | va | te | ex
            assert len(tr)+len(va)+len(te)+len(ex) == len(union)
            max_val = max(union)
            assert len(union)-1 == max_val

        if level_classes:
            n_classes = y.shape[1]
            class_indices = {}
            classes = y.argmax(axis=1)
            for ii in xrange(n_classes):
                class_indices[str(ii)] = np.nonzero(classes == ii)[0].tolist()
            total = 0
            for indices in class_indices.values():
                total += len(indices)
            assert total == y.shape[0]
            train_idx = []
            valid_idx = []
            test_idx = []
            extra_idx = []
            for indices in class_indices.values():
                tr, va, te, ex = split_indices(indices, frac_train)
                train_idx += tr
                valid_idx +=va
                test_idx += te
                extra_idx += ex
        else:
            n_examples = X.shape[0]
            indices = range(n_examples)
            train_idx, valid_idx, test_idx, extra_idx = split_indices(indices, frac_train)

        check_indices(train_idx, valid_idx, test_idx, extra_idx)

        self.indices = (train_idx, valid_idx, test_idx, extra_idx)


        if two_headed:
            y = np.hstack((y_consonant, y_vowel))
        elif consonant_prediction:
            assert not vowel_prediction
            y = y_consonant
        elif vowel_prediction:
            assert not consonant_prediction
            y = y_vowel

        if randomize_labels:
            if which_set == 'augment':
                raise NotImplementedError
            order = rng.permutation(y.shape[0])
            y = y[order]

        X_train = X[train_idx]
        X_valid = X[valid_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_valid = y[valid_idx]
        y_test = y[test_idx]

        self.train_mean = X_train.mean(0)

        if which_set == 'augment':
            img_shape = X_train.shape[1:]
            y_shape = y_train.shape[1:]
            if pm_aug_range is not None:
                possible_range = X_aug.shape[0]
                assert possible_range % 2 == 1
                assert possible_range >= 2*pm_aug_range+1
                pm_possible = int(np.round(possible_range-1)/2.)
                idxs = slice(pm_possible-pm_aug_range, possible_range-(pm_possible-pm_aug_range))
                X_aug = X_aug[idxs]
                y_aug = y_aug[idxs]
            X_aug = X_aug[:,train_idx]
            X_aug = X_aug.reshape(-1,*img_shape)
            y_aug = y_aug[:,train_idx]
            y_aug = y_aug.reshape(-1,*y_shape)
            X_aug = np.concatenate((X_train, X_aug))
            y_aug = np.concatenate((y_train, y_aug))
            order = rng.permutation(X_aug.shape[0])
            X_train = X_aug[order]
            y_train = y_aug[order]
            del X_aug
            del y_aug

        if (which_set == 'train') or (which_set == 'augment'):
            topo_view = X_train
            y_final = y_train
        elif which_set == 'valid':
            topo_view = X_valid
            y_final = y_valid
        else:
            topo_view = X_test
            y_final = y_test
        if center:
            topo_view = topo_view-self.train_mean[np.newaxis,...]

        kwargs = {}
        kwargs['axes'] = ('b', 0, 1, 'c')
        if y_labels is not None:
            if y_final.ndim > 1:
                y_final = y_final.argmax(axis=1)
            y_final = y_final.astype(int)
            kwargs['y_labels'] = y_labels
        else:
            y_final = y_final.astype('float32')

        super(ECoG, self).__init__(topo_view=topo_view.astype('float32'),
                                   y=y_final, **kwargs)

    def get_valid_set(self):
        """
        .. todo::

            WRITEME
        """
        args = {}
        args.update(self.args)
        del args['self']
        args['which_set'] = 'valid'
        return ECoG(**args)

    def get_test_set(self):
        """
        .. todo::

            WRITEME
        """
        args = {}
        args.update(self.args)
        del args['self']
        args['which_set'] = 'test'
        return ECoG(**args)
