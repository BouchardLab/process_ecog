from pylearn2.utils import serial
from pylearn2.config import yaml_parse
from pylearn2.datasets import ecog
from pylearn2.space import VectorSpace, Conv2DSpace, CompositeSpace
from pylearn2.expr import nnet
from pylearn2.models.mlp import FlattenerLayer
import os, h5py, theano, cPickle, copy
import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as T


def conf_mat2accuracy(c_mat, v_mat, cv_mat):
    c_accuracy = None
    v_accuracy = None
    cv_accuracy = None
    accuracy_per_cv = None

    if cv_mat is not None:
        cv_accuracy = np.zeros(len(cv_mat))
        accuracy_per_cv = np.zeros((len(cv_mat), 57))
        for ii, cvf in enumerate(cv_mat):
            cv_accuracy[ii] = np.diag(cvf).sum()/cvf.sum()
            for jj in range(57):
                accuracy_per_cv[ii,jj] = cvf[jj,jj]/cvf[jj].sum()
    if c_mat is not None:
        c_accuracy = np.zeros(len(c_mat))
        for ii, cf in enumerate(c_mat):
            c_accuracy[ii] = np.diag(cf).sum()/cf.sum()
    if v_mat is not None:
        v_accuracy = np.zeros(len(v_mat))
        for ii, vf in enumerate(v_mat):
            v_accuracy[ii] = np.diag(vf).sum()/vf.sum()

    return c_accuracy, v_accuracy, cv_accuracy, accuracy_per_cv

def indx_dict2conf_mat(indices_dicts, y_dims):
    c = None
    c_dim = 19
    v = None
    v_dim = 3
    cv = None
    cv_dim = c_dim*v_dim
    n_files = len(indices_dicts)
    n_targets = None
    n_folds = None

    def c_v_from_cv(cv, v_dim):
        return int(cv/v_dim), cv % v_dim 
    def cv_from_c_v(c, v, v_dim):
        return c*v_dim+v

    for idx_dict in indices_dicts:
        if n_folds is None:
            n_folds = len(idx_dict.keys())
        else:
            assert n_folds == len(idx_dict.keys())
        for key in idx_dict.keys():
            nt = len(idx_dict[key])
            assert not (nt != 1 and n_files != 1)
            if n_targets is None:
                n_targets = nt
            else:
                assert nt == n_targets
    if cv_dim in y_dims:
        cv = np.zeros((n_folds, cv_dim, cv_dim))
        c = np.zeros((n_folds, c_dim, c_dim))
        v = np.zeros((n_folds, v_dim, v_dim))
        idx_dict = indices_dicts[0]
        for key in idx_dict.keys():
            fold = int(key.split('fold')[1].split('.')[0])
            assert fold < n_folds
            indices = idx_dict[key][0]
            y_true = indices[:,0]
            y_pred = indices[:,1]
            for yt, yp in zip(y_true, y_pred):
                ct, vt = c_v_from_cv(yt, v_dim)
                cp, vp = c_v_from_cv(yp, v_dim)
                cv[fold, yt, yp] += 1
                c[fold, ct, cp] += 1
                v[fold, vt, vp] += 1
    elif (c_dim in y_dims) and (v_dim in y_dims) and n_files == 1:
        cv = np.zeros((n_folds, cv_dim, cv_dim))
        c = np.zeros((n_folds, c_dim, c_dim))
        v = np.zeros((n_folds, v_dim, v_dim))
        idx_dict = indices_dicts[0]
        for key in idx_dict.keys():
            fold = int(key.split('fold')[1].split('.')[0])
            assert fold < n_folds
            ci, vi = idx_dict[key]
            cti = ci[:,0]
            cpi = ci[:,1]
            vti = vi[:,0]
            vpi = vi[:,1]
            for ct, cp, vt, vp in zip(cti, cpi, vti, vpi):
                cvt = cv_from_c_v(ct, vt, v_dim)
                cvp = cv_from_c_v(cp, vp, v_dim)
                cv[fold, cvt, cvp] += 1
                c[fold, ct, cp] += 1
                v[fold, vt, vp] += 1
    elif (c_dim in y_dims) and (v_dim in y_dims) and n_files != 1:
        def get_key(keys, string):
            rval = [key for key in keys if string in key]
            assert len(rval) == 1
            return rval[0]

        cm, vm = indices_dicts
        for ii in xrange(n_folds):
            fold_str = 'fold'+str(ii)
            ck = get_key(cm.keys(), fold_str)
            vk = get_key(vm.keys(), fold_str)
            ci = cm[ck][0]
            vi = vm[vk][0]
        raise NotImplementedError

        cv = np.zeros((n_folds, cv_dim, cv_dim))
        c = np.zeros((n_folds, c_dim, c_dim))
        v = np.zeros((n_folds, v_dim, v_dim))
    elif c_dim in y_dims:
        c = np.zeros((n_folds, c_dim, c_dim))
        raise NotImplementedError
    elif v_dim in y_dims:
        v = np.zeros((n_folds, v_dim, v_dim))
        raise NotImplementedError
    else:
        raise ValueError('Data does not match dimensionality expectations')

    return c, v, cv


def get_model_results(model_folder, filename, fold, kwargs):
    kwargs = copy.deepcopy(kwargs)
    file_loc = os.path.join(model_folder, filename)
    model = serial.load(file_loc)
    X_sym = model.get_input_space().make_theano_batch()
    target_space = model.get_target_space()
    y_inpt = target_space.make_theano_batch()
    y_sym = y_inpt
    input_space = model.get_input_space()
    if kwargs['audio']:
        data_file = '${PYLEARN2_DATA_PATH}/ecog/audio_EC2_CV_mcep.h5'
    else:
        data_file = '${PYLEARN2_DATA_PATH}/ecog/EC2_CV_85_nobaseline_aug.h5'
    del kwargs['audio']
    ds = ecog.ECoG(data_file,
                   which_set='train',
                   fold=fold,
                   **kwargs)
    ts = ds.get_test_set()
    y_hat = model.fprop(X_sym)
    if isinstance(model.layers[-1], FlattenerLayer):
        comp_space = model.layers[-1].raw_layer.get_output_space()
        y_hat_list = list(comp_space.undo_format_as(y_hat, target_space))
        y_sym_list = list(target_space.format_as(y_inpt, comp_space))
        n_targets = len(y_hat_list)
    else:
        n_targets = 1
        y_hat_list = [y_hat]
        y_sym_list = [y_sym]
    misclass_sym = []
    indices_sym = []
    for ys, yh in zip(y_hat_list, y_sym_list):
        misclass_sym.append(nnet.Misclass(ys, yh))
        indices_sym.append(T.join(1, T.argmax(ys, axis=1, keepdims=True), T.argmax(yh, axis=1, keepdims=True)))

    f = theano.function([X_sym, y_inpt], misclass_sym+indices_sym+y_hat_list)
    it = ts.iterator(mode = 'sequential',
                     batch_size = ts.X.shape[0],
                     num_batches = 1,
                     data_specs = (CompositeSpace((model.get_input_space(),
                                                 model.get_target_space())),
                                   (model.get_input_source(), model.get_target_source())))
    X, y = it.next()
    rvals = f(X, y)
    misclass = list(rvals[:n_targets])
    indices = list(rvals[n_targets:2*n_targets])
    y_hats = list(rvals[2*n_targets:])
    return misclass, indices, y_hats

