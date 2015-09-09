#!/usr/bin/env python
import numpy as np
from train_rnn import train_model
from param_utils import filter_params, ecog_static_params
from load_data import pylearn2_ecog


def main(job_id, opt_params):
    # Load parameters for dataset
    ds_params, static_params = ecog_static_params()
    # Set defaults
    opt_params = filter_params(opt_params)
    folds = static_params['folds']

    train_misclass = np.zeros(folds)
    valid_misclass = np.zeros(folds)
    test_misclass = np.zeros(folds)

    for ii in range(folds):
        ds_params['fold'] = ii
        data = pylearn2_ecog(ds_params, static_params)
        tr, va, te = train_model(job_id, data, opt_params, static_params)
        train_misclass[ii] = tr
        valid_misclass[ii] = va
        test_misclass[ii] = te

    print '----------result------------'
    print 'train mean', train_misclass.mean()
    print 'train std', train_misclass.std()
    print 'valid mean', valid_misclass.mean()
    print 'valid std', valid_misclass.std()


if __name__ == "__main__":
    main(0, {})
