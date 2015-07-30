#!/usr/bin/env python
import numpy as np
from train_rnn import train_model
from param_utils import filter_params, ecog_static_params
from load_data import pylearn2_ecog


def main(job_id, opt_params):
    audio, batch_size, folds, init_params = ecog_static_params()
    opt_params = filter_params({})

    train_misclass = np.zeros(folds)
    valid_misclass = np.zeros(folds)
    test_misclass = np.zeros(folds)

    for ii in range(folds):
        static_params = init_params.copy()
        static_params['fold'] = ii
        data = pylearn2_ecog(audio, batch_size, static_params)
        tr, va, te = train_model(data, opt_params, static_params)
        train_misclass[ii] = tr
        valid_misclass[ii] = va
        test_misclass[ii] = te

    print '----------result------------'
    print 'mean', valid_misclass.mean()
    print 'std', valid_misclass.std()


if __name__ == "__main__":
    main(None, None)
