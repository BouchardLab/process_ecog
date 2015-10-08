#!/usr/bin/env python
import os, cPickle
from run_folds import get_result
from run_random import random_params
from hyp_params import get_params, make_dir
import numpy as np

print 'Imports done...'

json_file = 'config.json'
opt_params, fixed_params = get_params(json_file)
outcome = {'name': 'accuracy'}

solvers = ['svd', 'lsqr', 'eigen']
shrinkages = ['auto', 'float']
fls = [0., 1.]
pcs = 140

seed = 20150427
rng = np.random.RandomState(seed)

job_id = 0
fixed_params['job_id'] = job_id

make_dir(fixed_params)

results = []
for solver in solvers:
    for shrinkage in shrinkages:
        for fl in np.linspace(*fls, num=10):
            lda_job = {}
            lda_job['pcs'] = pcs
            lda_job['solver'] = solver
            if solver != 'svd':
                if shrinkage == 'auto':
                    lda_job['shrinkage'] = 'auto'
                else:
                    lda_job['shrinkage'] = fl

            valid_accuracy = get_result(lda_job, fixed_params, lda=True)
            results.append((valid_accuracy, lda_job))
            job_id += 1

with open('lda_pca.pkl', 'w') as f:
    cPickle.dump(results, f)
