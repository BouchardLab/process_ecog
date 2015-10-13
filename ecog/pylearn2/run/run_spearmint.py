import os
from run_folds import get_result
from hyp_params import get_params, make_dir
import numpy as np

def main(job_id, job):
    print 'Imports done...'
    json_file = 'config.json'
    op_params, fixed_params = get_params(json_file)

    fixed_params['job_id'] = job_id

    make_dir(fixed_params)

    return -get_result(job, fixed_params)
