import numpy as np
import h5py

def get_token_ids(filename):
    """
    Extracts tokens and ids from Ben's mat files
    Arguments
    ---------
    filename: string
            filename (with path if not in dir)
    Outputs
    -------
    token_ids: list of tuples
            token names and ids
    """
    f = h5py.File(filename,'r')
    get_token = lambda ref: f[ref].value.tostring().decode("utf-16")
    return zip(map(get_token,f['out/slist'][0]),np.unique(f['out/labs'].value.astype('int')))

