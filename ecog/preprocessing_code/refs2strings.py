import h5py,pdb
import numpy as np

def get_cv_ids(filename):
    """
    Extracts cvs and ids from Ben's mat files
    Arguments
    ---------
    filename: string
            filename (with path if not in dir)
    Outputs
    -------
    cv_ids: list of tuples
            cv names and ids
    """
    f = h5py.File(filename,'r')
    get_cv = lambda ref: f[ref].value.tostring().decode("utf-16").lower()
    return zip(map(get_cv,f['out/slist'][0]),np.unique(f['out/labs'].value.astype('int')))


def get_cvs(filename,cvArray):
    """
    Returns cv-data and cv-ids that found a match in cvArray. 
    The cv-ids are regenerated to match the indices in cvArray +1 
    (to avoid id=0).
    Arguments
    ---------
    filename: string
            filename (with path if not in dir)
    cvArray: array of strings
            Ordered array of target cvs
    Outputs
    -------
    new_data: array
            cv-data that found a match in cvs
    new_ids: array
            new cv-ids that found a match in cvs
    """
    f = h5py.File(filename,'r')
    #get cv-id map for this file
    cvIds = get_cv_ids(filename)
    #map trial-ids to cvs
    id2cv = lambda x: cvIds[np.where(np.asarray(cvIds)==str(x))[0]][0]
    allcvs = map(id2cv,f['out/labs'].value.astype('int').ravel())
    #select only if present in cvArray
    in_cvArray = lambda x : x in cvArray
    cvs_in_cvArray = np.asarray(map(in_cvArray,allcvs))
    new_cvs = np.asarray(allcvs)[cvs_in_cvArray]
    #change ids to match index in cvArray+1
    get_ids = lambda x: np.where(cvArray==x)[0][0]+1
    new_ids = np.asarray(map(get_ids,new_cvs))
    #get the data
    new_data = f['out/D'].value[cvs_in_cvArray]
    return new_data,new_ids

