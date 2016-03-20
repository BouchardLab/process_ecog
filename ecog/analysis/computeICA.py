#!/usr/bin/env python
import numpy as np
import h5py,multiprocessing,sys,os,time,pdb,glob
from sklearn.decomposition import FastICA,PCA
from scipy.io import loadmat
from optparse import OptionParser
from preprocessing_code.selectElectrodes import selectElectrodes

__author__ = 'Alex Bujan'

def main():
    usage = '%prog [options]'
    parser = OptionParser(usage)

    parser.add_option('-p','--path',type='string',default='')
    parser.add_option('-s','--subject',type='string',default='')
    parser.add_option('-n','--n_processes', type='int',default=1)
    parser.add_option('-m','--max_iter', type='int',default=1000)
    parser.add_option("--vsmc",action='store_true',dest='vsmc')
    parser.add_option("--zscore",action='store_true',dest='zscore')
    parser.add_option('-k','--n_components',type='int',default=-1)

    (options, args) = parser.parse_args()

    if options.zscore is None:
        zscore = False
    else:
        zscore = True

    if options.vsmc is None:
        vsmc = False
    else:
        vsmc = True

    if not os.path.isfile(options.path):
        ldir = glob.glob('%s*.h5'%options.path)
    else:
        ldir = [options.path]

    args_ = [(filename,options.subject,vsmc,zscore,options.n_components,options.max_iter)
            for filename in ldir]

    if len(ldir)>1:
        pool = multiprocessing.Pool(options.n_processes)
        print '\nComputing ICA in parallel with %i processes...'%(pool._processes)
        results = pool.map(computeICA,args_)
    else:
        print '\nComputing ICA serially ...'
        results = map(computeICA,args_)

def computeICA(args):

    try:
        process = multiprocessing.current_process()
        rank = int(process.name.split('-')[-1])-1
    except:
        rank = 0

    filename,subject,vsmc,zscore,n_components,max_iter=args

    if rank==0:
        print '\nLoading file %s ...'%filename

    with h5py.File(filename,'r') as f:
        X = f['X'].value
        y = f['y'].value
        blocks = f['block'].value

    blocks = np.unique(blocks)

    elects = selectElectrodes(subject,blocks,vsmc)

    X = X[...,elects]

    if zscore:
        X = (X-X.mean(1,keepdims=True))/X.std(1,keepdims=True)

    X_ica = np.zeros_like(X)

    if n_components==-1:
        n_components=X.shape[-1]

    if rank==0:
        print '\nComputing ICA with %i components ...'%n_components

    tic = time.time()

    for i in xrange(X.shape[0]):
        try:
            ica = FastICA(n_components=n_components,whiten=True,max_iter=max_iter)
            X_ica[i] = ica.fit_transform(X[i])
        except:
            X_ica[i] = np.ones_like(X[i])*np.nan
            if rank==0:
                print '\nUtterance %i could not be analyzed'%i
    if rank==0:
        print '\nICA analysis completed in %.4f seconds'%(time.time()-tic)

    bad_icas = np.unique(np.where(np.isnan(X_ica))[0])
    ids = np.setdiff1d(np.arange(X_ica.shape[0]),bad_icas)

    X_ica = X_ica[ids]
    y_ica = y[ids]
    blocks_ica = blocks[ids]

    output_path,output_filename = os.path.split(os.path.normpath(filename))
    output_path+='/ICA'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_filename = output_filename.split('.h5')[0]
    output_filename+='_%i_ICA.h5'%n_components
    output_filename = os.path.join(output_path,output_filename)

    if rank==0:
        print '\nSaving the data in %s ...'%output_filename

    with h5py.File(output_filename,'w') as f:
        f.create_dataset('X', data=X_ica.astype('float32'),compression='gzip')
        f.create_dataset('y', data=y_ica,compression='gzip')
        f.create_dataset('blocks',data=blocks_ica,compression='gzip')
        f.create_dataset('elects',data=elects,compression='gzip')

if __name__ == "__main__":
    main()
