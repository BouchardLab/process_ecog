#!/usr/bin/env python
import numpy as np
import argparse,h5py,multiprocessing,sys,os,time,pdb,glob
from sklearn.decomposition import FastICA,PCA
from scipy.io import loadmat

__author__ = 'Alex Bujan'

def computeICA(args):
    filename,electrode_path,vsmc,bad_path,zscore,n_components=args
    with h5py.File(filename,'r') as f:
        X = f['X'].value
        y = f['y'].value
        blocks_ = f['block'].value
    electrodes = loadmat(electrode_path)
    if vsmc:
        elects = np.hstack([electrodes['anatomy']['preCG'][0][0][0],electrodes['anatomy']['postCG'][0][0][0]])-1
    else:
        elects = electrodes
    blocks = np.unique(blocks_)
    for i in xrange(len(blocks)):
        if i==0:
            badElects = loadtxt(bad_path%blocks[i])-1
        else:
            badElects = np.hstack((badElects,loadtxt(bad_path%blocks[i])-1))
    badElects = np.unique(badElects)
    elects = np.setdiff1d(elects,badElects)
    X = X[...,elects]
    if zscore:
        X = (X-X.mean(1,keepdims=True))/X.std(1,keepdims=True)
    X_ica = np.zeros_like(X)
    if n_components is None:
        n_components=X.shape[-1]
    for i in xrange(X.shape[0]):
        try:
            ica = FastICA(n_components=n_components,whiten=True,max_iter=1000)
            X_ica[i] = ica.fit_transform(X[i])
        except:
            X_ica[i] = np.ones_like(X[i])*np.nan
            print 'Utterance %i could not be analyzed'%i
    bad_icas = np.unique(np.where(np.isnan(X_ica))[0])
    ids = setdiff1d(np.arange(X_ica.shape[0]),bad_icas)
    X_ica = X_ica[ids]
    y_ica = y[ids]
    blocks_ica = blocks_[ids]
    output_path,output_filename = os.path.split(os.path.normpath(s))
    output_path+='/ICA'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_filename = output_filename.split('.')[0]
    output_filename+='_%iICA.h5'%n_components
    output_filename = os.path.join(output_path,output_filename)
    with h5py.File(output_filename) as f:
        f.create_dataset('X', data=X_ica.astype('float32'),compression='gzip')
        f.create_dataset('y', data=y_ica,compression='gzip')
        f.create_dataset('blocks',data=blocks_ica,compression='gzip')
        f.create_dataset('elects',data=elects,compression='gzip')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute ICA')
    parser.add_argument('path', help='glob path to retrieve files')
    parser.add_argument('-n', '--processes', type=int, default=1)
    parser.add_argument('-e', '--electrode_path', default=None)
    parser.add_argument('-v', '--vsmc', type=bool, default=True)
    parser.add_argument('-b', '--bad_path', default=None)
    parser.add_argument('-z', '--zscore', type=bool, default=True)
    parser.add_argument('-k', '--n_components', default=None)
    args = parser.parse_args()
    ldir = glob.glob(args.path)
    args = [(filename,args.electrode_path,args.vsmc,\
            args.bad_path,args.zscore,args.n_components)
            for filename in ldir]
    if len(ldir)>1:
        pool = multiprocessing.Pool(args.processes)
        print '\nComputing ICA in parallel with %i processes...'%(pool._processes)
        results = pool.map(computeICA,args)
    else:
        print '\nComputing ICA serially ...'
        results = map(computeICA,args)


