#!/usr/bin/env python
import numpy as np
import h5py,multiprocessing,sys,os,time,pdb,glob
from scipy.io import loadmat
from optparse import OptionParser
from utils.selectElectrodes import selectElectrodes
from computePCA import computePCA


__author__ = 'Alex Bujan'

def main():
    usage = '%prog [options]'
    parser = OptionParser(usage)

    parser.add_option('-p','--path',type='string',default='')
    parser.add_option('-s','--subject',type='string',default='')
    parser.add_option('-n','--n_processes', type='int',default=1)
    parser.add_option('-m','--max_iter', type='int',default=1000)
    parser.add_option('-v','--vsmc',action='store_true',dest='vsmc')
    parser.add_option('-k','--n_components',type='int',default=-1)
    parser.add_option('-a','--analysis',type='string',default='dPCA')

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

    args_list = [(filename,options.subject,vsmc,options.n_components,\
              options.max_iter,options.analysis) for filename in ldir]

    if len(ldir)>1:
        pool = multiprocessing.Pool(options.n_processes)
        print '\nAnalysing in parallel with %i processes...'%(pool._processes)
        results = pool.map(compute,args_list)
    else:
        print '\nAnalysing serially ...'
        results = map(compute,args_list)

def compute(args):

    try:
        process = multiprocessing.current_process()
        rank = int(process.name.split('-')[-1])-1
    except:
        rank = 0

    filename,subject,vsmc,n_components,max_iter,analysis=args

    if rank==0:
        print '\nLoading file %s ...'%filename

    with h5py.File(filename,'r') as f:
        X = f['X'].value
        y = f['y'].value
        blocks = f['block'].value

    blocks = np.unique(blocks)

    elects = selectElectrodes(subject,blocks,vsmc)

    X = X[...,elects]

    t,m,n = X.shape

    if n_components<=0:
        n_components=X.shape[-1]

    X_new = np.zeros((m,n_components),dtype=np.complex)

    if analysis=='dPCA':
        X_dem = np.zeros((m,n))

    if rank==0:
        print '\nComputing %s with %i components ...'%(analysis,n_components)

    tic = time.time()

    for i in xrange(t):
        try:
            if analysis=='dPCA':
                X_new[i] = computePCA(X[i].T,n_components=n_components,whiten=True)[0].T
                X_dem[i] = X[i]*np.exp(-np.angle(X_new[0])*1j)
            elif analysis=='cICA':
                pass
        except:
            X_new[i] = np.ones_like(X[i])*np.nan
            if rank==0:
                print '\nTrial %i could not be analyzed'%i
    if rank==0:
        print '\n%s analysis completed in %.4f seconds'%(analysis,time.time()-tic)

    bad_trials = np.unique(np.where(np.isnan(X_new))[0])
    ids = np.setdiff1d(np.arange(m),bad_trials)

    if len(ids)!=0:
        X_new = X_new[ids]
        y_new = y[ids]
        blocks_new = blocks[ids]

    output_path,output_filename = os.path.split(os.path.normpath(filename))
    output_path+='/%s'%analysis

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_filename = output_filename.split('.h5')[0]
    output_filename+='_%i_%s.h5'%(n_components,analysis)
    output_filename = os.path.join(output_path,output_filename)

    if rank==0:
        print '\nSaving the data in %s ...'%output_filename

    with h5py.File(output_filename,'w') as f:
        f.create_dataset('X', data=X_new,compression='gzip')
        if analysis=='dPCA':
            f.create_dataset('Xd', data=X_new,compression='gzip')
        f.create_dataset('y', data=y_new,compression='gzip')
        f.create_dataset('blocks',data=blocks_new,compression='gzip')
        f.create_dataset('elects',data=elects,compression='gzip')

if __name__ == "__main__":
    main()
