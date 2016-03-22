#!/usr/bin/env python
from __future__ import division
import numpy as np
import h5py,multiprocessing,sys,os,time,pdb,glob
from scipy.io import loadmat
from optparse import OptionParser
from mpi4py import MPI

#custom modules
from utils.selectElectrodes import selectElectrodes
from computePCA import computePCA
from complex_FastICA import complex_FastICA as cica


__author__ = 'Alex Bujan'

def main():
    usage = '%prog [options]'
    parser = OptionParser(usage)

    parser.add_option('-p','--path',type='string',default='')
    parser.add_option('-s','--subject',type='string',default='EC2')
    parser.add_option('-i','--max_iter', type='int',default=1000)
    parser.add_option('-v','--vsmc',action='store_true',dest='vsmc')
    parser.add_option('-k','--n_components',type='int',default=-1)
    parser.add_option('-a','--analysis',type='string',default='cICA')

    (options, args) = parser.parse_args()

    if options.vsmc is None:
        vsmc = False
    else:
        vsmc = True

    ldir = glob.glob('%s'%options.path)

    compute(files=ldir,subject=options.subject,vsmc=vsmc,\
            n_components=options.n_components,max_iter=options.max_iter,\
            analysis=options.analysis)


def compute(files,subject='EC2',vsmc=True,\
            n_components=-1,max_iter=1000,\
            analysis='cICA'):

    """
    MPI
    """
    MAIN_comm = MPI.COMM_WORLD
    MAIN_size = MAIN_comm.Get_size()
    MAIN_rank = MAIN_comm.Get_rank()

    n_proc_per_file = MAIN_size//len(files)
    new_comm_id = MAIN_rank//n_proc_per_file

    #split the communicator/ a sub-communicator per file in files
    FILE_comm = MAIN_comm.Split(new_comm_id)
    FILE_rank = FILE_comm.Get_rank()
    FILE_size = FILE_comm.Get_size()

    if MAIN_rank==0:
	print '\nRank [%i]: Number of processes per file: %i'%(MAIN_rank,FILE_size)

    '''
    Load/Broadcast data
    '''
    if FILE_rank==0:
        filename = files[new_comm_id]
        print '\nRank [%i]: Loading file %s ...'%(MAIN_rank,filename)
        tic = MPI.Wtime()

        with h5py.File(filename,'r') as f:
            X       = f['X'][:6]#.value
            X_r     = X.real
            X_i     = X.imag
            y       = f['y'][:6]#.value
            blocks  = f['block'][:6]#.value
            t,m,n   = X.shape

        assert X_r.dtype=='f4','X_r dtype is not f4'

        #split X into number of samples per process
        #X = np.array(np.split(X,n_proc_per_file))
        X_r = np.array(np.split(X_r,n_proc_per_file))
        X_i = np.array(np.split(X_i,n_proc_per_file))
        n_trials_per_proc = X_r.shape[1]
    else:
        n_trials_per_proc = None
        m = None
        n = None
        X_r = None
        X_i = None

    if  FILE_rank==0:
        print '\nRank [%i]: Data loaded in %.4f seconds. Shape of data set: %s'%(MAIN_rank,MPI.Wtime()-tic,str(X.shape))

#    '''
#    Load/Broadcast data
#    '''
#    filename = files[new_comm_id]
   
#    print '\nMain rank: [%i];Main size: [%i];\tFile rank: [%i];File size: [%i]'%(MAIN_rank,MAIN_size,FILE_rank,FILE_size)
 
#    if FILE_rank==0:
#        print '\nRank [%i]: Loading file %s ...'%(MAIN_rank,filename)
#        tic = MPI.Wtime()

#    with h5py.File(filename,'r',driver='mpio',comm=FILE_comm) as f:
#        f.atomic=True
#        t,m,n   = f['X'].shape
#        n_trials_per_proc = t//n_proc_per_file
#        start = FILE_rank*n_trials_per_proc
#        my_X  = f['X'][start:start+n_trials_per_proc]
#        if FILE_rank==0:
#            y       = f['y'].value
#            blocks  = f['block'].value

#    if  FILE_rank==0:
#        print '\nRank [%i]: Data loaded in %.4f seconds.'%(MAIN_rank,MPI.Wtime()-tic)

    '''
    Select electrodes
    '''

    if MAIN_rank==0:
        print '\nRank [%i]: Loading/bcasting electrode information ...'%MAIN_rank
        tic = MPI.Wtime()
        elects = selectElectrodes(subject,np.unique(blocks),vsmc).astype('int')
        n_elects = len(elects)
    else:
        n_elects = None

    MAIN_comm.Barrier()
    n_elects = MAIN_comm.bcast(obj=n_elects)
    if MAIN_rank!=0:
        elects = np.zeros(n_elects,dtype='int')

    MAIN_comm.Barrier()
    MAIN_comm.Bcast([elects,MPI.INT])
    if MAIN_rank==0:
        print '\nRank [%i]: Electrode info loaded and bcasted in %.4f seconds.'%(MAIN_rank,MPI.Wtime()-tic)

    '''
    Scatter data
    '''
    if FILE_rank==0:
       print '\nRank [%i]: Scattering data across processes ...'%MAIN_rank
       tic = MPI.Wtime()

    FILE_comm.Barrier()
    n_trials_per_proc = FILE_comm.bcast(obj=n_trials_per_proc,root=0)
    m = FILE_comm.bcast(obj=m,root=0)
    n = FILE_comm.bcast(obj=n,root=0)

    my_X_r = np.zeros((n_trials_per_proc,m,n),dtype='f4')
    my_X_i = np.zeros((n_trials_per_proc,m,n),dtype='f4')

    FILE_comm.Barrier()
    FILE_comm.Scatter([X_r,MPI.FLOAT],[my_X_r,MPI.FLOAT])
    FILE_comm.Scatter([X_i,MPI.FLOAT],[my_X_i,MPI.FLOAT])
   

    if FILE_rank==0:
        print '\nRank [%i]: Data scattered in %.4f seconds.'%(MAIN_rank,MPI.Wtime()-tic) 

    my_X = my_X_r[...,elects] + 1j*my_X_i[...,elects]

    assert my_X.dtype=='c8','my_X dtype is not c8'

    if FILE_rank==0:
        print '\nRank [%i]: Shape of data set: %s'%(MAIN_rank,str(my_X.shape))

    _,m,n = my_X.shape

    assert n_trials_per_proc==_,'Something went wrong: the number of trials per process is not correct!'

    if n_components<=0:
        n_components=my_X.shape[-1]

    my_X_new = np.zeros((n_trials_per_proc,m,n_components),dtype='c8')

    if analysis=='dPCA':
        my_X_dem = np.zeros((n_trials_per_proc,m,n),dtype='c8')

    if FILE_rank==0:
        print '\nRank [%i]: Computing %s with %i components ...'%(MAIN_rank,analysis,n_components)
        tic = MPI.Wtime()

    for i in xrange(n_trials_per_proc):
#        try:
#            if analysis=='dPCA':
#                my_X_new[i] = computePCA(my_X[i].T,n_components=n_components,whiten=True)[0].T
#                my_X_dem[i] = (my_X[i].T*np.exp(-np.angle(my_X_new[i,:,0])*1j)).T
#            elif analysis=='cICA':
        my_X_new[i] = cica(my_X[i].T,n_components=n_components,whiten=True,\
                           max_iter=max_iter)[2].T
        if FILE_rank==0:
             print '\nRank [%i]: cICA analysis finished successfully %i/%i!'%(MAIN_rank,(i+1),n_trials_per_proc)
#        except:
#            print '\nRank [%i]: Trial %i in batch %i could not be analyzed'%(MAIN_rank,i,new_comm_id)

    my_X_new_r = my_X_new.real
    my_X_new_i = my_X_new.imag

    assert my_X_new_r.dtype=='f4','y_X_new_r dtype is not f4'

    if FILE_rank==0:
        print '\nRank [%i]: shape of my_X_new_r: %s'%(MAIN_rank,str(my_X_new_r.shape))

    if analysis=='dPCA':
        my_X_dem_r = my_X_dem.real
        my_X_dem_i = my_X_dem.imag

    if FILE_rank==0:
        print '\nRank [%i]: %s analysis completed in %.4f seconds'%(MAIN_rank,analysis,MPI.Wtime()-tic)
#        X_new = np.zeros((n_proc_per_file,n_trials_per_proc,m,n_components),dtype=np.complex)

        X_new_r = np.zeros((n_proc_per_file,n_trials_per_proc,m,n_components),dtype='f4')
        X_new_i = np.zeros((n_proc_per_file,n_trials_per_proc,m,n_components),dtype='f4')

        if analysis=='dPCA':
#            X_dem = np.zeros((n_proc_per_file,n_trials_per_proc,m,n),dtype=np.complex)
            X_dem_r = np.zeros((n_proc_per_file,n_trials_per_proc,m,n),dtype='f4')
            X_dem_i = np.zeros((n_proc_per_file,n_trials_per_proc,m,n),dtype='f4')
    else:
#        X_new = None
        X_new_r = None
        X_new_i = None
        if analysis=='dPCA':
#            X_dem = None
            X_dem_r = None
            X_dem_i = None

    if FILE_rank==0:
        print my_X_new_r.shape,X_new_r.shape,my_X_new_r.dtype,X_new_r.dtype

    FILE_comm.Barrier()
#    FILE_comm.Gather([my_X_new,MPI.F_COMPLEX],[X_new,MPI.F_COMPLEX])
    FILE_comm.Gather([my_X_new_r,MPI.FLOAT],[X_new_r,MPI.FLOAT])
    FILE_comm.Gather([my_X_new_i,MPI.FLOAT],[X_new_i,MPI.FLOAT])

    if analysis=='dPCA':
#        FILE_comm.Gather([my_X_dem,MPI.F_COMPLEX],[X_dem,MPI.F_COMPLEX])
        FILE_comm.Gather([my_X_dem_r,MPI.FLOAT],[X_dem_r,MPI.FLOAT])
        FILE_comm.Gather([my_X_dem_i,MPI.FLOAT],[X_dem_i,MPI.FLOAT])
        
    if FILE_rank==0:
#        X_new = X_new.reshape((n_proc_per_file*n_trials_per_proc,m,n_components))
        X_new = X_new_r + 1j*X_new_i
        X_new = X_new.reshape((n_proc_per_file*n_trials_per_proc,m,n_components))

        if analysis=='dPCA':
            X_dem = X_dem_r + 1j*X_dem_i
            X_dem = X_dem.reshape((n_proc_per_file*n_trials_per_proc,m,n))

        output_path,output_filename = os.path.split(os.path.normpath(filename))
        output_path+='/%s'%analysis

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        output_filename = output_filename.split('.h5')[0]
        output_filename+='_%i_%s.h5'%(n_components,analysis)
        output_filename = os.path.join(output_path,output_filename)

        if FILE_rank==0:
            print '\nRank [%i]: Saving the data in %s ...'%(FILE_rank,output_filename)

        with h5py.File(output_filename,'w') as f:
            f.create_dataset('X', data=X_new,compression='gzip')
            if analysis=='dPCA':
                f.create_dataset('Xd', data=X_dem,compression='gzip')
            f.create_dataset('y', data=y,compression='gzip')
            f.create_dataset('blocks',data=blocks,compression='gzip')
            f.create_dataset('elects',data=elects,compression='gzip')


#    output_path,output_filename = os.path.split(os.path.normpath(filename))
#    output_path+='/%s'%analysis

#    if not os.path.exists(output_path):
#        os.makedirs(output_path)

#    output_filename = output_filename.split('.h5')[0]
#    output_filename+='_%i_%s.h5'%(n_components,analysis)
#    output_filename = os.path.join(output_path,output_filename)

#    if FILE_rank==0:
#        print '\nRank [%i]: Saving the data in %s ...'%(MAIN_rank,output_filename)

#    FILE_comm.Barrier()
#    with h5py.File(output_filename,'w',driver='mpio',comm=FILE_comm) as f:
#        f.atomic = True
#        X_out = f.create_dataset('X',shape=(t,m,n_components),dtype=np.complex)
#        X_out[start:start+n_trials_per_proc,...] = my_X_new
#        if analysis=='dPCA':
#            Xd_out = f.create_dataset('Xd',shape=(t,m,n_components),dtype=np.complex)
#            Xd_out[start:start+n_trials_per_proc]=my_X_dem
#        y_out      = f.create_dataset(name='y',shape=(t,),dtype='int')
#        blocks_out = f.create_dataset(name='blocks',shape=(t,),dtype='int')
#        elects_out = f.create_dataset(name='elects',shape=(len(elects),),dtype='int')
#        if FILE_rank==0:
#            y_out[:] = y
#            blocks_out[:] = blocks
#            elects_out[:] = elects

#    MAIN_comm.Barrier()
#    if MAIN_rank==0:
        print '\nRank [%i]: Analysis complete!'%MAIN_rank

if __name__ == "__main__":
    main()
