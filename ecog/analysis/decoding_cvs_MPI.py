#!/usr/bin/env python
import time,pdb,os,h5py,sys,glob
import numpy as np
from sklearn import svm
from sklearn import metrics,cross_validation
from optparse import OptionParser
from sklearn import linear_model as lm
from mpi4py import MPI
from sklearn.linear_model import LogisticRegression,Perceptron

#custom modules
from kCrossVal import kCrossValy
from utils.selectElectrodes import selectElectrodes

__author__ = 'Alex Bujan'

def main():
    usage = '%prog [options]'
    parser = OptionParser(usage)

    parser.add_option("-p","--path",type="string")
    parser.add_option("-l","--labels",type="string")
    parser.add_option("-r","--part",type="string",default='none',)
    parser.add_option("-m","--model",type="string",default='svm')
    parser.add_option("-s","--subject",type="string",default='EC2')
    parser.add_option("-e","--elect",action='store_true',dest='elect')
    parser.add_option('-v','--vsmc',action='store_true',dest='vsmc')

    (options, args) = parser.parse_args()

    if options.elect:
        elect=True
    else:
        elect=False

    if options.vsmc:
        vsmc = True
    else:
        vsmc = False

    ldir = glob.glob('%s'%options.path)

    assert len(ldir)>0,'No files found!'

    run(files=ldir,labels=options.labels,part=options.part,\
        model=options.model,electrodes=elect,subject=options.subject,\
        vsmc=vsmc)

def run(files,labels,part='none',model='svm',electrodes=False,\
        subject='EC2',vsmc=True):

    """
    MPI
    """

    MAIN_comm = MPI.COMM_WORLD
    MAIN_size = MAIN_comm.Get_size()
    MAIN_rank = MAIN_comm.Get_rank()

    n_proc_per_file = MAIN_size//len(files)
    new_comm_id = MAIN_rank//n_proc_per_file

    FILE_comm = MAIN_comm.Split(new_comm_id)
    FILE_rank = FILE_comm.Get_rank()
    FILE_size = FILE_comm.Get_size()

    if FILE_rank==0:
        print '\nRank [%i]: Number of processes per file: %i'%(MAIN_rank,FILE_size)

    '''
    Load/Broadcast data
    '''

    if FILE_rank==0:
        filename = files[new_comm_id]
        print '\nRank [%i]: Loading file %s ...'%(MAIN_rank,filename)
        tic = MPI.Wtime()
        with h5py.File(filename,'r') as f:
            X       = f['X'].value
            if part=='amplitude':
                X   = abs(X).astype('f4')
            elif part =='phase':
                X   = np.angle(X).astype('f4')
            t,m,n   = X.shape
            y       = f['y'].value.astype('int')
            if MAIN_rank==0:
                blocks = f['block'].value
    else:
        t = None
        m = None
        n = None

    if  FILE_rank==0:
        print '\nRank [%i]: Data loaded in %.4f seconds. Shape of data set: %s'%(MAIN_rank,MPI.Wtime()-tic,str(X.shape))

    '''
    Select electrodes
    '''

    if electrodes:
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
    Bcast data
    '''

    if FILE_rank==0:
       print '\nRank [%i]: Bcasting data across processes ...'%MAIN_rank
       tic = MPI.Wtime()

    FILE_comm.Barrier()
    t = FILE_comm.bcast(obj=t,root=0)
    m = FILE_comm.bcast(obj=m,root=0)
    n = FILE_comm.bcast(obj=n,root=0)

    if FILE_rank!=0:
        X = np.zeros((t,m,n),dtype='f4')
        y = np.zeros(t,dtype='int')

    FILE_comm.Barrier()
    FILE_comm.Bcast([X,MPI.FLOAT])
    FILE_comm.Bcast([y,MPI.INT])

    if FILE_rank==0:
        print '\nRank [%i]: Data scattered in %.4f seconds.'%(MAIN_rank,MPI.Wtime()-tic) 

    if electrodes:
        X = X[...,elects]

    t,m,n = X.shape

    X = X.reshape((t,m*n))

    if FILE_rank==0:
        print '\nRank [%i]: # samples - %i ; # parameters - %i'%(MAIN_rank,t,m*n)

    '''
    Hyper-parameter and cross-validation ids
    '''

    n_hps = 10
    n_folds = 10

    C = np.logspace(-5,2,n_hps,dtype='f8')

    assert FILE_size//n_hps==n_folds,'Wrong number of processes.'

    HP_id = FILE_rank // n_folds
    CV_id = FILE_rank %  n_folds

    my_C = C[HP_id]
    my_seed = np.random.randint(9999,size=n_folds)[CV_id]

    if FILE_rank==0:
        print '\nRank [%i]: Getting train,validation and test ids ...'%(MAIN_rank)

    train_ids,val_ids,test_ids = kCrossValy(y,percent=.8,\
                                            n_folds=n_folds,seed=my_seed)

    if FILE_rank==0:
        print '\nRank [%i]: Getting task list ...'%(MAIN_rank)

    '''
    Labels
    '''

    labels   = h5py.File(labels,'r')
    taskList = labels['indices'].keys()

    '''
    Containers for storing results
    '''

    if FILE_rank==0:
        accuracy = np.zeros((len(taskList),FILE_size,3),dtype='f4')
    else:
        accuracy = None

    my_accuracy = np.zeros(3,dtype='f4')

    for i,task in enumerate(taskList):

        '''
        Map labels according to task
        '''

        y_map = map_labels(y-1,labels,task)

        """
        Initialize the classifier
        """

        if model=='svm':
            clf = svm.LinearSVC(C=my_C)
        elif model=='logistic':
            clf = LogisticRegression(C=my_C)
        elif model=='perceptron':
            clf = Perceptron(penalty='l1',alpha=my_C)
        elif model=='simple_perceptron':
            clf = Perceptron()
        elif model=='svm_l1':
            clf = svm.LinearSVC(C=my_C,penalty='l1',dual=False)

        """
        Training
        """

        if FILE_rank==0:
            print '\nRank [%i]: Training %s classifier with C=%.4f on task [%s] ...'%(model,my_C,task)
            tic = MPI.Wtime()

        clf.fit(X[train_ids[CV_id],:],y_map[train_ids[CV_id]])

        if FILE_rank==0:
            print '\tTraining completed in %.4f sec'%(MPI.Wtime(-tic))

        prediction_train = clf.predict(X[train_ids[CV_id],:])
        my_accuracy[0] = metrics.accuracy_score(y_map[train_ids[CV_id]],\
                                                prediction_train)

        """
        Validation
        """

        prediction_val  = clf.predict(X[val_ids[CV_id],:])
        my_accuracy[1] = metrics.accuracy_score(y_map[val_ids[CV_id]],\
                                                prediction_val)

        """
        Test
        """

        prediction_test  = clf.predict(X[test_ids[CV_id],:])
        my_accuracy[2]    = metrics.accuracy_score(y_map[test_ids[CV_id]],\
                                                  prediction_test)

        """
        Gather results
        """

        FILE_comm.Barrier()
        FILE_comm.Gather([accuracy[i],MPI.DOUBLE],[my_accuracy,MPI.DOUBLE])

        """
        Model selection
        """

        if FILE_rank==0:
            results_val = np.array(np.split(accuracy[i,:,1],n_hps)).mean(0)
            best_C = np.max(results_val)
            if len(best_C)>1:
                best_C = best_C[0]
            selection = (results_val==best_C)*1
        else:
            selection = np.zeros(n_hps,dtype='int')

        FILE_comm.Barrier()
        FILE_comm.Bcast([selection.astype('int'),MPI.INT])

        '''
        Create new containers
        '''

        n_labels = len(np.unique(y_map))

        if FILE_comm==0:
            CM = np.zeros((n_folds,n_labels**2),dtype='f4')
            W  = np.zeros((n_folds,n*m),dtype='f4')
        else:
            CM = None
            W  = None

        '''
        RMA to communicate the best weights and confusion matrix
        '''

        win_CM = MPI.Win.Create(CM,comm=FILE_comm)
        win_W  = MPI.Win.Create(W,comm=FILE_comm)

        win.Fence()

        if selection[HP_id]:

            cm = metrics.confusion_matrix(y_map[test_ids[CV_id]],\
                                    prediction_test).ravel().astype('f4')
            win_CM.Put(origin=cm,target_rank=0,target=CV_id*(n_labels**2)*4)

            w = clf.coef_.ravel().astype('f4')
            win_W.Put(origin=w,target_rank=0,target=CV_id*n*m*4)

        win.Fence()

        FILE_comm.Barrier()

        win.Free()

        if FILE_rank==0:

            if i==0:
                output_path,output_filename = os.path.split(os.path.normpath(filename))
                output_path+='/%s'%model

                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                output_filename = output_filename.split('.h5')[0]
                output_filename+='_%s_%s.h5'%(part,model)
                output_filename = os.path.join(output_path,output_filename)

                print '\nRank [%i]: Saving the data in %s ...'%(MAIN_rank,output_filename)

                f = h5py.File(output_filename,'r+')
                f.attrs['model'] = model
                f.attrs['param'] = C

            g = f.create_group(task)
            g.create_dataset(name='accuracy',data=accuracy[i],compression='gzip')
            g.create_dataset(name='CM',data=CM.reshape((n_folds,n_labels,n_labels)),\
                            compression='gzip')
            g.create_dataset(name='coeffs',data=weights.reshape(n,m),compression='gzip')

        FILE_comm.Barrier()

    if FILE_rank==0:
        f.close()
        print '\nRank [%i]: Analysis complete!'%MAIN_rank

def map_labels(inLabels,labels,lutKey):
    outLabels = np.zeros_like(inLabels)
    f = lambda label : labels['indices/%s'%lutKey].value[np.where(labels['indices/utterance'].value==label)][0]
    outLabels = map(f,inLabels)
    return np.asarray(outLabels)

if __name__=='__main__':
    main()
