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
    parser.add_option("-n","--n_tasks",type="int",default=5)
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
        vsmc=vsmc,n_tasks=options.n_tasks)

def run(files,labels,part='none',model='svm',electrodes=False,\
        subject='EC2',vsmc=True,n_tasks=5):

    """
    MPI
    """

    MAIN_comm = MPI.COMM_WORLD
    MAIN_size = MAIN_comm.Get_size()
    MAIN_rank = MAIN_comm.Get_rank()

    n_proc_per_file = MAIN_size//len(files)
    new_comm_id1 = MAIN_rank//n_proc_per_file

    FILE_comm = MAIN_comm.Split(new_comm_id1)
    FILE_rank = FILE_comm.Get_rank()
    FILE_size = FILE_comm.Get_size()

    n_proc_per_task = FILE_size//n_tasks
    new_comm_id2 = FILE_rank//n_proc_per_task

    TASK_comm = FILE_comm.Split(new_comm_id2)
    TASK_rank = TASK_comm.Get_rank()
    TASK_size = TASK_comm.Get_size()

    if TASK_rank==0:
        print '\nRank [%i]:'%MAIN_rank
        print '\n\t- Number of processes per file: %i'%FILE_size
        print '\n\t- Number of processes per task: %i'%TASK_size
        start_time = MPI.Wtime()

    '''
    Load/Broadcast data
    '''

    if FILE_rank==0:
        filename = files[new_comm_id1]
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
                blocks = f['blocks'].value
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

    '''
    Labels
    '''

    if FILE_rank==0:
       print '\nRank [%i]: Reading label information ...'%MAIN_rank
       tic = MPI.Wtime()

    labels   = h5py.File(labels,'r')
    taskList = labels['indices'].keys()

    if FILE_rank==0:
        print '\nRank [%i]: Label info read in %.4f seconds.'%(MAIN_rank,MPI.Wtime()-tic)

    tokens = np.array(['baa', 'bee', 'boo', 'daa', 'dee', 'doo', 'faa', 'fee', 'foo',
                       'gaa', 'gee', 'goo', 'haa', 'hee', 'hoo', 'kaa', 'kee', 'koo',
                       'laa', 'lee', 'loo', 'maa', 'mee', 'moo', 'naa', 'nee', 'noo',
                       'paa', 'pee', 'poo', 'raa', 'ree', 'roo', 'saa', 'see', 'shaa',
                       'shee', 'shoo', 'soo', 'taa', 'tee', 'thaa', 'thee', 'thoo', 'too',
                       'vaa', 'vee', 'voo', 'waa', 'wee', 'woo', 'yaa', 'yee', 'yoo',
                       'zaa', 'zee', 'zoo'], 
                        dtype='|S4')

    dropit = lambda x : tokens[x] in labels['labels/utterance'].value

    dropouts = np.array(map(dropit,y))

    X = X[dropouts,...]
    y = y[dropouts]

    t,m,n = X.shape

    rename = lambda x : np.where(tokens[x]==labels['labels/utterance'].value)[0][0]

    y = np.array(map(rename,y)).astype('int')

    X = X.reshape((t,m*n))

    if FILE_rank==0:
        print '\nRank [%i]: # samples - %i ; # parameters - %i'%(MAIN_rank,t,m*n)

    '''
    Hyper-parameter and cross-validation ids
    '''

    n_hps = 10
    n_folds = 10

    C = np.logspace(-5,2,n_hps,dtype='f8')

    assert TASK_size//n_hps==n_folds,'Wrong number of processes.'

    HP_id = TASK_rank // n_folds
    CV_id = TASK_rank %  n_folds

    my_C = C[HP_id]
    my_seed = np.random.randint(9999,size=n_folds)[CV_id]

    '''
    Containers for storing results
    '''

    task = taskList[new_comm_id2]

    my_accuracy = np.zeros(3,dtype='f4')

    accuracy = np.zeros((TASK_size,3),dtype='f4')

    '''
    Map labels according to task
    '''

    if FILE_rank==0:
       print '\nRank [%i]: Mapping labels to task ...'%MAIN_rank
       tic = MPI.Wtime()

    y_map = map_labels(y,labels,task)

    if FILE_rank==0:
        print '\nRank [%i]: Labels mapped in %.4f seconds.'%(MAIN_rank,MPI.Wtime()-tic)

    if FILE_rank==0:
        print '\nRank [%i]: Getting train,validation and test ids ...'%(MAIN_rank)

    train_ids,val_ids,test_ids = kCrossValy(y_map,percent=.8,\
                                            n_folds=n_folds,seed=my_seed)

    if FILE_rank==0:
        print '\nRank [%i]: Getting task list ...'%(MAIN_rank)

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
    if TASK_rank==0:
        print '\nRank [%i]: Training %s classifier with C=%.4f on task [%s] ...'%(MAIN_rank,model,my_C,task)
        tic = MPI.Wtime()

    clf.fit(X[train_ids[CV_id],:],y_map[train_ids[CV_id]])

    if TASK_rank==0:
        print '\nRank [%i]: Training on task %s completed in %.4f sec'%(MAIN_rank,task,MPI.Wtime()-tic)

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

    if TASK_rank==0:
        print '\nRank [%i]: Gathering results for task %s...'%(MAIN_rank,task)
        print '\n\t- Shape of my_accuracy: %s'%str(my_accuracy.shape)
        print '\n\t- Shape of accuracy: %s'%str(accuracy.shape)

    if task=='utterance':
        print my_accuracy

    TASK_comm.Barrier()
    TASK_comm.Gather([my_accuracy,MPI.FLOAT],[accuracy,MPI.FLOAT])

    """
    Model selection
    """
    n_labels = len(np.unique(y_map))

    if TASK_rank==0:
        results_val = np.array(np.split(accuracy[:,1],n_hps)).mean(1)
        best_C = np.max(results_val)
        try:
            selection = np.where(results_val==best_C)[0][0]
        except:
            selection = np.where(results_val==best_C[0])[0][0]

        '''
        Create new containers
        '''
        CM = np.zeros((n_folds,n_labels,n_labels),dtype='f4')
        W  = np.zeros((n_folds,n_labels,m*n),dtype='f4')

    '''
    RMA to communicate the best weights and confusion matrix
    '''

    cm = metrics.confusion_matrix(y_map[test_ids[CV_id]],\
                            prediction_test).astype('f4')

    w = clf.coef_.reshape((n_labels,m*n)).astype('f4')

    win_cm = MPI.Win.Create(cm,comm=TASK_comm)
    win_w  = MPI.Win.Create(w,comm=TASK_comm)

    win_cm.Fence()
    win_w.Fence()

    if TASK_rank==0:
        for k in xrange(n_folds):
            win_cm.Get(origin=CM[k],target_rank=(selection*n_folds)+k,target=0)
            win_w.Get(origin=W[k],target_rank=(selection*n_folds)+k,target=0)

    win_cm.Fence()
    win_w.Fence()

    win_cm.Free()
    win_w.Free()

    if TASK_rank==0:
        print '\nRank [%i]: Task %s ready for storage ...'%(MAIN_rank,task)

    FILE_comm.Barrier()

    '''
    Create a group to write the results
    '''

    store_ids = np.arange(0,FILE_size,n_hps*n_folds).tolist()
    store_grp = FILE_comm.Get_group().Incl(store_ids)
    STORE_comm = FILE_comm.Create(store_grp)

    FILE_comm.Barrier()

    if FILE_rank in store_ids:

        output_path,output_filename = os.path.split(os.path.normpath(filename))
        output_path+='/%s'%model

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        output_filename = output_filename.split('.h5')[0]
        output_filename+='_%s_%s.h5'%(part,model)
        output_filename = os.path.join(output_path,output_filename)

        f = h5py.File(output_filename,'w',driver='mpio',comm=STORE_comm)

        f.attrs['model'] = model
        f.attrs['param'] = C

        if FILE_rank==0:
            print '\nRank [%i]: Saving %s data to %s ...'%(MAIN_rank,task,output_filename)
            tic = MPI.Wtime()

        n_labels_ = np.zeros(STORE_comm.size,dtype='int')
        n_labels_ = STORE_comm.allgather(sendobj=n_labels,recvobj=n_labels_)

        for i,t in enumerate(taskList):
            g = f.create_group(t)
            g.create_dataset(name='accuracy',shape=(TASK_size,3),dtype='f4')
            g.create_dataset(name='CM',shape=(n_folds,n_labels_[i],n_labels),\
                             dtype='f4')
            g.create_dataset(name='coeffs',shape =(n_folds,n_labels_[i],m,n),\
                             dtype='f4')
            if task==t:
                g['accuracy'][:] = accuracy
                g['CM'][:] = CM
                g['coeffs'][:] = W.reshape((n_folds,n_labels,m,n)).astype('f4')

        if FILE_rank==0:
            print '\nRank [%i]: Task %s saved in %.4f seconds!'%(MAIN_rank,task,\
                                                                MPI.Wtime()-tic)

    FILE_comm.Barrier()

    if FILE_rank==0:
        f.close()
        print '\nRank [%i]: Analysis was completed in %.4f seconds!'%(MAIN_rank,MPI.Wtime()-start_time)

def map_labels(inLabels,labels,lutKey):
    outLabels = np.zeros_like(inLabels)
    f = lambda label : labels['indices/%s'%lutKey].value[np.where(labels['indices/utterance'].value==label)][0]
    outLabels = map(f,inLabels)
    return np.asarray(outLabels)

if __name__=='__main__':
    main()
