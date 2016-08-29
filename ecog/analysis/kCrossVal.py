import numpy as np
import pdb

def kCrossVal(X,percent=.8,n_folds=10,seed=1234):
    assert percent<1
    n_train     = int(X.shape[0]*percent)
    n_val = n_test = int(X.shape[0]*round((1-percent)/2.,1))

    train_ids   = np.zeros((n_folds,n_train))
    val_ids     = np.zeros((n_folds,n_val))
    test_ids    = np.zeros((n_folds,n_test))
    size = X.shape[0]/n_folds
    np.random.seed(seed)
    ids  = np.random.permutation(np.arange(X.shape[0]))
    for i in xrange(n_folds):
        if i==9:
            startVal     = size*i
            val_ids[i]   = ids[startVal:]
            test_ids[i]  = ids[:size]
            train_ids[i] = ids[size:startVal]
        else:
            startVal    = size*i
            startTest   = size*i+size
            val_ids[i]  = ids[startVal:startVal+size]
            test_ids[i] = ids[startTest:startTest+size]
            train_ids[i] = np.hstack((ids[:startVal],ids[startTest+size:]))
    return train_ids.astype('int'),\
            val_ids.astype('int'),\
            test_ids.astype('int')

def kCrossValy(y,percent=.8,n_folds=10,seed=1234):

    assert percent<1

    np.random.seed(seed)

    for i in xrange(n_folds):

        tmp_train_ids = []
        tmp_valid_ids = []
        tmp_test_ids  = []

        for clss in np.unique(y):

            tmp_class_ids = np.where(y==clss)[0]

            n_total_ids   = len(tmp_class_ids)

            n_train = int(np.floor(n_total_ids*percent))
            n_valid = int(np.floor(n_total_ids*round((1-percent)/2.,1)))
            n_test  = int(np.floor(n_total_ids*round((1-percent)/2.,1)))

            tmp_class_ids = tmp_class_ids[np.random.permutation(np.arange(n_total_ids))]

            tmp_train_ids.extend(tmp_class_ids[:n_train].tolist())
            tmp_valid_ids.extend(tmp_class_ids[n_train:n_train+n_valid].tolist())
            tmp_test_ids.extend(tmp_class_ids[n_train+n_valid:n_train+n_valid+n_test].tolist())

        train_ids,valid_ids,test_ids = np.asarray(tmp_train_ids),np.asarray(tmp_valid_ids),np.asarray(tmp_test_ids)

        assert len(np.intersect1d(train_ids,valid_ids))==0
        assert len(np.intersect1d(valid_ids,test_ids))==0

        if i==0:
            all_train_ids   = np.zeros((n_folds,train_ids.shape[0]))
            all_val_ids     = np.zeros((n_folds,valid_ids.shape[0]))
            all_test_ids    = np.zeros((n_folds,test_ids.shape[0]))

        all_train_ids[i]   = train_ids
        all_val_ids[i]     = valid_ids
        all_test_ids[i]    = test_ids
    return all_train_ids.astype('int'),\
           all_val_ids.astype('int'),\
           all_test_ids.astype('int')

def class_split(y,percent,n_folds=10,seed=1234):
    assert np.sum(percent)==1,"Percentages must sum to one."
    np.random.seed(seed)
    n_parts = len(percent)
    for i in xrange(n_folds):
        tmp_ids = []
        [tmp_ids.append([]) for p in xrange(n_parts)]
        for clss in np.unique(y):
            tmp_class_ids = np.where(y==clss)[0]
            n_total_ids   = len(tmp_class_ids)
            tmp_class_ids = tmp_class_ids[np.random.permutation(np.arange(n_total_ids))]
            n_ids = np.floor(n_total_ids*np.asarray(percent)).astype('int')
            for p in xrange(n_parts):
                start   = np.sum(n_ids[:p])
                end     = np.sum(n_ids[:p+1])
                tmp_ids[p].extend(tmp_class_ids[start:end].tolist())
        if i==0:
            array_list = []
            for p in xrange(n_parts):
                array_list.append(np.zeros((n_folds,len(tmp_ids[p]))))
        for p in xrange(n_parts):
            array_list[p][i] = np.asarray(tmp_ids[p])
    return array_list
