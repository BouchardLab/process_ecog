import cPickle, os, time
from pylearn2.config import yaml_parse
from pylearn2.utils import serial
from yaml_builder import build_yaml, build_dataset
import numpy as np

def get_final_val(fname, key):
    model = serial.load(fname)
    channels = model.monitor.channels
    return 1.-float(channels[key].val_record[-1])

def get_result(ins_dict, fixed_params, lda=False):
    n_folds = fixed_params['n_folds']
    scratch = fixed_params['scratch']
    exp_name = fixed_params['exp_name']
    job_id = fixed_params['job_id']
    valid_accuracy = np.zeros(n_folds)
    test_accuracy = np.zeros(n_folds)
    train_accuracy = np.zeros(n_folds)
    if lda:
        from sklearn.lda import LDA
        from pca import PCA
        print 'Starting training...'
        start = time.time()
        for fold in xrange(n_folds):
            pca_model = PCA(dim=ins_dict['pcs'])
            ds_params = fixed_params.copy()
            ds_params['fold'] = fold
            ds = yaml_parse.load(build_dataset(ds_params))
            X = pca_model.fit_transform(ds.X)
            model = LDA(ins_dict)
            model.fit(X, ds.y.argmax(axis=1))
            vs = ds.get_valid_set()
            ts = ds.get_test_set()
            train_accuracy[fold] = model.score(X, ds.y.argmax(axis=1))
            valid_accuracy[fold] = model.score(pca_model.transform(vs.X), vs.y.argmax(axis=1))
            test_accuracy[fold] = model.score(pca_model.transform(ts.X), ts.y.argmax(axis=1))
            filename = os.path.join(scratch, exp_name, str(job_id)+'_fold'+str(fold)+'.pkl')
            with open(filename, 'w') as f:
                cPickle.dump(model, f)
    else:
        if ins_dict['n_conv_layers'] > 0:
            fixed_params['conv'] = True
            fixed_params['in_shape'] = fixed_params['shape']
            fixed_params['in_channels'] = fixed_params['channels']
        else:
            fixed_params['conv'] = False
            fixed_params['in_shape'] = fixed_params['shape']
            fixed_params['in_channels'] = fixed_params['channels']
            #fixed_params['in_shape'] = np.prod(fixed_params['shape'])*fixed_params['channels']

        ins_dict = ins_dict.copy()
        fixed_params = fixed_params.copy()
        print 'Starting training...'
        start = time.time()
        for fold in xrange(n_folds):
            ins_dict['fold'] = fold
            ins_dict['filename'] = os.path.join(scratch, exp_name, str(job_id)+'_fold'+str(fold)+'.pkl')
            train = build_yaml(ins_dict, fixed_params)
            yaml_file = os.path.join(scratch, exp_name, str(job_id)+'_fold'+str(fold)+'.yaml')
            with open(yaml_file, 'w') as f:
                f.write(train)
            print train
            train = yaml_parse.load(train)
            train.main_loop()
            del train
            train_accuracy[fold] = get_final_val(ins_dict['filename'], 'train_y_misclass')
            valid_accuracy[fold] = get_final_val(ins_dict['filename'], 'valid_y_misclass')
            test_accuracy[fold] = get_final_val(ins_dict['filename'], 'test_y_misclass')

    for fold in xrange(n_folds):
        print '--------------------------------------'
        print 'Accuracy fold '+str(fold)+':'
        print 'train: ',train_accuracy[fold]
        print 'valid: ',valid_accuracy[fold]
        print 'test: ',test_accuracy[fold]
    print '--------------------------------------'
    print 'final_train_mean_'+str(job_id)+': ',train_accuracy.mean()
    print 'final_valid_mean'+str(job_id)+': ',valid_accuracy.mean()
    print 'final_test_mean'+str(job_id)+': ',test_accuracy.mean()
    print '--------------------------------------'
    print 'final_train_std'+str(job_id)+': ',train_accuracy.std()
    print 'final_valid_std'+str(job_id)+': ',valid_accuracy.std()
    print 'final_test_std'+str(job_id)+': ',test_accuracy.std()
    print '--------------------------------------'
    print 'Total training time in seconds'
    print time.time()-start
    return valid_accuracy
