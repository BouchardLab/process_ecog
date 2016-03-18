#!/usr/bin/env python
__author__ = 'David Conant, Jesse Livezey'

import argparse, h5py, multiprocessing, sys, os, time
import numpy as np

from .tokenize import transcripts, make_data


def htk_to_hdf5(path, blocks, output_folder=None, task='CV',
                align_window=None, align_pos = 0,
                data_type='HG', zscore='events',
                fband=None, mp=True):
    """
    Process task data into segments with labels.

    Parameters
    ----------
    path : str
        Path to subject folder.
    blocks : list of ints
        Blocks from which data is gathered.
    task : str
        Type of tokens to be extracted.
    align_window : list of two ints
        Time window in seconds around each token.
    align_pos : ints
        Align to start of this phoneme.
    data_type : str
        Type of data to use.

    Returns
    -------
    D : dict
        Dictionary containing tokens as keys and data as array.
    anat : str
        Anatomy of data channels
    start_times : dict
        Dictionary of start times per token.
    stop_times : dict
        Dictionary of stop times per token.
    """

    if output_folder is None:
        output_folder = path

    if task == 'CV':
        tokens = sorted(['baa', 'bee', 'boo', 'daa', 'dee', 'doo', 'faa', 'fee', 'foo',
                         'gaa', 'gee', 'goo', 'haa', 'hee', 'hoo', 'kaa', 'kee', 'koo',
                         'laa', 'lee', 'loo', 'maa', 'mee', 'moo', 'naa', 'nee', 'noo',
                         'paa', 'pee', 'poo', 'raa', 'ree', 'roo', 'saa', 'shaa', 'shee',
                         'shoo', 'see', 'soo', 'taa', 'thaa', 'thee', 'thoo', 'tee',
                         'too', 'vaa', 'vee', 'voo', 'waa', 'wee', 'woo','yaa','yee',
                         'yoo', 'zaa', 'zee', 'zoo'])
    else:
        raise ValueError("task must of one of ['CV']: "+str(task)+'.')

    if data_type not in ['HG','AS_I','AS_R','AS_AA']:
        raise ValueError("data_type must be one of ['HG','AS_I','AS_R']: "+str(data_type)+'.')

    if fband==None and 'AS' in data_type:
        raise ValueError("Specify the frequency band with data_type AS_I or AS_R")

    if align_window is None:
        align_window = np.array([-1., 1.])
    else:
        align_window = np.array(align_window)
        assert align_window[0] <= 0.
        assert align_window[1] >= 0.
        assert align_window[0] < align_window[1]

    def block_str(blocks):
        rval = 'blocks_'
        for block in blocks:
            rval += str(block) + '_'
        return rval

    def align_window_str(align_window):
        rval = 'align_window_' + str(align_window[0]) + '_to_'+ str(align_window[1])
        return rval

    folder, subject = os.path.split(os.path.normpath(path))
    if fband==None:
        fname = os.path.join(output_folder, 'hdf5', (subject + '_' + block_str(blocks)
                                           + task + '_' + data_type + '_'
                                           + align_window_str(align_window) + '_'
                                           + zscore + '.h5'))
    else:
        fname = os.path.join(output_folder, 'hdf5', (subject + '_' + block_str(blocks)
                                           + task + '_' + data_type + '_'
                                           + str(fband) + '_'
                                           + align_window_str(align_window) + '_'
                                           + zscore + '.h5'))

    anat = make_data.load_anatomy(path)

    D = dict((token, np.array([])) for token in tokens)
    stop_times = dict((token, np.array([])) for token in tokens)
    start_times = dict((token, np.array([])) for token in tokens)
    B = dict((token, np.array([],dtype='int')) for token in tokens)

    args = [(subject, block, path, tokens, align_pos, align_window, data_type, zscore, fband)
            for block in blocks]
    print '\nNumbers of blocks to be processed: %i'%(len(blocks))

    if mp and len(blocks)>1:
        pool = multiprocessing.Pool(len(blocks))
        print '\nProcessing blocks in parallel with %i processes...'%(pool._processes)
        results = pool.map(process_block, args)
    else:
        print '\nProcessing blocks serially ...'
        results = map(process_block, args)

    for Bstart, Bstop, BD, Bnumber in results:
        for token in tokens:
            start_times[token] = (np.hstack((start_times[token], Bstart[token])) if
                                  start_times[token].size else Bstart[token])
            stop_times[token] = (np.hstack((stop_times[token], Bstop[token])) if
                                 stop_times[token].size else Bstop[token])
            D[token] = (np.vstack((D[token], BD[token])) if
                        D[token].size else BD[token])
            B[token] = (np.hstack((B[token], Bnumber[token])) if
                        B[token].size else Bnumber[token])

    print('\nSaving to: '+fname)
    save_hdf5(fname, D, tokens, anat, B)

    return (D, anat, start_times, stop_times, B)

def process_block(args):
    """
    Process a single block.

    Parameters
    ----------
    subject : str
    block : int
    path : str
    tokens : list of str
    """
    subject, block, path, tokens, align_pos, align_window, data_type, zscore, fband = args

    try:
        process = multiprocessing.current_process()
        rank = int(process.name.split('-')[-1])-1
    except:
        rank = 0

    blockname = subject + '_B' + str(block)
    if rank==0:
        print('\nProcessing subject %s'%subject)
        print('-----------------------')
    blockpath = os.path.join(path, blockname)

    if rank==0:
        print('\nConvert parseout to dataframe ...')
    # Convert parseout to dataframe
    parseout = transcripts.parse(blockpath, blockname)
    df = transcripts.make_df(parseout, block, subject, align_pos)
    if rank==0:
        print('\nDataframe generated succesfully!')


    D = dict()
    stop_times = dict()
    start_times = dict()
    B = dict()

    all_event_times = np.array([])
    for token in tokens:
        match = (df['label'] == token)
        all_event_times = (np.hstack((all_event_times,
                                      df['align'][match & (df['mode'] == 'speak')]))
                           if all_event_times.size else df['align'][match & (df['mode'] == 'speak')])

    if rank==0:
        print('\nGenerating data matrices ...')
        count=1
        tic = time.time()

    for token in tokens:
        match = (df['label'] == token)
        event_times = df['align'][match & (df['mode'] == 'speak')]
        start = event_times.values + align_window[0]
        stop = event_times.values + align_window[1]

        start_times[token] = start.astype(float)
        stop_times[token] = stop.astype(float)

        if rank==0:
            tictic = time.time()

        data = make_data.run_makeD(blockpath, event_times, align_window,
                                   data_type, zscore, all_event_times, fband)

#        resampled_data = make_data.resample_data(data)
#        D[token] = resampled_data

        D[token] = data

        B[token] = np.ones(data.shape[0],dtype='int')*block


        if rank==0:
            toctoc = time.time()-tictic
            print('\n-->Data matrix %i/%i done in %.2f seconds'%(count,len(tokens),toctoc))
            print '\n-->Estimated remaining time: %.2f seconds'%(toctoc*(len(tokens)-count))
            count+=1

    if rank==0:
        toc = time.time()-tic
        print '\nData generated succesfully in %.2f seconds'%toc

    return (start_times, stop_times, D, B)

def save_hdf5(fname, D, tokens, anat, B):
    """
    Save processed data to hdf5.

    Parameters
    ----------
    fname : str
        Path to save output.
    D : dict
        Dictionary containing data for each token.
    tokens : list of str
        Tokens to save from D.
    """
    tokens = sorted(tokens)
    labels = np.array(range(len(tokens)))
    X = None
    y = None
    bn = None
    for label, token in zip(labels, tokens):
        X_t = D[token]
        if X is None:
            X = X_t
        else:
            X = np.vstack((X, X_t))
        if y is None:
            y = label * np.ones(X_t.shape[0], dtype=int)
        else:
            y = np.hstack((y, label * np.ones(X_t.shape[0], dtype=int)))

        bn_t = B[token]
        if bn is None:
           bn = bn_t
        else:
           bn = np.hstack((bn,bn_t))

    folder, f = os.path.split(fname)

    try:
        os.mkdir(folder)
    except OSError:
        pass

    with h5py.File(fname, 'w') as f:
        f.create_dataset('X', data=X.astype('float32'))
        f.create_dataset('y', data=y)
        f.create_dataset('block', data=bn)
        f.create_dataset('tokens', data=tokens)
        grp = f.create_group('anatomy')
        for key in sorted(anat.keys()):
            grp.create_dataset(key, data=anat[key])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess ECoG Data')
    parser.add_argument('path', help='path to subject folder')
    parser.add_argument('blocks', nargs='+', type=int)
    parser.add_argument('-o', '--output_folder', default=None)
    parser.add_argument('-t', '--task', default='CV')
    parser.add_argument('-w', '--align_window', nargs=2, type=float, default=[-.5, .79])
    parser.add_argument('-p', '--align_pos', type=int, default=1)
    parser.add_argument('-d', '--data_type', default='HG')
    parser.add_argument('-b', '--zscore', default='events')
    parser.add_argument('-m', '--mp', default=True)
    parser.add_argument('-f', '--fband', type=int, default=18)
    args = parser.parse_args()
    pdb.set_trace()
    htk_to_hdf5(args.path, args.blocks, args.output_folder, args.task,
                args.align_window, args.align_pos, args.data_type, args.zscore,
                args.fband, args.mp)
