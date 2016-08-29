import numpy as np
import h5py
import .HTK
import argparse

def htk_to_hdf5(block, subject, path):

    parts = ['HilbImag_4to200_40band','HilbReal_4to200_40band','HilbAA_70to150_8band']

    print('\nConverting HTK to HDF5 -- subject: {}'.format(subject))

    for part in parts:
        htk = HTK_hilb.readHTKs(os.path.join(path, subject,
                                             '{}_B{}'.format(subject, block),
                                             part))
        for i in xrange(htk['data'].shape[0]):
            print('\n\tBlock: B_{}; Part: {}; Freq_band: {}'.format(block, part, i))
            fname = os.path.join(path, subject,
                                 '{}_B{}'.format(subject, block),
                                 '{}_{}.h5'.format(part, i))
            with h5py.File(fname, 'w') as f:
                f.create_dataset(name='data', data=htk['data'][i].astype('f4'),
                                 compression='gzip')
                f.create_dataset(name='sampling_rate',
                                 data=np.array([htk['sampling_rate']/1e4]).astype('f4'),
                                 compression='gzip')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HTK to HDF5')
    parser.add_argument('-b', '--block', type=int, default=1)
    parser.add_argument('-p','--path', help='path to data folder')
    parser.add_argument('-s','--subject', help='subject code')
    args = parser.parse_args()
    htk_to_hdf5(args.block, args.subject, args.path)
