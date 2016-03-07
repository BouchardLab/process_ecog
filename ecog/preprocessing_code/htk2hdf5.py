import numpy as np
import h5py
import HTK_hilb
import argparse

def htk_to_hdf5(block):

    subject = 'GP31'

    path  = '/global/project/projectdirs/m2043/BRAINdata/Humans/tmp'

    parts = ['HilbImag_4to200_40band','HilbReal_4to200_40band','HilbAA_70to150_8band']

    parts = ['HilbImag_4to200_40band','HilbReal_4to200_40band','HilbAA_70to150_8band']

    print '\nConverting HTK to HDF5 -- subject: %s'%subject

    for part in parts:

        htk = HTK_hilb.readHTKs('%s/%s/%s_B%i/%s'%(path,subject,subject,block,part))

        for i in xrange(htk['data'].shape[0]):

            print '\n\tBlock: B_%i; Part: %s; Freq_band: %i'%(block,part,i)

            with h5py.File('%s/%s/%s_B%i/%s_%i.h5'%(path,subject,subject,block,part,i),'w') as f:

                f.create_dataset(name='data',data=htk['data'][i].astype('f4'),\
                                 compression='gzip')

                f.create_dataset(name='sampling_rate',data=np.array([htk['sampling_rate']/1e4]).astype('f4'),\
                                 compression='gzip')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HTK to HDF5')
    parser.add_argument('-b', '--block', type=int, default=1)
    args = parser.parse_args()
    htk_to_hdf5(args.block)
