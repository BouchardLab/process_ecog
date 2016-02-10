from __future__ import division
import numpy as np
import h5py
import HTK
from pyfftw.interfaces.numpy_fft import fft,ifft,fftfreq
from scipy.io import loadmat

import downSampleEcog as dse
import subtractCAR as scar
import applyLineNoiseNotch as notch


subject = 'GP31'

block = 'GP31_B1'

path = '/global/project/projectdirs/m2043/BRAINdata/Humans/tmp'

def main(path,subject,block,rate=400.,vsmc_only=True):

    b_path = '%s/%s/%s'%(path,subject,block)

    """
    Load raw HTK files
    """
    rd_path = '%s/RawHTK'%b_path
    HTKoutR = HTK.readHTKs(rd_path)

    """
    Select electrodes
    """
    electrodes = loadmat('%s/Anatomy/%s_anatomy.mat'%(path,subject))
    if vsmc_only:
        elects = np.hstack([electrodes['anatomy']['preCG'][0][0][0],\
                            electrodes['anatomy']['postCG'][0][0][0]])-1
    else:
        elecs = electrodes-1
    badElects = loadtxt('/%s/Artifacts/badChannels.txt'%b_path)-1
    elects = np.setdiff1d(elects,badElects)

    X = HTKoutR['data'][elects]

    """
    Downsample to 400 Hz
    """
    X = dse.downsampleEcog(X,rate,HTKoutR['sampling_rate']/1e4)

    """
    Discard bad segments
    """
    #TODO
    badSgm = loadmat('%s/Artifacts/badTimeSegments.mat'%b_path)['badTimeSegments']

    """
    Subtract CAR
    """
    X = scar.subtractCAR(X)

    """
    Apply Notch filters
    """
    X = notch.applyLineNoiseNotch(X,rate)

    """
    Apply Hilbert transform
    """
    X = aht.applyHilbertTransform(X,rate,center=87.,sd=)




if __name__=='__main__':
    main()
