from __future__ import division
import numpy as np
import h5py
import HTK_hilb as htk
reload(htk)
from pyfftw.interfaces.numpy_fft import fft,ifft,fftfreq
from scipy.io import loadmat
from optparse import OptionParser


import downSampleEcog as dse
reload(dse)
import subtractCAR as scar
reload(scar)
import applyLineNoiseNotch as notch
reload(notch)
import applyHilbertTransform as aht
reload(aht)
import deleteBadTimeSegments as dbts
reload(dbts)

__authors__ = "Alex Bujan (adapted from Kris Bouchard)"


def main():
    usage = '%prog [options]'

    parser = OptionParser(usage)

    parser.add_option("--subject",type="string",default='GP31',\
        help="Subject code")

    parser.add_option("--block",type="string",default='B1',\
        help="Block number eg: 'B1'")

    parser.add_option("--path",type="string",default='',\
        help="Path to the data")

    parser.add_option("--rate",type="float",default=400.,\
        help="Sampling rate of the processed signal (optional)")

    parser.add_option("--vsmc",action='store_true',\
        dest='vsmc',help="Include vSMC electrodes only (optional)")

    parser.add_option("--store",action='store_true',\
        dest='store',help="Store results (optional)")

    parser.add_option("--ct",type="float",default=87.75,\
        help="Center frequency of the Gaussian filter (optional)")

    parser.add_option("--sd",type="float",default=3.65,\
        help="Standard deviation of the Gaussian filter (optional)")

    parser.add_option("--srf",type="float",default=1.,\
        help="Sampling rate factor. Read notes in HTK.py (optional)")

    (options, args) = parser.parse_args()

    assert options.path!='',IOError('Inroduce a correct data path!')

    if options.vsmc:
        vsmc=True
    else:
        vsmc=False

    if options.store:
        store=True
    else:
        store=False

    transform(path=options.path,subject=options.subject,block=options.block,\
              rate=options.rate,vsmc=vsmc,ct=options.ct,sd=options.sd,\
              store=store,srf=options.srf)

def transform(path,subject,block,rate=400.,vsmc=True,\
              ct=87.75,sd=3.65,store=False,srf=1):

    b_path = '%s/%s/%s_%s'%(path,subject,subject,block)

    """
    Load raw HTK files
    """
    rd_path = '%s/RawHTK'%b_path
    HTKoutR = htk.readHTKs(rd_path)
    X = HTKoutR['data']

    """
    Downsample to 400 Hz
    """
    X = dse.downsampleEcog(X,rate,HTKoutR['sampling_rate']/srf)

    """
    Subtract CAR
    """
    X = scar.subtractCAR(X)

    """
    Select electrodes
    """
    electrodes = loadmat('%s/%s/Anatomy/%s_anatomy.mat'%(path,subject,subject))
    if vsmc:
        elects = np.hstack([electrodes['anatomy']['preCG'][0][0][0],\
                            electrodes['anatomy']['postCG'][0][0][0]])-1
    else:
        elecs = electrodes-1
    badElects = np.loadtxt('/%s/Artifacts/badChannels.txt'%b_path)-1
    elects = np.setdiff1d(elects,badElects)

    X = X[elects]

    """
    Discard bad segments
    """
    #TODO
#    badSgm = loadmat('%s/Artifacts/badTimeSegments.mat'%b_path)['badTimeSegments']
#    dbts.deleteBadTimeSegments(X,rate,badSgm)

    """
    Apply Notch filters
    """
    X = notch.applyLineNoiseNotch(X,rate)

    """
    Apply Hilbert transform
    """
    Xas = aht.applyHilbertTransform(X,rate,ct,sd)

    """
    Store the results
    """

    if store:
        with h5py.File('%s/pcsd_data/%s_%s_AS_%.1f_%.1f.h5'%(b_path,subject,block,ct,sd)) as f:
            f.attrs['sampling_rate'] = rate
            f.attrs['hilb_ct'] = ct
            f.attrs['hilb_sd'] = sd
            f.create_dataset(name='X',data=X,compression='gzip')
            f.create_dataset(name='X_imag',data=Xas.imag,compression='gzip')
            f.create_dataset(name='X_real',data=Xas.real,compression='gzip')
    else:
        return X,Xas


if __name__=='__main__':
    main()
