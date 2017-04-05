#!/usr/bin/env python
import numpy as np
import h5py
from scipy.io import loadmat

def selectElectrodes(subject,blocks,vsmc=True):
    electrodes = loadmat('/project/projectdirs/m2043/BRAINdata/Humans/%s/anatomy/%s_anatomy.mat'%(subject,subject))
    if vsmc:
        elects = np.hstack([electrodes['anatomy']['preCG'][0][0][0],electrodes['anatomy']['postCG'][0][0][0]])-1
    else:
        elects = electrodes
    blocks = np.unique(blocks)
    for i in xrange(len(blocks)):
        file_path = '/project/projectdirs/m2043/BRAINdata/Humans/%s/%s_B%i/Artifacts/badChannels.txt'%(subject,subject,blocks[i])
        if i==0:
            badElects = np.loadtxt(file_path)-1
        else:
            badElects = np.hstack((badElects,np.loadtxt(file_path)-1))
    badElects = np.unique(badElects)
    elects = np.setdiff1d(elects,badElects)
    return elects
