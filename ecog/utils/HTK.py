# This code has been modified from David Moses' original version by Ben Dichter to be stand-alone

#-------------------------------------------------------------------------------
# Name:        HTK
# Purpose:     To interact with .htk files
#
# Author:      David Moses
#
# Created:     Dec 10, 2013
# Copyright:   (c) David Moses 2013
#-------------------------------------------------------------------------------

import os
import numpy as np


# In the MATLAB function named writeHTK, the sampling rate is multiplied by a
# scaling factor before being written to the file. The reason for doing this is
# unknown. That scaling factor is stored here as a static variable for
# conversion from what the file specifies as the sampling rate into the actual
# sampling rate (in Hz), assuming that the scaling factor was applied during
# the HTK file construction.
SAMPLING_RATE_FACTOR = 1e4

def read_HTK(file_path, data_type='HTK', big_endian=True,
             scale_s_rate=False):
    """
    Reads data from a HTK file

    Parameters
    ----------
    file_path : str
        The HTK file name
    data_type : str
        Specifies the data type (view getFilePath for
                               more details)
    big_endian : bool
        Specifies whether or not the data is big endian
        (if False, then the data is little endian)
    scale_s_rate : bool
        Specifies whether or note to divide the sampling
        rate specified by the HTK file by the sampling
        rate scaling factor

    Returns
    -------
    A dictionary containing the data from the file
    """

    # Creates the endian variable that will be used when specifying data types
    if big_endian:
        endian = '>'
    else:
        endian = '<'

    # Opens the HTK file
    with open(file_path, 'rb') as f:

        # Reads values from the header
        num_samples    = np.fromfile(f, dtype=endian+'i4', count=1)[0]
        sampling_rate  = np.fromfile(f, dtype=endian+'i4', count=1)[0]
        sample_size    = np.fromfile(f, dtype=endian+'i2', count=1)[0]
        parameter_kind = np.fromfile(f, dtype=endian+'i2', count=1)[0]

        # Reads the data
        data = np.fromfile(f, dtype=endian+'f4').reshape((num_samples, -1)).T

    # Scales the sampling rate (if desired)
    if scale_s_rate:
        sampling_rate = float(sampling_rate) / SAMPLING_RATE_FACTOR

    # Returns a dictionary containing the data from the file
    return {'num_samples':    num_samples,
            'sampling_rate':  float(sampling_rate),
            'sample_size':    sample_size,
            'parameter_kind': parameter_kind,
            'data':           data}

def write_HTK(file_data, file_path, data_type='HTK', big_endian=True,
              scale_s_rate=False):
    """
    Writes data to a HTK file.

    Parameters
    ----------
    file_data : dict
        A dictionary containing the following items:
            'num_samples'    : The number of samples
            'sampling_rate'  : The sampling rate (in Hz)
            'sample_size'    : Unknown
            'parameter_kind' : Unknown
            'data'           : The data points
        The sample size and parameter kind items are
        optional; the others are required.
    file_name : str
        The HTK file name
    data_type : str
        Specifies the data type (view getFilePath for
        more details)
    big_endian : bool
        Specifies whether or not the data is big endian
        (if False, then the data is little endian)
    scale_s_rate : bool
        Specifies whether or note to multiply the
        sampling rate specified by the HTK file by the
        sampling rate scaling factor
    """

    # Creates the endian variable that will be used when specifying data types
    if big_endian:
        endian = '>'
    else:
        endian = '<'

    # Obtains relevant values from the file data dictionary. If the sample
    # size or the parameter kind keys are not in the dictionary, then
    # arbitrary values are used.
    num_samples   = file_data['num_samples']
    sampling_rate = file_data['sampling_rate']
    data          = file_data['data']

    try:
        sample_size = file_data['sample_size']
    except KeyError:
        sample_size = 0

    try:
        parameter_kind = file_data['parameter_kind']
    except KeyError:
        parameter_kind = 8971

    # Scales the sampling rate (if desired)
    if scale_s_rate:
        sampling_rate = float(sampling_rate) * SAMPLING_RATE_FACTOR

    # Opens the HTK file
    with open(file_path, 'wb') as f:

        # Writes the header values
        np.array(num_samples,    dtype=endian+'i4').tofile(f)
        np.array(sampling_rate,  dtype=endian+'i4').tofile(f)
        np.array(sample_size,    dtype=endian+'i2').tofile(f)
        np.array(parameter_kind, dtype=endian+'i2').tofile(f)

        # Writes the data
        np.array(data, dtype=endian+'f4').T.tofile(f)

def read_HTKs(dir_path, electrodes=None, fbands=None):
    """
    Reads all of the HTK files in a directory, takes the mean,
    and outputs a matrix that contains all of the channels together

    Parameters
    ----------
    dir_path : str
        Directory that contains HTK files
    electrodes : list
        List of which electrodes to include. Default is all in dir

    Returns
    -------
    A dictionary containing the data from the directory
    """

    num_files = len(os.listdir(dir_path))
    if electrodes is None:
        electrodes = range(256)

    htkout = read_HTK(os.path.join(dir_path, 'Wav11.htk'))
    num_fbands, num_samples = htkout['data'].shape
    
    if fbands is None:
        alldata = np.zeros((num_fbands, len(electrodes), num_samples))
    else:
        alldata = np.zeros((len(fbands), len(electrodes), num_samples))

    for ifile in electrodes:
        htkout = read_HTK(os.path.join(dir_path,
                                       'Wav{}{}.htk'.format(int(np.ceil((ifile+1)/64.)),
                                                            np.mod(ifile, 64)+1)))
        if fbands is None:
            alldata[:,ifile,:] = htkout['data']
        else:
            alldata[:,ifile,:] = htkout['data'][fbands]

    return {'num_samples':    htkout['num_samples'],
            'sampling_rate':  float(htkout['sampling_rate']),
            'sample_size':    htkout['sample_size'],
            'parameter_kind': htkout['parameter_kind'],
            'data':           np.squeeze(alldata)}
