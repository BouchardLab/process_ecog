import os

import numpy as np

from ecog.tokenize import transcripts


__all__ = ['load_bad_times']


def load_bad_times(block_path):
    """
    Load bad time segments.

    Parameters
    ----------
    block_path : str
        Path to block to load bad electrodes from.

    Returns
    -------
    bad_times : ndarray
        Pairs of start and stop times for bad segments.
    """

    bad_times = []
    try:
        lab_time_conversion = transcripts.lab_time_conversion
        with open(os.path.join(block_path, 'Artifacts',
                               'bad_time_segments.lab'), 'rt') as f:
            lines = f.readlines()
            for line in lines:
                if 'e' in line:
                    start, stop = line.split(' ')[:2]
                    bad_times.append([float(start)/lab_time_conversion,
                                      float(stop)/lab_time_conversion])
    except IOError:
        return np.array([])

    bad_times = np.array(bad_times)

    return bad_times
