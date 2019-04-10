import numpy as np


__all__ = ['load_bad_times']


def load_bad_times(nwb):
    """
    Load bad time segments.

    Parameters
    ----------
    block_path : str
        Path to block to load bad electrodes from.

    Returns
    -------
    bad_times : ndarray, (n_windows, 2)
        Pairs of start and stop times for bad segments.
    """
    times = nwb.invalid_times
    if times is not None:
        start = nwb.invalid_times['start_time'].data[:]
        stop = nwb.invalid_times['stop_time'].data[:]
        bad_times = np.stack([start, stop], axis=1)
    else:
        bad_times = np.array([])
    return bad_times
