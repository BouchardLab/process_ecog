import h5py, os
import numpy as np

from ..utils import bands


__all__ = ['restrict_to_good_vsmc']

def restrict_to_good_vsmc(file_name, remove_per_sample_baseline=False):
    neuro_bands = bands.neuro['bands']

    # Load data
    band_X = dict()
    with h5py.File(file_name, 'r') as f:
        for b in neuro_bands:
            band_X[b] = f['X{}'.format(b)][:]
        y = f['y'][:]
        tokens = f['tokens'][:]
        anatomy = f['anatomy'][:]
        sampling_freqs = f['sampling_freqs'].value

    # Select vSMC
    precentral = np.array([ai == 'precentral' for ai in anatomy)
    postcentral = np.array([ai == 'postcentral' for ai in anatomy)
    vsmc = np.logical_or(precentral, postcentral)
    vsmc = np.nonzero(vsmc)[0]

    # Get good examples and electrodes (not NaN)
    # First find any examples where all data is NaN
    good_examples = None
    for b in neuro_bands:
        band_ex = []
        d = band_X[b]
        for ii, ex in enumerate(d):
            if not np.isnan(ex).all():
                band_ex.append(ii)
        if good_examples is None:
            good_examples = set(band_ex)
        else:
            good_examples = good_examples.intersection(set(band_ex))

    # And remove them
    ex_filtered = dict()
    for b in neuro_bands:
        d = band_X[b]
        ex_filtered[b] = d[sorted(good_examples)]
    y_ex_filtered = y[sorted(good_examples)]

    # Then find vsmc channels that have NaNs
    good_vsmc = set(vsmc)
    for b in neuro_bands:
        d = ex_filtered[b]
        good_vsmc = good_vsmc.intersection(np.nonzero(1-np.isnan(d.sum(axis=(0, 2))).astype(int))[0])
    good_vsmc = good_vsmc.intersection(vsmc)

    # And remove them
    vsmc_filtered = dict()
    for b in neuro_bands:
        d = ex_filtered[b]
        vsmc_filtered[b] = d[:, sorted(good_vsmc), :]

    n_batch = None
    n_elect = None
    time = dict()
    for b in neuro_bands:
        if n_batch is None:
            n_batch = vsmc_filtered[b].shape[0]
        else:
            assert n_batch == vsmc_filtered[b].shape[0]
        if n_elect is None:
            n_elect = vsmc_filtered[b].shape[1]
        else:
            assert n_elect == vsmc_filtered[b].shape[1]
        time[b] = vsmc_filtered[b].shape[2]

    folder, f_name = os.path.split(file_name)
    name = '.'.join(f_name.split('.')[:-1])
    new_f_name = '{}_vsmc.h5'.format(name)
    save_data = vsmc_filtered
    if remove_per_sample_baseline:
        new_f_name = '{}_vsmc_nobaseline.h5'.format(name)
        n_avg_hg = 10
        base_time = time[neuro_bands[-1]]
        debaselined = dict()
        for b in neuro_bands:
            n_avg = max(1, int(np.around(n_avg_hg*time[b]/base_time)))
            d = vsmc_filtered[b]
            start_avg = d[:, :, :n_avg].mean(axis=-1, keepdims=True)
            end_avg = d[:, :, -n_avg:].mean(axis=-1, keepdims=True)
            mean = (start_avg+end_avg)/2.
            debaselined[b] = d - mean
        save_data = debaselined

    # Save file
    save_path = os.path.join(folder, new_f_name)
    tmp_path = '{}_tmp'.format(save_path)
    with h5py.File(tmp_path, 'w') as f:
        for b in neuro_bands:
            f.create_dataset('X{}'.format(b), data=debaselined[b])
        f.create_dataset('y', data=y_ex_filtered)

    os.rename(tmp_path, save_path)
