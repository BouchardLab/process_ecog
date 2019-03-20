from __future__ import print_function, division
import h5py, os, sys
import numpy as np
from scipy.io import loadmat

from ecog.utils import bands

file_name = sys.argv[1]
neuro_bands = bands.neuro['bands']

# Load data

band_X = dict()
with h5py.File(file_name, 'r') as f:
    for b in neuro_bands:
        band_X[b] = f['X{}'.format(b)].value.real
        print(b, band_X[b][np.logical_not(np.isnan(band_X[b]))].max())
    y = f['y'].value
    tokens = f['tokens'].value
    anatomy = dict()
    for key in f['anatomy'].keys():
        anatomy[key] = f['anatomy'][key].value-1
    sampling_freqs = f['sampling_freqs'].value

vsmc = np.hstack((anatomy['preCG'], anatomy['postCG']))

for ii, token in enumerate(tokens):
    print(token, (y == ii).astype(int).sum())


# Get good examples and electrods (not NaN)

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
print('good examples', len(good_examples), d.shape[0])

ex_filtered = dict()
for b in neuro_bands:
    d = band_X[b]
    ex_filtered[b] = d[sorted(good_examples)]
y_ex_filtered = y[sorted(good_examples)]

good_vsmc = set(vsmc)
for b in neuro_bands:
    d = ex_filtered[b]
    good_vsmc = good_vsmc.intersection(np.nonzero(1-np.isnan(d.sum(axis=(0, 2))).astype(int))[0])
good_vsmc = good_vsmc.intersection(vsmc)
print('good electrodes', len(good_vsmc))

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


vowels = sorted(list(set([label[-2:] for label in tokens])))
consonants = sorted(list(set([label[:-2] for label in tokens])))
vowel_map = {name: idx for idx, name in enumerate(vowels)}
consonant_map = {name: idx for idx, name in enumerate(consonants)}

vowel_labels = np.zeros(n_batch, dtype=y_ex_filtered.dtype)
consonant_labels = np.zeros(n_batch, dtype=y_ex_filtered.dtype)

for ii, y_label in enumerate(y_ex_filtered):
    vowel_labels[ii] = vowel_map[tokens[y_label][-2:]]
    consonant_labels[ii] = consonant_map[tokens[y_label][:-2]]

n_idx = len(tokens)
n_vowels = len(vowels)
n_consonants = len(consonants)

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

# Save file

folder, f_name = os.path.split(file_name)
name = '.'.join(f_name.split('.')[:-1])
new_f_name = ''.join((name, '_nobaseline', '.h5'))
with h5py.File(os.path.join(folder, new_f_name), 'w') as f:
    for b in neuro_bands:
        f.create_dataset('X{}'.format(b), data=debaselined[b])
    f.create_dataset('y', data=y_ex_filtered)
