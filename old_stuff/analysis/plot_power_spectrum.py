#!/usr/bin/env python
import h5py, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

filename = sys.argv[1]

assert os.path.exists(filename)

with h5py.File(filename, 'r') as f:
    X = np.squeeze(f['X'].value)
    y = np.squeeze(f['y'].value).argmax(axis=1)

fft = np.fft.rfftn(X, axes=(1,))
power = np.absolute(fft)
freqs = np.fft.rfftfreq(X.shape[1], d=.005)
mean = power.mean(axis=(0,2))
std = power.std(axis=(0,2))

n_classes = y.max()+1
n_electrodes = X.shape[2]

def make_plots(freqs, mean, std, title, pp):
    print std.min()
    fig = plt.figure()
    plt.plot(freqs, mean, 'k', color='#CC4F1B')
    plt.fill_between(freqs, mean-std, mean+std,
                     alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title(str(title)+' Linear')
    pp.savefig(fig)
    plt.close(fig)

    fig = plt.figure()
    plt.loglog(freqs, mean, 'k', color='#CC4F1B')
    plt.fill_between(freqs, mean-std, mean+std,
                     alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title(str(title)+' LogLog')
    pp.savefig()
    plt.close(fig)

with PdfPages('plots.pdf') as pp:
    print('Combined')
    make_plots(freqs, mean, std, 'Combined CV+Electrode', pp)
    print('Classes')
    for n in range(n_classes):
        Xp = X[y == n]
        fft = np.fft.rfftn(Xp, axes=(1,))
        power = np.absolute(fft)
        freqs = np.fft.rfftfreq(Xp.shape[1], d=.005)
        mean = power.mean(axis=(0,2))
        std = power.std(axis=(0,2))
        make_plots(freqs, mean, std, 'CV: '+str(n)+', Combined Electrode', pp)
    print('Electrodes')
    for n in range(n_electrodes):
        Xp = X[..., n][..., np.newaxis]
        fft = np.fft.rfftn(Xp, axes=(1,))
        power = np.absolute(fft)
        freqs = np.fft.rfftfreq(Xp.shape[1], d=.005)
        mean = power.mean(axis=(0,2))
        std = power.std(axis=(0,2))
        make_plots(freqs, mean, std, 'Combined CV, Electrode: '+str(n), pp)

