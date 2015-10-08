#!/usr/bin/env python
import sys, os
import numpy as np

folder = sys.argv[1]
prepend = os.path.join(os.getcwd(), folder)
files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(prepend, f))]

files = [f for f in files if '.out' in f]
error = np.zeros((3,len(files)))
std = np.zeros((3,len(files)))

for ii, f in enumerate(files):
    with open(os.path.join(prepend, f), 'r') as fh:
        lines = fh.readlines()
        if 'Got result' in lines[-2]:
            #print lines[-12].split(' ')
            error[0,ii] = float(lines[-12].split(' ')[-1])
            error[1,ii] = float(lines[-11].split(' ')[-1])
            error[2,ii] = float(lines[-10].split(' ')[-1])
            std[0,ii] = float(lines[-8].split(' ')[-1])
            std[1,ii] = float(lines[-7].split(' ')[-1])
            std[2,ii] = float(lines[-6].split(' ')[-1])
max_idx = error[1].argmax()
print os.path.join(folder,files[error.argmax()])
print 'train: ', error[0, max_idx], std[0, max_idx]
print 'valid: ', error[1, max_idx], std[1, max_idx]
print 'test: ', error[2, max_idx], std[2, max_idx]
