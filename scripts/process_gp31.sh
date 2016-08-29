#!/bin/bash -l
python -u read_chang_data.py /project/projectdirs/m2043/BRAINdata/Humans/GP31 1 2 4 6 9 21 63 65 67 69 71 78 82 83 --zscore 'events' --output_folder $SCRATCH/output
