#!/bin/bash -l
python -u read_chang_data.py /project/projectdirs/m2043/BRAINdata/Humans/GP33 1 5 30 --zscore 'events' --output_folder $SCRATCH/output
