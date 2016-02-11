#!/bin/bash -l
python -u read_chang_data.py /project/projectdirs/m2043/BRAINdata/Humans/EC2 1 8 9 15 76 89 105 --zscore 'event' --output_folder $SCRATCH/output
