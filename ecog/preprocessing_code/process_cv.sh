#!/bin/bash -l
#SBATCH -p regular
#SBATCH -N 3
#SBATCH -t 04:00:00
#SBATCH -J process_data
#SBATCH -o process_output.o%j
# SBATCH --qos premium

module load python
module load h5py
#module load cython
#module swap numpy numpy/1.9.0_mkl
#module swap scipy scipy/0.14.0_mkl

export PYTHONPATH="/global/homes/j/jlivezey/pandas:$PYTHONPATH"

srun -n 24 -N 1 ./read_chang_data.py /project/projectdirs/m2043/BRAINdata/Humans/EC2 1 8 9 15 76 89 105 --zscore 'data' --output_folder $SCRATCH &
srun -n 24 -N 1 ./read_chang_data.py /project/projectdirs/m2043/BRAINdata/Humans/tmp/EC9 15 39 46 49 53 60 63 --zscore 'data' --output_folder $SCRATCH &
srun -n 24 -N 1 ./read_chang_data.py /project/projectdirs/m2043/BRAINdata/Humans/tmp/GP31 1 2 4 6 9 21 63 65 67 69 71 78 82 83 --zscore 'data' --output_folder $SCRATCH &
wait
