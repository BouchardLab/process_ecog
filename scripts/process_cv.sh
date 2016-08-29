#!/bin/bash -l
#SBATCH -p regular
#SBATCH -N 1
#SBATCH -t 05:00:00
#SBATCH -J process_data
#SBATCH -o process_output.o%j

module load python
module load h5py
#module load cython
#module swap numpy numpy/1.9.0_mkl
#module swap scipy scipy/0.14.0_mkl

export PYTHONPATH="/global/homes/j/jlivezey/pandas:$PYTHONPATH"

#srun -N 1 -n 1 -c 32 ./scripts/process_ec2.sh &
#srun -N 1 -n 1 -c 32 ./scripts/process_ec9.sh &
#srun -N 1 -n 1 -c 32 ./scripts/process_gp31.sh &
srun -N 1 -n 1 -c 24 ./scripts/process_gp33.sh &
wait
