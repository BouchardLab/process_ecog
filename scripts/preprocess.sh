#!/bin/bash -l
#SBATCH -p regular
#SBATCH -N 4
#SBATCH -t 05:00:00
#SBATCH -J preprocess_data
#SBATCH -o preprocess_output.o%j

module load python
module load h5py

export PYTHONPATH="/global/homes/j/jlivezey/ecog:$PYTHONPATH"

srun -N 1 -n 1 -c 32 /global/homes/j/jlivezey/ecog/scripts/preprocess_ec2.sh &
srun -N 1 -n 1 -c 32 /global/homes/j/jlivezey/ecog/scripts/preprocess_ec9.sh &
srun -N 1 -n 1 -c 32 /global/homes/j/jlivezey/ecog/scripts/preprocess_gp31.sh &
srun -N 1 -n 1 -c 32 /global/homes/j/jlivezey/ecog/scripts/preprocess_gp33.sh &
wait
