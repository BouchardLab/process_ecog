#!/bin/bash -l
#SBATCH -p regular
#SBATCH --qos=premium
#SBATCH -N 3
#SBATCH -t 02:00:00
#SBATCH -J tokenize_data
#SBATCH -o tokenize_output.o%j


if [ "$NERSC_HOST" == "edison" ]
then
  cores=24
fi
if [ "$NERSC_HOST" == "cori" ]
then
  cores=32
fi

export PATH="$HOME/anaconda/bin:$PATH"

echo $(which python)
echo $PATH


srun -N 1 -n 1 -c "$cores" python -u tokenize_data.py /project/projectdirs/m2043/jlivezey/EC2 1 8 9 15 76 89 105 --zscore 'file' --data_type AA_avg --align_pos 1 --output_folder $SCRATCH/output &
#srun -N 1 -n 1 -c "$cores" python -u tokenize_data.py /project/projectdirs/m2043/jlivezey/EC9 15 39 46 49 53 60 63 --zscore 'file' --data_type AA_avg --align_pos 1 --output_folder $SCRATCH/output &
srun -N 1 -n 1 -c "$cores" python -u tokenize_data.py /project/projectdirs/m2043/jlivezey/GP31 1 2 4 6 9 21 63 65 67 69 71 78 82 83 --zscore 'file' --data_type AA_avg --align_pos 1 --output_folder $SCRATCH/output &
srun -N 1 -n 1 -c "$cores" python -u tokenize_data.py /project/projectdirs/m2043/jlivezey/GP33 1 5 30 --zscore 'file' --data_type AA_avg --align_pos 1 --output_folder $SCRATCH/output &
wait
