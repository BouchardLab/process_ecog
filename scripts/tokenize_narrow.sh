#!/bin/bash -l
#SBATCH -M escori
#SBATCH -q bigmem
##SBATCH -q debug
##SBATCH -q premium
#SBATCH -N 1
##SBATCH -t 00:30:00
#SBATCH -t 10:00:00
#SBATCH -J tokenize_data
#SBATCH -o tokenize_output.o%j
#SBATCH --mem=250GB
##SBATCH -C haswell


if [ "$NERSC_HOST" == "edison" ]
then
  cores=24
fi
if [ "$NERSC_HOST" == "cori" ]
then
  cores=32
fi

export PATH="$HOME/anaconda/bin:$PATH"
#source activate cv_process

echo $(which python)
echo $PATH

export MKL_NUM_THREADS=$cores
export OMP_NUM_THREADS=$cores


#srun -N 1 -n 1 -c "$cores" python -u tokenize_data.py /project/projectdirs/m2043/jlivezey/EC2 1 8 9 15 76 89 105 --zscore 'none' --data_type AA_ff --align_pos 1 --output_folder $SCRATCH/output &
#srun -N 1 -n 1 -c "$cores" python -u tokenize_data.py /project/projectdirs/m2043/jlivezey/EC9 15 39 46 49 53 60 63 --zscore 'none' --data_type AA_ff --align_pos 1 --output_folder $SCRATCH/output &
srun -N 1 -n 1 -c "$cores" python -u tokenize_data.py /project/projectdirs/m2043/jlivezey/GP31 1 2 4 6 9 21 63 65 67 69 71 78 82 83 --zscore 'none' --data_type AA_ff --align_pos 1 --output_folder $SCRATCH/output &
#srun -N 1 -n 1 -c "$cores" python -u tokenize_data.py /project/projectdirs/m2043/jlivezey/GP33 1 5 30 --zscore 'none' --data_type AA_ff --align_pos 1 --output_folder $SCRATCH/output &

# bigmem
#srun -N 1 -n 1 python -u tokenize_data.py /project/projectdirs/m2043/jlivezey/GP31 1 2 4 6 9 21 63 65 67 69 71 78 82 83 --zscore 'none' --data_type AA_ff --align_pos 1 --output_folder $SCRATCH/output2 &

# split for ff
#srun -N 1 -n 1 -c "$cores" python -u tokenize_data.py /project/projectdirs/m2043/jlivezey/GP31 1 2 4 6 9 21 63 65 --zscore 'none' --data_type AA_ff --align_pos 1 --output_folder $SCRATCH/output &
#srun -N 1 -n 1 -c "$cores" python -u tokenize_data.py /project/projectdirs/m2043/jlivezey/GP31 67 69 71 78 82 83 --zscore 'None' --data_type AA_ff --align_pos 1 --output_folder $SCRATCH/output &
wait
