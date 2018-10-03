#!/bin/bash -l
#SBATCH --qos=premium
#SBATCH -N 14
#SBATCH -t 02:30:00
#SBATCH -J preprocess_data
#SBATCH -o preprocess_output.o%j


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

# 7 blocks
for b in 1 8 9 15 76 89 105; do
  srun -N 1 -n 1 -c "$cores" python -u preprocess_data.py /project/projectdirs/m2043/jlivezey EC2 "$b" &
  #srun -N 1 -n 1 -c "$cores" python -u preprocess_data.py /project/projectdirs/m2043/jlivezey EC2 "$b" -e &
done

# 7 blocks
for b in 15 39 46 49 53 60 63; do
  srun -N 1 -n 1 -c "$cores" python -u preprocess_data.py /project/projectdirs/m2043/jlivezey EC9 "$b" &
done

# 14 blocks
#for b in 1 2 4 6 9 21 63 65 67 69 71 78 82 83; do
#for b in 1 9 63 65 69 83; do
#  srun -N 1 -n 1 -c "$cores" python -u preprocess_data.py /project/projectdirs/m2043/jlivezey GP31 "$b" -n &
#done

# 3 blocks
#for b in 1 5 30; do
#  srun -N 1 -n 1 -c "$cores" python -u preprocess_data.py /project/projectdirs/m2043/jlivezey GP31 "$b" -n &
#done

wait
