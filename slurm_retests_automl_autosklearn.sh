#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH -t UNLIMITED
#SBATCH --mem=0
#SBATCH --job-name=autosk_on_digen
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH -p defq

source /home/ribeirop/common/minconda3/etc/profile.d/conda.sh

conda activate autosklearn_digen_env_final

python retest_on_larger_testset_autosklearn.py