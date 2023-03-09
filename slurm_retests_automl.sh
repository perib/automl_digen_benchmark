#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH -t 20:00:00
#SBATCH --mem=0
#SBATCH --job-name=tpot_h20_on_digen
#SBATCH --mail-type=FAIL,BEGIN,END

#SBATCH -p defq

#module load java/jre1.8.0_45
source /home/ribeirop/common/minconda3/etc/profile.d/conda.sh
module load java/jre1.8.0_341
conda activate tpot_digen_env_final

python retest_on_larger_testset.py