#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH -t 24:00:00
#SBATCH --mem=0
#SBATCH --job-name=tpot_h20_on_digen
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH -p defq


source /home/ribeirop/common/minconda3/etc/profile.d/conda.sh
module load java/jre1.8.0_341
conda activate tpot_digen_env_final

python run_tests_automl.py \
--njobs 48 \
--savepath 'AutoML_Results_HPC' \
--localcachedir 'Datasets' \
--num_runs 10 \