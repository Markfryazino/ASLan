#!/bin/bash
#SBATCH --job-name=filtration_exp-100
#SBATCH --error=./filtration_exp-100.err
#SBATCH --output=./filtration_exp-100.log
#SBATCH --time=10-0:0
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mail-user=rofinmp@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --array=0-2

module purge
module load Python/Anaconda_v11.2020

source deactivate
source activate nlp

OUT_SIZES=(30 50 100)

declare -a idxs=(0 1 2 3 4)
for idx in "${idxs[@]}"
do
    python ./filtration_exp.py --size 10 --state $idx --out_size ${OUT_SIZES[$SLURM_ARRAY_TASK_ID]} --fake_path gen_results-size-10-hwu64
done