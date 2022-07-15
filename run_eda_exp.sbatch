#!/bin/bash
#SBATCH --job-name=eda_exp
#SBATCH --error=./eda_exp.err
#SBATCH --output=./eda_exp.log
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3

module purge
module load Python/Anaconda_v11.2020

source deactivate
source activate nlp

INFLATIONS=(3 5 10 20)

declare -a idxs=(0 1 2 3 4)
for idx in "${idxs[@]}"
do
    python ./eda_exp.py --size 10 --state $idx --inflation ${INFLATIONS[$SLURM_ARRAY_TASK_ID]}
done