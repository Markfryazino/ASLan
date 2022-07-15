#!/bin/bash
#SBATCH --job-name=hwu_100
#SBATCH --error=./hwu_100-%a.err
#SBATCH --output=./hwu_100-%a.log
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --array=0-5

module purge
module load Python/Anaconda_v11.2020

source deactivate
source activate nlp

MODELS=(roberta-base bert-base-uncased distilroberta-base distilbert-base-uncased funnel-transformer-small-base mobilebert-uncased)

declare -a idxs=(0 1 2 3 4)
for idx in "${idxs[@]}"
do
    python ./classification_exp.py --dataset hwu64 --fake_path filtered-size-10to100-hwu64 --size 10 --state $idx --model "../data/${MODELS[$SLURM_ARRAY_TASK_ID]}"
done