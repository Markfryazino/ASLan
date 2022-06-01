#!/bin/bash
#SBATCH --job-name=cls_exp
#SBATCH --error=./cls_exp.err
#SBATCH --output=./cls_exp.log
#SBATCH --time=3-0:0
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1

module purge
module load Python/Anaconda_v11.2020

source deactivate
source activate nlp

declare -a idxs=(0 1 2 3 4)
for idx in "${idxs[@]}"
do
    python ./classification_exp.py --size 10 --state $idx --fake_path filtered-size-10to30 --model ../data/mobilebert-uncased
done