#!/bin/bash
#SBATCH --job-name=hwu_test
#SBATCH --error=./hwu_test.err
#SBATCH --output=./hwu_test.log
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1

module purge
module load Python/Anaconda_v11.2020

source deactivate
source activate nlp

python ./classification_exp.py --dataset hwu64 --fake_path filtered-size-10to30-hwu64 --size 10 --state 1 --model ../data/distilroberta-base