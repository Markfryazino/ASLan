#!/bin/bash
#SBATCH --job-name=test
#SBATCH --error=./test.err
#SBATCH --output=./test.log
#SBATCH --time=5-0:0
#SBATCH --cpus-per-task=1
#SBATCH --gpus=0

module purge
module load Python/Anaconda_v11.2020

source deactivate
source activate nlp

python ./test_wandb.py
