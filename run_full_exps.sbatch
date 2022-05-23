#!/bin/bash
#SBATCH --job-name=full_exps
#SBATCH --error=./full_exps.err
#SBATCH --output=./full_exps.log
#SBATCH --time=10-0:0
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=type_e
#SBATCH --mail-user=rofinmp@gmail.com
#SBATCH --mail-type=ALL

module purge
module load Python/Anaconda_v11.2020

source deactivate
source activate nlp

python ./full_exps.py
