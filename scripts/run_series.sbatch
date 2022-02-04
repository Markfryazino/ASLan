#!/bin/bash
#SBATCH --job-name=knn_series
#SBATCH --error=./knn_series.err
#SBATCH --output=./knn_series.log
#SBATCH --time=5-0:0
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=type_e
#SBATCH --mail-user=rofinmp@gmail.com
#SBATCH --mail-type=ALL

module purge
module load Python/Anaconda_v11.2020

source deactivate
source activate nlp

python ./knn_few_shot_series.py
