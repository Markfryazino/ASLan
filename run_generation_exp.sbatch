#!/bin/bash
#SBATCH --job-name=gen-hwu-%a
#SBATCH --error=./gen-hwu-%a.err
#SBATCH --output=./gen-hwu-%a.log
#SBATCH --time=20-0:0
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mail-user=rofinmp@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --array=0-4

module purge
module load Python/Anaconda_v11.2020

source deactivate
source activate nlp

python ./generation_exp.py --size 10 --state $SLURM_ARRAY_TASK_ID