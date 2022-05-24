#!/bin/bash
#SBATCH --job-name=gen_exp
#SBATCH --error=./gen_exp.err
#SBATCH --output=./gen_exp.log
#SBATCH --time=10-0:0
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mail-user=rofinmp@gmail.com
#SBATCH --mail-type=ALL

module purge
module load Python/Anaconda_v11.2020

source deactivate
source activate nlp

declare -a idxs=(4)
for idx in "${idxs[@]}"
do
    python ./generation_exp.py --size 10 --state $idx
done