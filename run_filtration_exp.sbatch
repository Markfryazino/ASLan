#!/bin/bash
#SBATCH --job-name=filtration_exp
#SBATCH --error=./filtration_exp.err
#SBATCH --output=./filtration_exp.log
#SBATCH --time=10-0:0
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mail-user=rofinmp@gmail.com
#SBATCH --mail-type=ALL

module purge
module load Python/Anaconda_v11.2020

source deactivate
source activate nlp

declare -a idxs=(0 1 2 3 4)
for idx in "${idxs[@]}"
do
    python ./filtration_exp.py --size 10 --state $idx --out_size 30 --fake_path gen_results-size-10-banking77
done