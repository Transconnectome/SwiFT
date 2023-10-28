#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --nodelist node1
#SBATCH -c 32
#SBATCH --ntasks-per-node=1

source /usr/anaconda3/etc/profile.d/conda.sh
conda activate py39
pip install monai

python preprocessing-ABCD-rest.py

