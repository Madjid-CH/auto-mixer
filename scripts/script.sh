#!/bin/sh
#SBATCH --job-name=mm_imdb
#SBATCH --time=100:00:00
#SBATCH --partition=volta
#SBATCH --gres=gpu:1
#SBATCH --output=./out_mm_imdb.log
#SBATCH --error=./err_mm_imdb.log

# Define environment Conda and install packages
srun python run.py