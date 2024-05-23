#!/bin/sh
#SBATCH --job-name=mm_imdb
#SBATCH --time=100:00:00
#SBATCH --partition=volta
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=10
#SBATCH --output=./out_mm_imdb.log
#SBATCH --error=./err_mm_imdb.log
#SBATCH --mail-type=ALL # (BEGIN, END, FAIL or ALL)
#SBATCH --mail-user=a.chergui@esi-sba.dz

# Define environment Conda and install packages
srun python run.py