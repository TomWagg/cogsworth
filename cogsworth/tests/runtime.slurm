#!/bin/bash
## Job Name
#SBATCH --job-name=scaling
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=0:30:00
#SBATCH -o /mnt/home/twagg/runtime_%A.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tomwagg@uw.edu
#SBATCH --export=all

source /mnt/home/twagg/.bashrc
conda activate cogsworth

cd /mnt/home/twagg/cogsworth/cogsworth/tests

for processes in 1 2 4 8 16 32 64 128
do
    python runtime.py -p $processes -n 1000
done