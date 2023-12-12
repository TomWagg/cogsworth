#!/bin/bash
## Job Name
#SBATCH --job-name=sn_dists
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=0:30:00
#SBATCH -o /mnt/home/twagg/cogsworth/FIRE/logs_%a_%A.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twagg@flatironinstitute.org
#SBATCH --export=all

source /mnt/home/twagg/.bashrc
conda activate cogsworth

echo "Getting supernova distances"

python /mnt/home/twagg/cogsworth/FIRE/get_supernova_distances.py
