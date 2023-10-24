#!/bin/bash
## Job Name
#SBATCH --job-name=boundedness_cogsworth
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=2:00:00
#SBATCH -o /mnt/home/twagg/cogsworth/FIRE/logs_%a_%A.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twagg@flatironinstitute.org
#SBATCH --export=all

source /mnt/home/twagg/.bashrc
conda activate cogsworth

echo "Starting 10.0 run"

python /mnt/home/twagg/cogsworth/FIRE/fire_runner.py -a 10.0 -p 128
