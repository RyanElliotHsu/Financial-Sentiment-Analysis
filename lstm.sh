#!/bin/bash
#SBATCH -n 30
#SBATCH -t 30:00:00
#SBATCH --mem=100G
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=reh424@nyu.edu

source ~/.bashrc
# conda init bash
# conda shell.bash hook
# conda activate /scratch/reh424/conda-envs/lstm

#Your application commands go here
python lstm.py