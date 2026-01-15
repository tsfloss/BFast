#!/bin/bash
#SBATCH --nodes=3                
#SBATCH --ntasks-per-node=4     
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=BFast_dist.out
#SBATCH --account=L-AUT_025
#SBATCH --gpus-per-node=4
#SBATCH --mem=96G
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg

nvidia-smi

source ~/.venv/main/bin/activate
srun python BFast_dist.py