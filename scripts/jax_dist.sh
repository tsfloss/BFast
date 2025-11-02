#!/bin/bash
#SBATCH --nodes=2                
#SBATCH --ntasks-per-node=4     
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=jax_dist.out
#SBATCH --account=L-AUT_025
#SBATCH --gpus-per-node=4
#SBATCH --mem=96G
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg

source ~/.venv/main/bin/activate
srun python jax_dist.py