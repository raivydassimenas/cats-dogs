#!/bin/bash
#SBATCH --job-name=pytorch_job            # Job name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Total number of tasks
#SBATCH --cpus-per-task=4                 # Number of CPU cores per task
#SBATCH --gres=gpu:4                      # Number of GPUs per node
#SBATCH --mem=32G                        # Memory per node
#SBATCH --time=12:00:00                   # Max time
#SBATCH --output=slurm-%j.out             # Output file

# Load modules or activate conda environment
module load anaconda
source activate pytorch_env

# Run the PyTorch distributed training script
python -m torch.distributed.launch --nproc_per_node=4 main.py
