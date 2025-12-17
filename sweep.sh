#!/bin/bash
#SBATCH --job-name=ser_sweep
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

cd $SLURM_SUBMIT_DIR

source /home/users/nblanco/miniconda3/etc/profile.d/conda.sh
conda activate emotionenv

python run_sweep.py \
  --data-root /home/users/nblanco/data \
  --epochs 40 \
  --outdir sweep_configs 