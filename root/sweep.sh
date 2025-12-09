#!/bin/bash
#SBATCH --job-name=ser_sweep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

source ~/.bashrc
conda activate emotionenv

python run_sweep.py \
  --data-root /Users/nicolleblanco/Ser-trash/data \
  --epochs 40 \
  --outdir sweep_configs