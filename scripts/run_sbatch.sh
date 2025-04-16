#!/bin/bash
#SBATCH --partition=tail-lab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a40
#SBATCH --mem-per-gpu="8GB"
#SBATCH --qos="short"
#SBATCH --exclude="spot"
conda init
conda activate dl
cd /nethome/zwang910/research/unsupervised-prototypes

python train-deep-taxonnet.py \
  --batch_size=128 \
  --epochs=500 \
  --linear_probing_epochs=50 \
  --wandb=true \
  --wandb_run_name=vade-6-vae-pt-100 \
  --n_layers=6 \
  --model_save_path=vade-6-vae-pt-100 \
  --device_id=0 \
  --pretraining_epochs=100 \
  --kl1_weight=1.0 \
