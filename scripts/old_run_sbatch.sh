#!/bin/bash
#SBATCH --partition=tail-lab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a40
#SBATCH --mem-per-gpu="8GB"
#SBATCH --qos="short"
#SBATCH --exclude="spot,heistotron"
source ~/.bashrc
conda init
conda activate dl
cd /nethome/zwang910/research/unsupervised-prototypes

python train-deep-taxonnet.py \
  --batch_size=128 \
  --epochs=300 \
  --linear_probing_epochs=50 \
  --wandb=true \
  --wandb_run_name=vade-10-rec-5-reg-1.2-fmnist-novar \
  --n_layers=10 \
  --model_save_path=vade-10-rec-5-reg-1.2-fmnist-novar \
  --device_id=0 \
  --pretraining_epochs=0 \
  --kl1_weight=1 \
  --recon_weight=5 \
  --dkl_margin=1.2 \
