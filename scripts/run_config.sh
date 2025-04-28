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

python train-deep-taxonnet.py --config scripts/configs/default.json
