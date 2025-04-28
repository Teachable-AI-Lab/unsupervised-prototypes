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

python train-deep-taxonnet.py --config scripts/configs/config-fmnist-0.json
python train-deep-taxonnet.py --config scripts/configs/config-fmnist-1.json
python train-deep-taxonnet.py --config scripts/configs/config-fmnist-2.json
python train-deep-taxonnet.py --config scripts/configs/config-fmnist-3.json
python train-deep-taxonnet.py --config scripts/configs/config-fmnist-4.json
python train-deep-taxonnet.py --config scripts/configs/config-fmnist-5.json
python train-deep-taxonnet.py --config scripts/configs/config-fmnist-6.json
python train-deep-taxonnet.py --config scripts/configs/config-fmnist-7.json
python train-deep-taxonnet.py --config scripts/configs/config-fmnist-8.json
python train-deep-taxonnet.py --config scripts/configs/config-fmnist-9.json
python train-deep-taxonnet.py --config scripts/configs/config-fmnist-10.json
python train-deep-taxonnet.py --config scripts/configs/config-fmnist-11.json
python train-deep-taxonnet.py --config scripts/configs/config-fmnist-12.json
python train-deep-taxonnet.py --config scripts/configs/config-fmnist-13.json
python train-deep-taxonnet.py --config scripts/configs/config-fmnist-14.json
python train-deep-taxonnet.py --config scripts/configs/config-fmnist-15.json
python train-deep-taxonnet.py --config scripts/configs/config-fmnist-16.json
python train-deep-taxonnet.py --config scripts/configs/config-fmnist-17.json
python train-deep-taxonnet.py --config scripts/configs/config-fmnist-18.json
python train-deep-taxonnet.py --config scripts/configs/config-fmnist-19.json

