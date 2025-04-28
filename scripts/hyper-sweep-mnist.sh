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

python train-deep-taxonnet.py --config scripts/configs/config-mnist-0.json
python train-deep-taxonnet.py --config scripts/configs/config-mnist-1.json
python train-deep-taxonnet.py --config scripts/configs/config-mnist-2.json
python train-deep-taxonnet.py --config scripts/configs/config-mnist-3.json
python train-deep-taxonnet.py --config scripts/configs/config-mnist-4.json
python train-deep-taxonnet.py --config scripts/configs/config-mnist-5.json
python train-deep-taxonnet.py --config scripts/configs/config-mnist-6.json
python train-deep-taxonnet.py --config scripts/configs/config-mnist-7.json
python train-deep-taxonnet.py --config scripts/configs/config-mnist-8.json
python train-deep-taxonnet.py --config scripts/configs/config-mnist-9.json
python train-deep-taxonnet.py --config scripts/configs/config-mnist-10.json
python train-deep-taxonnet.py --config scripts/configs/config-mnist-11.json
python train-deep-taxonnet.py --config scripts/configs/config-mnist-12.json
python train-deep-taxonnet.py --config scripts/configs/config-mnist-13.json
python train-deep-taxonnet.py --config scripts/configs/config-mnist-14.json
python train-deep-taxonnet.py --config scripts/configs/config-mnist-15.json
python train-deep-taxonnet.py --config scripts/configs/config-mnist-16.json
python train-deep-taxonnet.py --config scripts/configs/config-mnist-17.json
python train-deep-taxonnet.py --config scripts/configs/config-mnist-18.json
python train-deep-taxonnet.py --config scripts/configs/config-mnist-19.json

