#!/bin/zsh
python train-deep-taxonnet.py \
--batch_size=128 \
--epochs=500 \
--linear_probing_epochs=50 \
--wandb=true \
--wandb_run_name=vade-6 \
--n_layers=6 \
--model_save_path=vade-6 \
--device_id=0