#!/bin/zsh
python train-deep-taxonnet.py \
--batch_size=128 \
--epochs=500 \
--linear_probing_epochs=50 \
--wandb=True \
--normalize=True \
--wandb_run_name=mse-normalized \
--n_layers=8 \
--model_save_path=mse-normalized \
--loss_fn=mse \
--device=2