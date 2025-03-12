#!/bin/zsh
python train-deep-taxonnet.py \
--batch_size=128 \
--epochs=500 \
--linear_probing_epochs=50 \
--wandb=True \
--wandb_run_name=mse-unnormalized \
--n_layers=8 \
--model_save_path=mse-unnormalized \
--loss_fn=mse \
--device=3