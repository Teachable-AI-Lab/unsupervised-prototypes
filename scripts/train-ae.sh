#!/bin/zsh
python train-autoencoder.py \
--batch_size=128 \
--epochs=500 \
--linear_probing_epochs=50 \
--tau=0.3 \
--commitment_weight=0.5 \
--wandb=true \
--wandb_run_name=AE-baseline \
--n_layers=8 \
--model_save_path=mse-tree-loss \
--loss_fn=mse \
--device=3