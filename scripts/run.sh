#!/bin/zsh
python train-deep-taxonnet.py \
--batch_size=128 \
--epochs=500 \
--linear_probing_epochs=50 \
--tau=0.8 \
--commitment_weight=1 \
--wandb=true \
--wandb_run_name=mse-tree-loss-9 \
--n_layers=9 \
--model_save_path=mse-tree-loss-9 \
--loss_fn=mse \
--device=1