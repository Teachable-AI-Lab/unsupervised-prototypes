#!/bin/zsh
python train-deep-taxonnet.py \
--batch_size=128 \
--epochs=500 \
--linear_probing_epochs=50 \
--tau=0.5 \
--commitment_weight=1.0 \
--wandb=True \
--wandb_run_name=mse-unnormalized-3x3 \
--n_layers=8 \
--model_save_path=mse-unnormalized-3x3 \
--loss_fn=mse \
--device=1