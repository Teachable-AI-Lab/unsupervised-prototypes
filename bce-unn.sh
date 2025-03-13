#!/bin/zsh
python train-deep-taxonnet.py \
--batch_size=128 \
--epochs=500 \
--linear_probing_epochs=50 \
--tau=0.5 \
--commitment_weight=0.25 \
--wandb=True \
--wandb_run_name=bce-unnormalized \
--n_layers=8 \
--model_save_path=bce-unnormalized \
--loss_fn=bce \
--device=1