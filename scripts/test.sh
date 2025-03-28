#!/bin/zsh
python train-deep-taxonnet.py \
--batch_size=128 \
--epochs=100 \
--linear_probing_epochs=10 \
--wandb_run_name=samll_test \
--n_layers=3 \
--model_save_path=test \
--device=1