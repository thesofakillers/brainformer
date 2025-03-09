#!/bin/bash

# Exit on error
set -e

# Print commands before executing them
set -x

# Create directories if they don't exist
mkdir -p checkpoints

# Set number of GPUs to use
NUM_GPUS=1  # Adjust based on your available hardware

# Run the training script with torchrun and Muon optimizer
python train.py \
  --data_dir eeg2meg_data \
  --input_channels 70 \
  --output_channels 306 \
  --seq_len 256 \
  --batch_size 1 \
  --split_data \
  --val_ratio 0.2 \
  --input_file eeg2meg_inputs.pt \
  --output_file eeg2meg_outputs.pt \
  --patch_size 4 \
  --lr 0.0001 \
  --weight_decay 0.6 \
  --epochs 100 \
  --save_dir checkpoints \
  --save_every 10 \
  --log_every 10 \
  --num_heads 2 \
  --num_enc_layers 4 \
  --num_dec_layers 4 \
  --wandb_watch=all  # wandb params
echo "EEG2MEG training with Muon optimizer completed successfully!" 
