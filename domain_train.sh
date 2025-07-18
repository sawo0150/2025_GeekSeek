#!/bin/bash



python domain_train.py \
  --domain-groups '[["knee_x4","knee_x8"],["brain_x4", "brain_x8"]]' \
  --config-paths 'configs/domain.yaml,configs/domain.yaml' \
  --epochs-per-block 1 \
  --total-epochs 30 \
  --wandb-project fastmri_domain_train \
  --result-root /home/swpants05/Desktop/2025_FastMri/result/
