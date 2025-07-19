#!/bin/bash



python domain_train.py \
  --domain-groups '[["brain_x4", "brain_x8"], ["knee_x4","knee_x8"]]' \
  --config-paths 'configs/domain_cluster.yaml,configs/domain_cluster.yaml' \
  --epochs-per-block 30 \
  --total-epochs 30 \
  --wandb-project AMP_sweep \
  --result-root /home/introai7/.bin/.fmri/result/
