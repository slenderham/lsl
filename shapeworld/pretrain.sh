#!/bin/bash

HYPO_LAMBDA=1

python lsl/pretrain.py --cuda \
    --batch_size 32 \
    --seed $RANDOM \
    --log_interval 10\
    --backbone resnet18\
    --data_dir /data/cw9951/hard_sw\
    exp/contrastive