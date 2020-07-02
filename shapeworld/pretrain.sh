#!/bin/bash

HYPO_LAMBDA=20

python lsl/pretrain.py --cuda \
    --batch_size 32 \
    --seed $RANDOM \
    --log_interval 10\
    --debug_example\
    exp/contrastive
