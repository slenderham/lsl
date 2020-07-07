#!/bin/bash

HYPO_LAMBDA=1

python lsl/pretrain.py --cuda \
    --batch_size 64 \
    --seed $RANDOM \
    --log_interval 10\
    --backbone resnet18\
    --comparison cosine\
    --save_feats\
    --data_dir /Users/wangchong/Downloads/hard_sw\
    --debug_example\
    exp/contrastive