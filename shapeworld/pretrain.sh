#!/bin/bash

HYPO_LAMBDA=1

python lsl/pretrain.py --cuda \
    --batch_size 100\
    --seed $RANDOM \
    --temperature 0.1\
    --lr 0.001\
    --save_feats\
    --log_interval 10\
    --backbone conv4\
    --comparison cosine\
    --data_dir /data/cw9951/hard_sw\
    exp/contrastive
