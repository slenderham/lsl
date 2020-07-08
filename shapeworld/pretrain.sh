#!/bin/bash

HYPO_LAMBDA=1

python lsl/pretrain.py --cuda \
    --batch_size 32\
    --seed $RANDOM \
    --temperature 0.1\
    --lr 0.001\
    --log_interval 10\
    --backbone conv4\
    --comparison cosine\
    --data_dir /Users/wangchong/Downloads/hard_sw\
    --pairing im+lang_im+im\
    exp/contrastive
