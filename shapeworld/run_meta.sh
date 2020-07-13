#!/bin/bash

python lsl/train.py --cuda \
    --batch_size 100 \
    --seed $RANDOM \
    --comparison dotp\
    --lr 0.0001\
    --data_dir /data/cw9951/easy_sw\
    --backbone pretrained\
    --normalize_feats\
    --mlp_on_feats\
    exp/meta
