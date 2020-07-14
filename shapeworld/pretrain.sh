#!/bin/bash

python lsl/pretrain.py --cuda \
    --batch_size 16\
    --seed $RANDOM \
    --temperature 0.1\
    --lr 0.0001\
    --epochs 0\
    --log_interval 10\
    --backbone vgg16\
    --comparison cosine\
    --data_dir /data/cw9951/easy_sw\
    --skip_eval\
    --pairing im+lang_by_lang\
    exp/contrastive
