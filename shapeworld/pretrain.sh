#!/bin/bash

python lsl/pretrain.py --cuda \
    --batch_size 100\
    --seed $RANDOM \
    --temperature 0.1\
    --lr 0.001\
    --log_interval 10\
    --backbone vgg16\
    --comparison cosine\
    --data_dir /Users/wangchong/Downloads/easy_sw\
    --skip_eval\
    --pairing im+lang_by_lang\
    exp/contrastive
