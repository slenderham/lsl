#!/bin/bash

python lsl/train.py --cuda \
    --batch_size 100 \
    --seed $RANDOM \
    --comparison bilinear\
    --data_dir /Users/wangchong/Downloads/easy_sw\
    --backbone pretrained\
    exp/meta