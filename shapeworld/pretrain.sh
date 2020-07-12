#!/bin/bash

python lsl/pretrain.py --cuda \
    --batch_size 100\
    --seed $RANDOM \
    --temperature 0.1\
    --lr 0.001\
    --log_interval 10\
    --backbone conv4\
    --comparison cosine\
    --data_dir /data/cw9951/easy_sw\
    --skip_eval\
    --pairing im+lang_by_im\
    exp/contrastive
