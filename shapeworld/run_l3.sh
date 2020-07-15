#!/bin/bash

python lsl/train.py --cuda \
    --infer_hyp \
    --hypo_lambda 1.0 \
    --batch_size 100 \
    --seed $RANDOM \
    --backbone vgg16_fixed\
    --mlp_on_feats\
    --comparison dotp\
    --hidden_size 512\
    --lr 0.0001\
    --data_dir /data/cw9951/easy_sw\
    exp/l3
