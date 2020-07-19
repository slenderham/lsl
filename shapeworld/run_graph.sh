#!/bin/bash

HYPO_LAMBDA=1

python lsl/graph.py --cuda \
    --hypo_lambda $HYPO_LAMBDA \
    --batch_size 100 \
    --seed $RANDOM \
    --comparison dotp\
    --hidden_size 32\
    --lr 0.001\
    --data_dir /Users/wangchong/Downloads/hard_sw\
    exp/graph
