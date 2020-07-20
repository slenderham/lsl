#!/bin/bash

HYPO_LAMBDA=10

python lsl/graph.py --cuda \
    --hypo_lambda $HYPO_LAMBDA \
    --batch_size 64\
    --seed $RANDOM \
    --temperature 0.1\
    --comparison dotp\
    --hidden_size 128\
    --lr 0.0001\
    --data_dir /data/cw9951/easy_sw\
    exp/graph
