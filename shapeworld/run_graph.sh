#!/bin/bash

HYPO_LAMBDA=1

python lsl/graph.py --cuda \
    --hypo_lambda $HYPO_LAMBDA \
    --batch_size 32 \
    --seed $RANDOM \
    --temperature 0.1\
    --comparison dotp\
    --hidden_size 256\
    --lr 0.0001\
    --data_dir /data/cw9951/hard_sw\
    exp/graph
