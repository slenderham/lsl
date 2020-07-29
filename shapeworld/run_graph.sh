#!/bin/bash

HYPO_LAMBDA=1

python lsl/graph.py --cuda \
    --hypo_lambda $HYPO_LAMBDA \
    --batch_size 32\
    --seed $RANDOM \
    --temperature 0.1\
    --hypo_lambda 1.0\
    --concept_lambda 0.0\
    --comparison dotp\
    --hidden_size 64\
    --lr 0.001\
    --skip_eval\
    --save_checkpoint\
    --oracle_world_config\
    --data_dir /data/cw9951/hard_sw\
    exp/graph
