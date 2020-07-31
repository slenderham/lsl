#!/bin/bash

HYPO_LAMBDA=1

python lsl/set_pred.py --cuda \
    --hypo_lambda $HYPO_LAMBDA \
    --batch_size 32\
    --seed $RANDOM \
    --temperature 0.1\
    --hypo_lambda 1.0\
    --concept_lambda 0.0\
    --comparison dotp\
    --hidden_size 128\
    --lr 0.0004\
    --num_slots 3\
    --skip_eval\
    --save_checkpoint\
    --load_checkpoint\
    --oracle_world_config\
    --data_dir /Users/wangchong/Downloads/hard_sw\
    exp/graph
