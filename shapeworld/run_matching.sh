#!/bin/bash

HYPO_LAMBDA=1

python lsl/train_new.py --cuda \
    --hypo_lambda $HYPO_LAMBDA \
    --batch_size 4\
    --seed $RANDOM \
    --temperature 0.25\
    --hypo_lambda 1.0\
    --concept_lambda 0.0\
    --pos_weight 1.0\
    --comparison dotp\
    --hidden_size 128\
    --lr 0.0001\
    --num_slots 3\
    --aux_task matching\
    --save_checkpoint\
    --skip_eval\
    --data_dir /Users/wangchong/Downloads/hard_sw\
    exp/graph
