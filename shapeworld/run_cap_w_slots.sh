#!/bin/bash

HYPO_LAMBDA=1

python lsl/train_new.py --cuda \
    --hypo_lambda $HYPO_LAMBDA \
    --batch_size 32\
    --seed $RANDOM \
    --temperature 0.1\
    --hypo_lambda 1.0\
    --concept_lambda 0.0\
    --pos_weight 1.0\
    --comparison dotp\
    --aux_task caption_slot\
    --hidden_size 128\
    --lr 0.0001\
    --num_slots 3\
    --save_checkpoint\
    --data_dir /data/cw9951/hard_sw\
    exp/caption_w_slots
