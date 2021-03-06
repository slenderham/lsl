#!/bin/bash

HYPO_LAMBDA=5

python lsl/train.py --cuda \
    --predict_concept_hyp \
    --hypo_lambda $HYPO_LAMBDA \
    --batch_size 100 \
    --seed $RANDOM \
    --backbone slot_attn\
    --mlp_on_feats\
    --comparison dotp\
    --hidden_size 512\
    --lr 0.001\
    --data_dir /data/cw9951/hard_sw\
    exp/slot_attn
