#!/bin/bash

HYPO_LAMBDA=20

python lsl/pretrain.py --cuda \
    --batch_size 64 \
    --seed $RANDOM \
    --log_interval 10\
    --save_feats\
    exp/contrastive

python lsl/train.py --cuda \
    --predict_concept_hyp \
    --hypo_lambda $HYPO_LAMBDA \
    --batch_size 64 \
    --seed $RANDOM \
    exp/contrastive