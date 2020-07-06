#!/bin/bash

HYPO_LAMBDA=20

python lsl/train.py --cuda \
    --predict_concept_hyp \
    --hypo_lambda $HYPO_LAMBDA \
    --batch_size 100 \
    --seed $RANDOM \
    --data_dir /Users/wangchong/Downloads/easy_sw\
    exp/lsl
