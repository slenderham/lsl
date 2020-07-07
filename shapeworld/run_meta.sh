#!/bin/bash

python lsl/train.py --cuda \
    --batch_size 100 \
    --seed $RANDOM \
    --backbone conv4\
    --tune_backbone\
    exp/meta
