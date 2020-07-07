#!/bin/bash

python lsl/train.py --cuda \
    --batch_size 100 \
    --seed $RANDOM \
    --comparison bilinear\
    exp/meta