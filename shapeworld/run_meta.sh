#!/bin/bash

python lsl/train.py --cuda \
    --batch_size 100 \
    --seed $RANDOM \
    --lr 0.01\
    --optim sgd\
    exp/meta
