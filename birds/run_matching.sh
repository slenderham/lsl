#!/bin/bash

# Standard LSL
python fewshot/run_cl.py \
        --n 1 \
        --log_dir exp/acl/lsl \
        --glove_init \
        --max_lang_per_class 20
