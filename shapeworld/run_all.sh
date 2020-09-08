#!/bin/bash

python lsl/train_new.py --cuda \
    --batch_size 32\
    --seed $RANDOM \
    --temperature 0.1\
    --hidden_size 128\
    --lr 0.0004\
    --pt_epochs 150\
    --ft_epochs 20\
    --hypo_model bi_gru\
    --aux_task matching_slot\
    --save_checkpoint\
    --freeze_slots\
    --data_dir /data/cw9951/easy_sw
    exp/matching_slot

python lsl/train_new.py --cuda \
    --batch_size 32\
    --seed $RANDOM \
    --temperature 0.1\
    --hidden_size 128\
    --lr 0.0004\
    --pt_epochs 150\
    --ft_epochs 20\
    --hypo_model bi_gru\
    --aux_task matching_image\
    --freeze_slots\
    --data_dir /data/cw9951/easy_sw
    exp/matching_image

python lsl/train_new.py --cuda \
    --batch_size 32\
    --seed $RANDOM \
    --temperature 0.1\
    --hidden_size 128\
    --lr 0.0004\
    --pt_epochs 150\
    --ft_epochs 20\
    --num_slots 6\
    --hypo_model uni_gru\
    --aux_task caption_slot\
    --freeze_slots\
    --data_dir /data/cw9951/easy_sw
    exp/caption_slot

python lsl/train_new.py --cuda \
    --batch_size 32\
    --seed $RANDOM \
    --temperature 0.1\
    --hidden_size 128\
    --lr 0.0004\
    --pt_epochs 150\
    --ft_epochs 20\
    --hypo_model uni_gru\
    --aux_task caption_image\
    --save_checkpoint\
    --freeze_slots\
    --data_dir /data/cw9951/easy_sw
    exp/caption_image

