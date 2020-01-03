#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES=$device python /lustre/home/acct-eezy/eezy/4khdr/codes/EDVR/train.py \
        --opt /lustre/home/acct-eezy/eezy/4khdr/codes/EDVR/options/train/train_EDVR_L_Preload_use_hdr_as_input.yml