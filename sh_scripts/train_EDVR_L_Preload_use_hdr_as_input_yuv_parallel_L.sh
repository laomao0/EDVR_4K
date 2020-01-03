#!/usr/bin/env bash


python -m torch.distributed.launch \
      --nproc_per_node=4 \
      --master_port=1589 \
      /lustre/home/acct-eezy/eezy/4khdr/codes/EDVR/train_parallel.py \
      --opt /lustre/home/acct-eezy/eezy/4khdr/codes/EDVR/options/train/train_EDVR_L_Preload_use_hdr_as_input_yuv_parallel_L.yml \
      --launcher pytorch

