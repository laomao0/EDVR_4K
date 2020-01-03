#!/usr/bin/env bash
#
#echoho "using cuda:"$1


##CUDA_VISIBLE_DEVICES=$1
#python test_4k_hdr_with_GT_stitching.py \
#                        --input_path /DATA7_DB7/data/4khdr/data/Dataset/val_540p \
#                        --gt_path /DATA7_DB7/data/4khdr/data/Dataset/val_4k \
#                        --output_path /DATA7_DB7/data/4khdr/data/Results/001_EDVRwTSA_scratch_lr4e-4_600k_4KHDR_archived_191102-132426 \
#                        --model_path /DATA5_DB8/data/4khdr/codes/experiments/001_EDVRwTSA_scratch_lr4e-4_600k_4KHDR_archived_191102-132426/models/latest_G.pth \
#                        --gpu_id 4
#


python /lustre/home/acct-eezy/eezy/4khdr/codes/EDVR/test_4k_hdr_with_GT.py \
        --input_path /lustre/home/acct-eezy/eezy/4khdr/codes/HeJing_4K_SuperResolution-master/val_data/lr \
        --gt_path /lustre/home/acct-eezy/eezy/4khdr/codes/HeJing_4K_SuperResolution-master/val_data/hr \
        --output_path /lustre/home/acct-eezy/eezy/4khdr/codes/HeJing_4K_SuperResolution-master/val_data/predict \
        --model_path /lustre/home/acct-eezy/eezy/4khdr/codes/experiments/train_EDVR_L_Preload_use_hdr_as_input_cos/models/best_G.pth \
        --opt  /lustre/home/acct-eezy/eezy/4khdr/codes/EDVR/options/train/train_EDVR_L_Preload_use_hdr_as_input.yml \
        --gpu_id 0