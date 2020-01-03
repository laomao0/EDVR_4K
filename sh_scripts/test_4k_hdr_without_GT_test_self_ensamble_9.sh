#!/usr/bin/env bash
#
#device=$1 #the first arg is the device num.
#echo Using CUDA device $device

#python test_4k_hdr_with_GT_stitching_process_screen_change.py \
#        --input_path /DATA7_DB7/data/4khdr/data/Dataset/val_540p \
#        --gt_path /DATA7_DB7/data/4khdr/data/Dataset/val_4k \
#        --screen_notation  /DATA7_DB7/data/4khdr/data/Dataset/4khdr_frame_notation.json \
#        --output_path /DATA7_DB7/data/4khdr/data/Results/001_EDVRwTSA_scratch_lr4e-4_600k_4KHDR_archived_191102-132426-screen-detect \
#        --model_path /DATA5_DB8/data/4khdr/codes/experiments/001_EDVRwTSA_scratch_lr4e-4_600k_4KHDR_archived_191102-132426/models/latest_G.pth \
#        --opt  /DATA5_DB8/data/4khdr/codes/codes/options/train/my_train_EDVR_L.yml \
#        --gpu_id 4
#


#CUDA_VISIBLE_DEVICES=$1
#CUDA_VISIBLE_DEVICES=$device



# python test_4k_hdr_without_GT_padding.py \
#         --input_path /lustre/home/acct-eezy/eezy/4khdr/data/Dataset/test_540p \
#         --screen_notation  /mnt/lustre/shanghai/cmic/home/xyz18/Dataset/test_4khdr_frame_notation.json \
#         --use_screen_notation 0 \
#         --output_path /lustre/home/acct-eezy/eezy/4khdr/data/Dataset/test_4k_from_LDR_540p \
#         --model_path /lustre/home/acct-eezy/eezy/4khdr/codes/experiments/train_EDVR_L_Preload_30000_nobug_ssim_10_LR_1e_5_suzhou_remove_baddata/models/latest_G.pth \
#         --opt  /lustre/home/acct-eezy/eezy/4khdr/codes/EDVR/options/train/train_EDVR_L_Preload_30000_nobug_ssim_12_LR_1e_5_suzhou_remove_baddata.yml \
#         --gpu_id 0 

python /lustre/home/acct-eezy/eezy/4khdr/codes/EDVR/test_4k_hdr_without_GT_padding_self_ensamble_1230_multi.py \
        --input_path /lustre/home/acct-eezy/eezy/4khdr/data/Dataset/test_540p \
        --screen_notation  /mnt/lustre/shanghai/cmic/home/xyz18/Dataset/test_4khdr_frame_notation.json \
        --use_screen_notation 0 \
        --output_path /lustre/home/acct-eezy/eezy/4khdr/data/Dataset/test_4k_from_LDR_540p_self_ensamble \
        --model_path /lustre/home/acct-eezy/eezy/4khdr/codes/experiments/train_EDVR_L_Preload_30000_nobug_ssim_10_LR_1e_5_suzhou_remove_baddata/models/latest_G.pth \
        --opt  /lustre/home/acct-eezy/eezy/4khdr/codes/EDVR/options/train/train_EDVR_L_Preload_30000_nobug_ssim_12_LR_1e_5_suzhou_remove_baddata.yml \
        --gpu_id 0 \
        --file_idx 9


