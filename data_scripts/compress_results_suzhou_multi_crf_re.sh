#!/usr/bin/env bash

#python compress_results.py \
#    --ffmpeg_dir /DATA5_DB8/data/4khdr/codes/codes/ffmpeg \
#    --imgs_folder_input /DATA7_DB7/data/4khdr/data/Results/train_EDVR_L_Preload_30000_nobug/sharp_bicubic\
#    --output_folder /DATA7_DB7/data/4khdr/data/Results/train_EDVR_L_Preload_30000_nobug/\
#    --bitrate 150\

#python compress_results.py \
#    --ffmpeg_dir /DATA5_DB8/data/4khdr/codes/codes/ffmpeg \
#    --imgs_folder_input /DATA7_DB7/data/4khdr/data/Results/train_EDVR_L_Preload_30000_nobug/sharp_bicubic \
#    --output_folder /DATA7_DB7/data/4khdr/data/Results/train_EDVR_L_Preload_30000_nobug/Compressed_r25_v120 \
#    --bitrate 120 \

# python compress_results.py \
#     --ffmpeg_dir /DATA5_DB8/data/4khdr/codes/codes/ffmpeg \
#     --imgs_folder_input /DATA7_DB7/data/4khdr/data/Results/train_EDVR_L_Preload_30000_nobug_221_20000_G_test_540p/sharp_bicubic \
#     --output_folder /DATA7_DB7/data/4khdr/data/Results/train_EDVR_L_Preload_30000_nobug_221_20000_G_test_540p/compressed_120 \
#     --bitrate 120 \

# 1106, yjpan
# python compress_results.py \
#     --ffmpeg_dir /DATA5_DB8/data/4khdr/codes/codes/ffmpeg \
#     --imgs_folder_input /DATA7_DB7/data/4khdr/data/Results/train_EDVR_L_Preload_30000_nobug_221_20000_G_test_540p/sharp_bicubic \
#     --output_folder     /DATA7_DB7/data/4khdr/data/Results/train_EDVR_L_Preload_30000_nobug_221_20000_G_test_540p/compressed_120 \
#     --bitrate 120 \

# /DATA5_DB8/data/4khdr/codes/codes/ffmpeg/ffmpeg -r 25 -i  /DATA7_DB7/data/4khdr/data/Results/train_EDVR_L_Preload_30000_nobug_221_20000_G_test_540p/sharp_bicubic/54003182/%05d.png   -b:v 100M -vcodec libx265 /DATA7_DB7/data/4khdr/data/Results/train_EDVR_L_Preload_30000_nobug_221_20000_G_test_540p/compressed_120/54003182.mp4

## 56459353
#/DATA5_DB8/data/4khdr/codes/codes/ffmpeg/ffmpeg -r 25 -i  /DATA7_DB7/data/4khdr/data/Results/train_EDVR_L_Preload_30000_nobug_221_20000_G_test_540p/sharp_bicubic/$1/%05d.png   -b:v $2M -vcodec libx265 /DATA7_DB7/data/4khdr/data/Results/train_EDVR_L_Preload_30000_nobug_221_20000_G_test_540p/compressed_120/$1.mp4


#python compress_results.py \
#    --ffmpeg_dir ffmpeg \
#    --imgs_folder_input /DATA7_DB7/data/4khdr/data/Results/train_EDVR_L_Preload_30000_nobug_221_20000_G_test_540p/sharp_bicubic \
#    --output_folder /DATA7_DB7/data/4khdr/data/Results/train_EDVR_L_Preload_30000_nobug_221_20000_G_test_540p/compressed_yuv422p \
#    --bitrate 10 \



#python compress_results_suzhou.py \
#    --ffmpeg_dir /mnt/lustre/shanghai/cmic/home/xyz18/codes/codes/codes/ffmpeg/ffmpeg \
#    --imgs_folder_input /mnt/lustre/shanghai/cmic/home/xyz18/Results/train_EDVR_L_Preload_30000_nobug_ssim_7_LR_1e-5_suzhou/sharp_bicubic \
#    --output_folder /mnt/lustre/shanghai/cmic/home/xyz18/Results/train_EDVR_L_Preload_30000_nobug_ssim_7_LR_1e-5_suzhou/compressed_yuv422p \
#    --bitrate 9 \

#python compress_results_suzhou.py \
##    --ffmpeg_dir /mnt/lustre/shanghai/cmic/home/xyz18/codes/codes/codes/ffmpeg/ffmpeg \
##    --imgs_folder_input /mnt/lustre/shanghai/cmic/home/xyz18/Results/train_EDVR_L_Preload_30000_nobug_ssim_9_LR_1e_5_suzhou_remove_baddata/sharp_bicubic \
##    --output_folder /mnt/lustre/shanghai/cmic/home/xyz18/Results/train_EDVR_L_Preload_30000_nobug_ssim_9_LR_1e_5_suzhou_remove_baddata/compressed_yuv422p \
##    --bitrate 9 \

#python compress_results_suzhou.py \
#    --ffmpeg_dir /mnt/lustre/shanghai/cmic/home/xyz18/codes/codes/codes/ffmpeg/ffmpeg \
#    --imgs_folder_input /mnt/lustre/shanghai/cmic/home/xyz18/Results/train_EDVR_L_Preload_30000_nobug_nf192_6_ssim_parallel/sharp_bicubic \
#    --output_folder /mnt/lustre/shanghai/cmic/home/xyz18/Results/train_EDVR_L_Preload_30000_nobug_nf192_6_ssim_parallel/compressed_yuv422p \
#    --crf 8 \

#python compress_results_suzhou_multi_crf.py \
#    --ffmpeg_dir /mnt/lustre/shanghai/cmic/home/xyz18/codes/codes/codes/ffmpeg/ffmpeg \
#    --imgs_folder_input /mnt/lustre/shanghai/cmic/home/xyz18/Results/train_EDVR_L_Preload_30000_nobug_ssim_10_LR_1e_5_suzhou_remove_baddata/sharp_bicubic \
#    --output_folder /mnt/lustre/shanghai/cmic/home/xyz18/Results/train_EDVR_L_Preload_30000_nobug_ssim_10_LR_1e_5_suzhou_remove_baddata/compressed_yuv422p \
#    --crf 8

#python compress_results_suzhou_multi_crf.py \
#    --ffmpeg_dir /mnt/lustre/shanghai/cmic/home/xyz18/codes/codes/codes/ffmpeg/ffmpeg \
#    --imgs_folder_input /mnt/lustre/shanghai/cmic/home/xyz18/Results/train_EDVR_L_Preload_30000_nobug_ssim_predeblur_parral_suzhou_3/sharp_bicubic \
#    --output_folder /mnt/lustre/shanghai/cmic/home/xyz18/Results/train_EDVR_L_Preload_30000_nobug_ssim_predeblur_parral_suzhou_3/compressed_yuv422p \
#    --crf 8

#python compress_results_suzhou_multi_crf.py \
#    --ffmpeg_dir /mnt/lustre/shanghai/cmic/home/xyz18/codes/codes/codes/ffmpeg/ffmpeg \
#    --imgs_folder_input /mnt/lustre/shanghai/cmic/home/xyz18/Results/train_EDVR_L_Preload_30000_nobug_ssim_predeblur_parral_suzhou_5/sharp_bicubic \
#    --output_folder /mnt/lustre/shanghai/cmic/home/xyz18/Results/train_EDVR_L_Preload_30000_nobug_ssim_predeblur_parral_suzhou_5/compressed_yuv422p \
#    --crf 8

# python compress_results_suzhou_multi_crf.py \
#     --ffmpeg_dir /mnt/lustre/shanghai/cmic/home/xyz18/codes/codes/codes/ffmpeg/ffmpeg \
#     --imgs_folder_input /mnt/lustre/shanghai/cmic/home/xyz18/Results/train_EDVR_L_Preload_30000_nobug_ssim_11_LR_1e_5_suzhou_remove_baddata/sharp_bicubic \
#     --output_folder /mnt/lustre/shanghai/cmic/home/xyz18/Results/train_EDVR_L_Preload_30000_nobug_ssim_11_LR_1e_5_suzhou_remove_baddata/compressed_yuv422p \
#     --crf 8

# python compress_results_suzhou_multi_crf_re.py \
#     --ffmpeg_dir /lustre/home/acct-eezy/eezy/4khdr/codes/EDVR/ffmpeg/ffmpeg \
#     --imgs_folder_input /lustre/home/acct-eezy/eezy/4khdr/codes/HDRNet/logs/test_4k_infer_self_ensamble \
#     --output_folder /lustre/home/acct-eezy/eezy/4khdr/Results/test_4k_infer_self_ensamble \
#     --crf 8

# python compress_results_suzhou_multi_crf_re.py \
#     --ffmpeg_dir /lustre/home/acct-eezy/eezy/4khdr/codes/EDVR/ffmpeg/ffmpeg \
#     --imgs_folder_input /lustre/home/acct-eezy/eezy/4khdr/codes/HDRNet/logs/1226_2035_infer_aug \
#     --output_folder  /lustre/home/acct-eezy/eezy/4khdr/Results/1226_2035_infer_aug \
#     --crf 8

python compress_results_suzhou_multi_crf_re.py \
    --ffmpeg_dir /lustre/home/acct-eezy/eezy/4khdr/codes/EDVR/ffmpeg/ffmpeg \
    --imgs_folder_input /lustre/home/acct-eezy/eezy/4khdr/codes/HDRNet/logs/test_4k_aug_hdr2_from_1226_2035_infer_aug \
    --output_folder  /lustre/home/acct-eezy/eezy/4khdr/Results/test_4k_aug_hdr2_from_1226_2035_infer_aug \
    --crf 8

