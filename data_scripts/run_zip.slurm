#!/bin/bash
#SBATCH --job-name=video_create
#SBATCH --partition=small
#SBATCH -n 1
#SBATCH -o /lustre/home/acct-eezy/eezy/4khdr/Results/201912027.out
#SBATCH -e /lustre/home/acct-eezy/eezy/4khdr/Results/201912027.err
#SBATCH -N 1
#SBATCH --exclusive

module purge
module load miniconda2/4.6.14-gcc-4.8.5 cuda/9.0.176-gcc-4.8.5 cudnn/7.0.4

source activate /lustre/home/acct-eezy/eezy/4khdr/torch11

# sh create_dataset_4k.sh
# python opencv_test_59714265.py
# sh compress_results_suzhou_multi_crf_re.sh
# sh /lustre/home/acct-eezy/eezy/4khdr/codes/EDVR/data_scripts/compress_results_suzhou_multi_crf.sh

cd /lustre/home/acct-eezy/eezy/4khdr/Results/test_4k_infer_rotate_self_ensamble_EDVR_self_en
zip test_4k_infer_rotate_self_ensamble_EDVR_self_en.zip *.mp4