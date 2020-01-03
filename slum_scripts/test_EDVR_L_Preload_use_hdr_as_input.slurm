#!/bin/bash
#SBATCH --job-name=test_EDVR_L_Preload_use_hdr_as_input
#SBATCH --partition=dgx2
#SBATCH -n 1
#SBATCH -o /lustre/home/acct-eezy/eezy/4khdr/codes/EDVR/slum_logs/test_EDVR_L_Preload_use_hdr_as_input.out
#SBATCH -e /lustre/home/acct-eezy/eezy/4khdr/codes/EDVR/slum_logs/test_EDVR_L_Preload_use_hdr_as_input.err
#SBATCH -N 1
#SBATCH --gres=gpu:1

# ulimit -l unlimited
# ulimit -s unlimited

#export LD_LIBRARY_PATH=/mnt/lustre/cm/shared/global/src/machinelearning/anaconda2/lib:$LD_LIBRARY_PATH

#source /usr/share/Modules/init/bash

module purge
module load miniconda2/4.6.14-gcc-4.8.5 cuda/9.0.176-gcc-4.8.5 cudnn/7.0.4

source activate /lustre/home/acct-eezy/eezy/4khdr/torch11

sh /lustre/home/acct-eezy/eezy/4khdr/codes/EDVR/test_4k_hdr_with_GT.sh