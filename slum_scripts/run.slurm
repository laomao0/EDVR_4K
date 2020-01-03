#!/bin/bash
#SBATCH --job-name=create_4k_data_1230
#SBATCH --partition=dgx2
#SBATCH -n 1
#SBATCH -o create_4k_data_1230.out
#SBATCH -e create_4k_data_1230.err
#SBATCH -N 1
#SBATCH --gres=gpu:1

# ulimit -l unlimited
# ulimit -s unlimited

#export LD_LIBRARY_PATH=/mnt/lustre/cm/shared/global/src/machinelearning/anaconda2/lib:$LD_LIBRARY_PATH

#source /usr/share/Modules/init/bash

module purge
module load miniconda2/4.6.14-gcc-4.8.5 cuda/9.0.176-gcc-4.8.5 cudnn/7.0.4

source activate /lustre/home/acct-eezy/eezy/4khdr/torch11

# sh test_4k_hdr_without_GT.sh
sh test_4k_hdr_without_GT_test_self_ensamble.sh