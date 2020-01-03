#!/bin/bash
#SBATCH --job-name=run_hdr_lmdb_parall
#SBATCH --partition=dgx2
#SBATCH -n 1
#SBATCH -o /lustre/home/acct-eezy/eezy/4khdr/codes/EDVR/slum_logs/run_hdr_lmdb_parall.out
#SBATCH -e /lustre/home/acct-eezy/eezy/4khdr/codes/EDVR/slum_logs/run_hdr_lmdb_parall.err
#SBATCH -N 1
#SBATCH --gres=gpu:2

# ulimit -l unlimited
# ulimit -s unlimited

#export LD_LIBRARY_PATH=/mnt/lustre/cm/shared/global/src/machinelearning/anaconda2/lib:$LD_LIBRARY_PATH

#source /usr/share/Modules/init/bash

module purge
module load miniconda2/4.6.14-gcc-4.8.5 cuda/9.0.176-gcc-4.8.5 cudnn/7.0.4

source activate /lustre/home/acct-eezy/eezy/4khdr/torch11

python /lustre/home/acct-eezy/eezy/4khdr/codes/EDVR/train.py --opt ../options/train/train_EDVR_woTSA_M_hdr_test.yml