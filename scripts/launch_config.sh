


#!/bin/bash

#SBATCH -J lysto64_D4_ddim_selfc_da_deep_local
#SBATCH --mem=80G
#SBATCH -c 32
#SBATCH --time=24:05:00
#SBATCH --partition=a40
#SBATCH --export=ALL
#SBATCH --output=%x.%j.log
#SBATCH --gres=gpu:4
#SBATCH --account=deadline
#SBATCH --qos=deadline
#SBATCH --signal=USR1
#SBATCH --signal=B:USR1@60


export PYTHONPATH="${PYTHONPATH}:/ssd005/projects/watml/shared_conda"

handler() 
{
    echo "###############################" && 
    echo "function handler called at $(date)" &&
    echo "###############################" && 
    # do whatever cleanup you want here;
    # checkpoint, sync, etc
    sbatch ${BASH_SOURCE[0]} 
}

trap handler SIGUSR1

/ssd005/projects/watml/shared_conda/cm/bin/mpiexec -n 4 \
    /ssd005/projects/watml/shared_conda/cm/bin/python \
    /ssd005/projects/watml/field/source_codes/Group-Diffusion-Vector-rc2/edm_train.py  \
    model=1stage_edm_deep \
    model.g_output=D4 \
    model.self_cond=True \
    training=analog_bit_ddim \
    diffusion=ddim \
    data=lysto64_aug \
    data.data_dir='/ssd005/projects/watml/data/lysto128' \
    misc.logger_dir='dummy' \
    misc.stat_summary_dir='/ssd005/projects/watml/field/group_inv_rc2/lysto64_D4_ddim_selfc_da_deep_local' \
    misc.exp_name='lysto64_D4_ddim_selfc_da_deep_local' \
    training.global_batch_size=32 \
    training.dropout=0.1 \
    training.use_fp16=False \
    training.lr=0.0002 \
    training.save_interval=2500 training.sampling_interval=50000 \
    slurm.user_id=$USER slurm.slurm_id=$SLURM_JOB_ID &
wait