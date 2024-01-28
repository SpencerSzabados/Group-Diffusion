#!/bin/bash
#SBATCH -J sing
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH -c 8
#SBATCH --time=200:00:00
#SBATCH --partition=a40
#SBATCH --export=ALL
#SBATCH --output=%x.%j.log
#SBATCH --gres=gpu:1

module load cuda-11.7 && module load singularity-ce

singularity run -e --nv \
    --bind /h/field/DiffGAN/cifar10:/cifar10 \
    --bind /h/field/DiffGAN/circ_diff_vector:/circ_diff_vector \
    --bind /checkpoint:/checkpoint \
    /h/field/ssd004/cm_export.sif \
    python /circ_diff_vector/edm_train.py --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 256  --image_size 32 --lr 0.0001 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --schedule_sampler lognormal --use_fp16 True --weight_decay 0.0 --weight_schedule uniform --data_dir /cifar10 --save_interval 10000 --slurm_id $SLURM_JOB_ID --user_id $USER

