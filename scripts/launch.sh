# File contains command line scripts for launching the training of different diffusion
# models for different image dataset and noise schedules.

######################################################################################
# Testing code 
######################################################################################

######################################################################################
# Trainning non-group equivarient EDM model on unconditional C4 toy image dataset
######################################################################################
# Train without group-equivariant layers
OPENAI_LOGDIR=/home/checkpoints/Group-Diffusion/c4toy_example_non_eqv python edm_train.py --g_equiv False --g_input Z2_K --g_output Z2_K --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 512 --image_size 28 --lr 0.0001 --num_channels 64 --num_head_channels 32 --num_res_blocks 1 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --save_interval 1000 --data_dir /home/datasets/c4_toy

# Training using DDIM model
OPENAI_LOGDIR=/home/checkpoints/Group-Diffusion/c4toy_example_eqv_ddim python edm_train.py --g_equiv True --g_input Z2_K --g_output C4_K --diff_type ddim --data_augment 0 --eqv_reg None --pred_type x --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 256 --image_size 28 --lr 0.0001 --num_channels 64 --num_head_channels 32 --num_res_blocks 1 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --save_interval 1000 --data_dir /home/datasets/c4_toy

# Training using DDIM and incremental FID sampling 
OPENAI_LOGDIR=/home/checkpoints/Group-Diffusion/c4toy_example_eqv_ddim python edm_train_fid.py --g_equiv True --g_input Z2_K --g_output C4_K --diff_type ddim --sampler ddim --data_augment 0 --eqv_reg None --pred_type x --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 256 --image_size 28 --lr 0.0001 --num_channels 64 --num_head_channels 32 --num_res_blocks 1 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --save_interval 1000 --sampling_interval 1000 --data_dir /home/datasets/c4_toy --sampling_dir /home/datasets/fid_samples --resume_checkpoint /home/checkpoints/Group-Diffusion/c4toy_example_eqv_ddim


######################################################################################
# Training group equivariant EDM model on class-conditional Rot-MNIST
######################################################################################
OPENAI_LOGDIR=/home/checkpoints/Group-Diffusion/rot_mnist_6000 mpiexec --allow-run-as-root -n 2 python edm_train.py --g_equiv True --g_input Z2_K --g_output C4_K --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 512 --image_size 28 --lr 0.0001 --num_channels 64 --num_head_channels 32 --num_res_blocks 1 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --save_interval 1000 --data_dir /home/datasets/rot_mnist_6000

# Command to resume training from checkpoint 
OPENAI_LOGDIR=/home/checkpoints/Group-Diffusion/rot_mnist_6000 mpiexec --allow-run-as-root -n 2 python edm_train.py --g_equiv True --g_input Z2_K --g_output C4_K --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 512 --image_size 28 --lr 0.0001 --num_channels 64 --num_head_channels 32 --num_res_blocks 1 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --save_interval 1000 --data_dir /home/datasets/rot_mnist_6000 --resume_checkpoint /home/checkpoints/Group-Diffusion/rot_mnist_6000/

# Training using DDIM and incremental FID sampling 
# Hyperparameter searching
# TML3 - c4_mnist_600  - reg model - FID: 17.96
CUDA_VISIBLE_DEVICES=1 OPENAI_LOGDIR=/home/checkpoints/Group-Diffusion/c4_mnist_600_reg_ddim_ch124 python edm_train_fid.py --g_equiv False --g_input Z2_K --g_output C4_K --diff_type ddim --sampler ddim --data_augment 0 --eqv_reg C4 --pred_type x --attention_resolutions 32,16,8 --class_cond True --channel_mult 1,2,4 --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 128 --image_size 28 --lr 0.0001 --num_channels 64 --num_head_channels 32 --num_res_blocks 1 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --save_interval 1000 --sampling_interval 1000 --data_dir /home/datasets/c4_mnist_600 --ref_dir /home/datasets/c4_mnist_50000 --sampling_dir /home/datasets/fid_samples_3 --resume_checkpoint /home/checkpoints/Group-Diffusion/c4_mnist_600_reg_ddim_ch124/
# TML3 - c4_mnist_600  - reg model - FID: 14.33, 10.51, 
CUDA_VISIBLE_DEVICES=0 OPENAI_LOGDIR=/home/checkpoints/Group-Diffusion/c4_mnist_600_reg_ddim_ch124_r2 python edm_train_fid.py --g_equiv False --g_input Z2_K --g_output C4_K --diff_type ddim --sampler ddim --data_augment 0 --eqv_reg C4 --pred_type x --attention_resolutions 32,16,8 --class_cond True --channel_mult 1,2,4 --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 500 --image_size 28 --lr 0.0001 --num_channels 64 --num_head_channels 32 --num_res_blocks 2 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --save_interval 1000 --sampling_interval 1000 --data_dir /home/datasets/c4_mnist_600 --ref_dir /home/datasets/c4_mnist_50000 --sampling_dir /home/datasets/fid_samples_2 --resume_checkpoint /home/checkpoints/Group-Diffusion/c4_mnist_600_reg_ddim_ch124_r2/
# TML1 - c4_mnist_3000 - reg model - FID: 21.60, 15.68, 9.66, 7.43, 6.47, 5.85, 5.15, 4.15
CUDA_VISIBLE_DEVICES=0 OPENAI_LOGDIR=/home/checkpoints/Group-Diffusion/c4_mnist_3000_reg_ddim_ch124_r2 python edm_train_fid.py --g_equiv False --g_input Z2_K --g_output C4_K --diff_type ddim --sampler ddim --data_augment 0 --eqv_reg C4 --pred_type x --attention_resolutions 32,16,8 --class_cond True --channel_mult 1,2,4 --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 500 --image_size 28 --lr 0.0001 --num_channels 64 --num_head_channels 32 --num_res_blocks 2 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --save_interval 1000 --sampling_interval 1000 --data_dir /home/datasets/c4_mnist_3000 --ref_dir /home/datasets/c4_mnist_50000 --sampling_dir /home/datasets/fid_samples_7 --resume_checkpoint /home/checkpoints/Group-Diffusion/c4_mnist_3000_reg_ddim_ch124_r2/
# TML1 - c4_mnist_6000 - reg model - FID: 15.23, 8.95, 8.70, 5.57, 5.206
CUDA_VISIBLE_DEVICES=1 OPENAI_LOGDIR=/home/checkpoints/Group-Diffusion/c4_mnist_6000_reg_ddim_ch124_r2 python edm_train_fid.py --g_equiv False --g_input Z2_K --g_output C4_K --diff_type ddim --sampler ddim --data_augment 0 --eqv_reg C4 --pred_type x --attention_resolutions 32,16,8 --class_cond True --channel_mult 1,2,4 --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 500 --image_size 28 --lr 0.0001 --num_channels 64 --num_head_channels 32 --num_res_blocks 2 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --save_interval 1000 --sampling_interval 1000 --data_dir /home/datasets/c4_mnist_6000 --ref_dir /home/datasets/c4_mnist_50000 --sampling_dir /home/datasets/fid_samples_3 --resume_checkpoint /home/checkpoints/Group-Diffusion/c4_mnist_6000_reg_ddim_ch124_r2/

# TML3 - c4_mnist_600  - weight-tied model - FID: 18.73
CUDA_VISIBLE_DEVICES=1 OPENAI_LOGDIR=/home/checkpoints/Group-Diffusion/c4_mnist_600_eqv_ddim_ch124 python edm_train_fid.py --g_equiv True --g_input Z2_K --g_output C4_K --diff_type ddim --sampler ddim --data_augment 0 --eqv_reg None --pred_type x --attention_resolutions 32,16,8 --class_cond True --channel_mult 1,2,4 --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 128 --image_size 28 --lr 0.0001 --num_channels 64 --num_head_channels 32 --num_res_blocks 1 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --save_interval 1000 --sampling_interval 1000 --data_dir /home/datasets/c4_mnist_600 --ref_dir /home/datasets/c4_mnist_50000 --sampling_dir /home/datasets/fid_samples --resume_checkpoint /home/checkpoints/Group-Diffusion/c4_mnist_600_eqv_ddim_ch124/
# TML3 - c4_mnist_3000 - weight-tied model - FID: 15.37
CUDA_VISIBLE_DEVICES=0 OPENAI_LOGDIR=/home/checkpoints/Group-Diffusion/c4_mnist_3000_eqv_ddim_ch124 python edm_train_fid.py --g_equiv True --g_input Z2_K --g_output C4_K --diff_type ddim --sampler ddim --data_augment 0 --eqv_reg None --pred_type x --attention_resolutions 32,16,8 --class_cond True --channel_mult 1,2,4 --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 128 --image_size 28 --lr 0.0001 --num_channels 64 --num_head_channels 32 --num_res_blocks 1 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --save_interval 1000 --sampling_interval 1000 --data_dir /home/datasets/c4_mnist_3000 --ref_dir /home/datasets/c4_mnist_50000 --sampling_dir /home/datasets/fid_samples_4 --resume_checkpoint /home/checkpoints/Group-Diffusion/c4_mnist_3000_eqv_ddim_ch124/
# TML2 - c4_mnist_6000 - weight-tied model - FID: 10.63
CUDA_VISIBLE_DEVICES=0 OPENAI_LOGDIR=/home/checkpoints/Group-Diffusion/c4_mnist_6000_eqv_ddim_ch124 python edm_train_fid.py --g_equiv True --g_input Z2_K --g_output C4_K --diff_type ddim --sampler ddim --data_augment 0 --eqv_reg None --pred_type x --attention_resolutions 32,16,8 --class_cond True --channel_mult 1,2,4 --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 128 --image_size 28 --lr 0.0001 --num_channels 64 --num_head_channels 32 --num_res_blocks 1 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --save_interval 1000 --sampling_interval 1000 --data_dir /home/datasets/c4_mnist_6000 --ref_dir /home/datasets/c4_mnist_50000 --sampling_dir /home/datasets/fid_samples --resume_checkpoint /home/checkpoints/Group-Diffusion/c4_mnist_6000_eqv_ddim_ch124/

# TML3 - c4_mnist_600  - weight-tied model - FID: 8.47
CUDA_VISIBLE_DEVICES=0 OPENAI_LOGDIR=/home/checkpoints/Group-Diffusion/c4_mnist_600_eqv_ddim_ch1222 python edm_train_fid.py --g_equiv True --g_input Z2_K --g_output C4_K --diff_type ddim --sampler ddim --data_augment 0 --eqv_reg None --pred_type x --attention_resolutions 32,16,8 --class_cond True --channel_mult 1,2,4 --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 128 --image_size 28 --lr 0.0001 --num_channels 64 --num_head_channels 32 --num_res_blocks 2 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --save_interval 1000 --sampling_interval 1000 --data_dir /home/datasets/c4_mnist_600 --ref_dir /home/datasets/c4_mnist_50000 --sampling_dir /home/datasets/fid_samples --resume_checkpoint /home/checkpoints/Group-Diffusion/c4_mnist_600_eqv_ddim_ch1222/
# TML3 - c4_mnist_3000 - weight-tied model - FID: 10.92, 9.25, 9.23
CUDA_VISIBLE_DEVICES=1 OPENAI_LOGDIR=/home/checkpoints/Group-Diffusion/c4_mnist_3000_eqv_ddim_ch124_r2 python edm_train_fid.py --g_equiv True --g_input Z2_K --g_output C4_K --diff_type ddim --sampler ddim --data_augment 0 --eqv_reg None --pred_type x --attention_resolutions 32,16,8 --class_cond True --channel_mult 1,2,4 --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 128 --image_size 28 --lr 0.0001 --num_channels 64 --num_head_channels 32 --num_res_blocks 2 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --save_interval 1000 --sampling_interval 1000 --data_dir /home/datasets/c4_mnist_3000 --ref_dir /home/datasets/c4_mnist_50000 --sampling_dir /home/datasets/fid_samples_2 --resume_checkpoint /home/checkpoints/Group-Diffusion/c4_mnist_3000_eqv_ddim_ch124_r2/
# TML2 - c4_mnist_6000 - weight-tied model - FID: 7.29, 6.82, 5.99
CUDA_VISIBLE_DEVICES=0 OPENAI_LOGDIR=/home/checkpoints/Group-Diffusion/c4_mnist_6000_eqv_ddim_ch124_r2 python edm_train_fid.py --g_equiv True --g_input Z2_K --g_output C4_K --diff_type ddim --sampler ddim --data_augment 0 --eqv_reg None --pred_type x --attention_resolutions 32,16,8 --class_cond True --channel_mult 1,2,4 --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 128 --image_size 28 --lr 0.0001 --num_channels 64 --num_head_channels 32 --num_res_blocks 2 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --save_interval 1000 --sampling_interval 1000 --data_dir /home/datasets/c4_mnist_6000 --ref_dir /home/datasets/c4_mnist_50000 --sampling_dir /home/datasets/fid_samples --resume_checkpoint /home/checkpoints/Group-Diffusion/c4_mnist_6000_eqv_ddim_ch124_r2/

# Models for validating equivariance using NLL using pfode
# TML - c4_mnist - weight-tied model      - FID: 33.88
CUDA_VISIBLE_DEVICES=0 OPENAI_LOGDIR=/home/checkpoints/Group-Diffusion/c4_mnist_eqv_pfode_ch124_r2 python edm_train_fid.py --g_equiv True --g_input Z2_K --g_output C4_K --diff_type pfode --sampler heun --data_augment 0 --eqv_reg None --pred_type x --attention_resolutions 32,16,8 --class_cond True --channel_mult 1,2,4 --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 128 --image_size 28 --lr 0.0001 --num_channels 64 --num_head_channels 32 --num_res_blocks 2 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --save_interval 1000 --sampling_interval 1000 --data_dir /home/datasets/c4_mnist --ref_dir /home/datasets/c4_mnist_50000 --sampling_dir /home/datasets/fid_samples --resume_checkpoint /home/checkpoints/Group-Diffusion/c4_mnist_eqv_pfode_ch124_r2/
# TML - c4_mnist - non-equviariance model - FID: 8.54
CUDA_VISIBLE_DEVICES=1 OPENAI_LOGDIR=/home/checkpoints/Group-Diffusion/c4_mnist_pfode_ch124_r2 python edm_train_fid.py --g_equiv False --g_input Z2_K --g_output C4_K --diff_type pfode --sampler heun --data_augment 0 --eqv_reg None --pred_type x --attention_resolutions 32,16,8 --class_cond True --channel_mult 1,2,4 --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 128 --image_size 28 --lr 0.0001 --num_channels 64 --num_head_channels 32 --num_res_blocks 2 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --save_interval 1000 --sampling_interval 1000 --data_dir /home/datasets/c4_mnist --ref_dir /home/datasets/c4_mnist_50000 --sampling_dir /home/datasets/fid_samples_3 --resume_checkpoint /home/checkpoints/Group-Diffusion/c4_mnist_pfode_ch124_r2/


# ERROR: Models were trained on incorrect dataset - trained on mnist that was rotated (0-4)deg rather than (0-4)*90deg
# TML3 - c4_mnist_600  - weight-tied model - FID: 8.86
CUDA_VISIBLE_DEVICES=0 OPENAI_LOGDIR=/home/checkpoints/Group-Diffusion/c4_mnist_600_eqv_ddim_ch124 python edm_train_fid.py --g_equiv True --g_input Z2_K --g_output C4_K --diff_type ddim --sampler ddim --data_augment 0 --eqv_reg None --pred_type x --attention_resolutions 32,16,8 --class_cond True --channel_mult 1,2,4 --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 128 --image_size 28 --lr 0.0001 --num_channels 64 --num_head_channels 32 --num_res_blocks 1 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --save_interval 1000 --sampling_interval 1000 --data_dir /home/datasets/c4_mnist_600 --ref_dir /home/datasets/c4_mnist_50000 --sampling_dir /home/datasets/fid_samples --resume_checkpoint /home/checkpoints/Group-Diffusion/c4_mnist_600_eqv_ddim_ch124/
# TML3 - c4_mnist_3000 - weight-tied model - FID: 6.64
CUDA_VISIBLE_DEVICES=0 OPENAI_LOGDIR=/home/checkpoints/Group-Diffusion/c4_mnist_3000_eqv_ddim_ch124 python edm_train_fid.py --g_equiv True --g_input Z2_K --g_output C4_K --diff_type ddim --sampler ddim --data_augment 0 --eqv_reg None --pred_type x --attention_resolutions 32,16,8 --class_cond True --channel_mult 1,2,4 --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 128 --image_size 28 --lr 0.0001 --num_channels 64 --num_head_channels 32 --num_res_blocks 1 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --save_interval 1000 --sampling_interval 1000 --data_dir /home/datasets/c4_mnist_3000 --ref_dir /home/datasets/c4_mnist_50000 --sampling_dir /home/datasets/fid_samples_4 --resume_checkpoint /home/checkpoints/Group-Diffusion/c4_mnist_3000_eqv_ddim_ch124/
# TML2 - c4_mnist_6000 - weight-tied model - FID: 5.47
CUDA_VISIBLE_DEVICES=0 OPENAI_LOGDIR=/home/checkpoints/Group-Diffusion/c4_mnist_6000_eqv_ddim_ch124 python edm_train_fid.py --g_equiv True --g_input Z2_K --g_output C4_K --diff_type ddim --sampler ddim --data_augment 0 --eqv_reg None --pred_type x --attention_resolutions 32,16,8 --class_cond True --channel_mult 1,2,4 --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 128 --image_size 28 --lr 0.0001 --num_channels 64 --num_head_channels 32 --num_res_blocks 1 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --save_interval 1000 --sampling_interval 1000 --data_dir /home/datasets/c4_mnist_6000 --ref_dir /home/datasets/c4_mnist_50000 --sampling_dir /home/datasets/fid_samples --resume_checkpoint /home/checkpoints/Group-Diffusion/c4_mnist_6000_eqv_ddim_ch124/


######################################################################################
# Sampling code 
######################################################################################
OPENAI_LOGDIR=/home/datasets/fid_samples/ python image_sample.py --training_mode edm --g_equiv True --g_input Z2_K --g_output C4_K --batch_size 512 --sigma_max 80 --sigma_min 0.002 --s_churn 0 --steps 40 --sampler heun --model_path /home/datasets/logs/model037000.pt --attention_resolutions 32,16,8  --class_cond False --dropout 0.0 --image_size 28 --num_channels 64 --num_head_channels 32 --num_res_blocks 1 --num_samples 64 --resblock_updown True --use_fp16 False --use_scale_shift_norm False --weight_schedule karras

# Class conditional sampling code
OPENAI_LOGDIR=/home/datasets/fid_samples/ python image_sample.py --training_mode edm --g_equiv True --g_input Z2_K --g_output C4_K --num_samples 48000 --batch_size 1024 --sigma_max 80 --sigma_min 0.002 --s_churn 0 --steps 40 --sampler heun --model_path /home/checkpoints/Group-Diffusion/rot_mnist_6000/model261000.pt --attention_resolutions 32,16,8 --class_cond True --dropout 0.0 --image_size 28 --num_channels 64 --num_head_channels 32 --num_res_blocks 1 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --weight_schedule karras

# Sampling code for using torchvision grid, small batch samples used for illustration
python image_sample_grid.py --training_mode edm --g_equiv True --g_input Z2_K --g_output C4_K --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.1 --num_samples 10 --batch_size 10 --image_size 28 --num_channels 64 --num_head_channels 32 --num_res_blocks 1 --resblock_updown True --use_fp16 False --weight_schedule karras --model_path /home/checkpoints/Group-Diffusion/rot_mnist_6000/model261000.pt 


