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
# TML3 - weight-tied model - FID:10.63 (min 8.86)
CUDA_VISIBLE_DEVICES=0 OPENAI_LOGDIR=/home/checkpoints/Group-Diffusion/c4_mnist_600_eqv_ddim_ch124 python edm_train_fid.py --g_equiv True --g_input Z2_K --g_output C4_K --diff_type ddim --sampler ddim --data_augment 0 --eqv_reg None --pred_type x --attention_resolutions 32,16,8 --class_cond True --channel_mult 1,2,4 --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 128 --image_size 28 --lr 0.0001 --num_channels 64 --num_head_channels 32 --num_res_blocks 1 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --save_interval 1000 --sampling_interval 1000 --data_dir /home/datasets/c4_mnist_600 --ref_dir /home/datasets/c4_mnist_50000 --sampling_dir /home/datasets/fid_samples --resume_checkpoint /home/checkpoints/Group-Diffusion/c4_mnist_600_eqv_ddim_ch124/
# TML3 - reg model - FID
CUDA_VISIBLE_DEVICES=1 OPENAI_LOGDIR=/home/checkpoints/Group-Diffusion/c4_mnist_600_reg_ddim_ch124 python edm_train_fid.py --g_equiv False --g_input Z2_K --g_output C4_K --diff_type ddim --sampler ddim --data_augment 0 --eqv_reg C4 --pred_type x --attention_resolutions 32,16,8 --class_cond True --channel_mult 1,2,4 --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 128 --image_size 28 --lr 0.0001 --num_channels 64 --num_head_channels 32 --num_res_blocks 1 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --save_interval 1000 --sampling_interval 1000 --data_dir /home/datasets/c4_mnist_600 --ref_dir /home/datasets/c4_mnist_50000 --sampling_dir /home/datasets/fid_samples_3 --resume_checkpoint /home/checkpoints/Group-Diffusion/c4_mnist_600_reg_ddim_ch124/
# TML3 - weight-tied model - FID:
CUDA_VISIBLE_DEVICES=0 OPENAI_LOGDIR=/home/checkpoints/Group-Diffusion/c4_mnist_3000_eqv_ddim_ch124 python edm_train_fid.py --g_equiv True --g_input Z2_K --g_output C4_K --diff_type ddim --sampler ddim --data_augment 0 --eqv_reg None --pred_type x --attention_resolutions 32,16,8 --class_cond True --channel_mult 1,2,4 --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 128 --image_size 28 --lr 0.0001 --num_channels 64 --num_head_channels 32 --num_res_blocks 1 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --save_interval 1000 --sampling_interval 1000 --data_dir /home/datasets/c4_mnist_3000 --ref_dir /home/datasets/c4_mnist_50000 --sampling_dir /home/datasets/fid_samples_4 --resume_checkpoint /home/checkpoints/Group-Diffusion/c4_mnist_3000_eqv_ddim_ch124/


# TML3 - weight-tied model
CUDA_VISIBLE_DEVICES=1 OPENAI_LOGDIR=/home/checkpoints/Group-Diffusion/c4_mnist_600_eqv_ddim_ch144 python edm_train_fid.py --g_equiv True --g_input Z2_K --g_output C4_K --diff_type ddim --sampler ddim --data_augment 0 --eqv_reg None --pred_type x --class_cond True --channel_mult 1,4,4 --attention_resolutions 32,16,8  --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 128 --image_size 28 --lr 0.0001 --num_channels 64 --num_head_channels 32 --num_res_blocks 1 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --save_interval 1000 --sampling_interval 1000 --data_dir /home/datasets/c4_mnist_600 --ref_dir /home/datasets/c4_mnist_50000 --sampling_dir /home/datasets/fid_samples_2 --resume_checkpoint /home/checkpoints/Group-Diffusion/c4_mnist_600_eqv_ddim_ch144/
# TML3 
CUDA_VISIBLE_DEVICES=1 OPENAI_LOGDIR=/home/checkpoints/Group-Diffusion/c4_mnist_600_ddim_ch144_reg python edm_train_fid.py --g_equiv True --g_input Z2_K --g_output C4_K --diff_type ddim --sampler ddim --data_augment 0 --eqv_reg C4 --pred_type x --class_cond True --channel_mult 1,4,4 --attention_resolutions 32,16,8  --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 128 --image_size 28 --lr 0.0001 --num_channels 64 --num_head_channels 32 --num_res_blocks 1 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --save_interval 1000 --sampling_interval 1000 --data_dir /home/datasets/c4_mnist_600 --ref_dir /home/datasets/c4_mnist_50000 --sampling_dir /home/datasets/fid_samples_2 --resume_checkpoint /home/checkpoints/Group-Diffusion/c4_mnist_600_ddim_ch144_reg/
# TML3 

######################################################################################
# Sampling code 
######################################################################################
OPENAI_LOGDIR=/home/datasets/fid_samples/ python image_sample.py --training_mode edm --g_equiv True --g_input Z2_K --g_output C4_K --batch_size 512 --sigma_max 80 --sigma_min 0.002 --s_churn 0 --steps 40 --sampler heun --model_path /home/datasets/logs/model037000.pt --attention_resolutions 32,16,8  --class_cond False --dropout 0.0 --image_size 28 --num_channels 64 --num_head_channels 32 --num_res_blocks 1 --num_samples 64 --resblock_updown True --use_fp16 False --use_scale_shift_norm False --weight_schedule karras

# Class conditional sampling code
OPENAI_LOGDIR=/home/datasets/fid_samples/ python image_sample.py --training_mode edm --g_equiv True --g_input Z2_K --g_output C4_K --num_samples 48000 --batch_size 1024 --sigma_max 80 --sigma_min 0.002 --s_churn 0 --steps 40 --sampler heun --model_path /home/checkpoints/Group-Diffusion/rot_mnist_6000/model261000.pt --attention_resolutions 32,16,8 --class_cond True --dropout 0.0 --image_size 28 --num_channels 64 --num_head_channels 32 --num_res_blocks 1 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --weight_schedule karras

# Sampling code for using torchvision grid, small batch samples used for illustration
python image_sample_grid.py --training_mode edm --g_equiv True --g_input Z2_K --g_output C4_K --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.1 --num_samples 10 --batch_size 10 --image_size 28 --num_channels 64 --num_head_channels 32 --num_res_blocks 1 --resblock_updown True --use_fp16 False --weight_schedule karras --model_path /home/checkpoints/Group-Diffusion/rot_mnist_6000/model261000.pt 


