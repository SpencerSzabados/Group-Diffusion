# File contains command line scripts for launching the training of different diffusion
# models for different image dataset and noise schedules.

######################################################################################
# Testing code 
######################################################################################

# Train model on C4 group images 
OPENAI_LOGDIR=/home/checkpoints/Group-Diffusion/c4toy_example_non_eqv python edm_train.py --g_equiv False --g_input Z2_K --g_output Z2_K --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 512 --image_size 28 --lr 0.0001 --num_channels 64 --num_head_channels 32 --num_res_blocks 1 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --save_interval 1000 --data_dir /home/datasets/c4_toy


######################################################################################
# Training group equivariant EDM models on class-conditional Rot-MNIST
######################################################################################
OPENAI_LOGDIR=/home/checkpoints/Group-Diffusion/rot_mnist_6000 mpiexec --allow-run-as-root -n 2 python edm_train.py --g_equiv True --g_input Z2_K --g_output C4_K --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 512 --image_size 28 --lr 0.0001 --num_channels 64 --num_head_channels 32 --num_res_blocks 1 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --save_interval 1000 --data_dir /home/datasets/rot_mnist_6000


######################################################################################
# Sampling code 
######################################################################################
OPENAI_LOGDIR=/home/fid_samples/ python image_sample.py --training_mode edm --g_equiv True --g_input Z2 --g_output C4 --batch_size 512 --sigma_max 80 --sigma_min 0.002 --s_churn 0 --steps 40 --sampler heun --model_path /home/datasets/logs/model037000.pt --attention_resolutions 32,16,8  --class_cond False --dropout 0.0 --image_size 28 --num_channels 64 --num_head_channels 32 --num_res_blocks 1 --num_samples 64 --resblock_updown True --use_fp16 False --use_scale_shift_norm False --weight_schedule karras

# Class conditional sampling code
OPENAI_LOGDIR=/home/fid_samples/ mpiexec --allow-run-as-root -n 2 python image_sample.py --training_mode edm --g_equiv True --g_input Z2_K --g_output C4_K --num_samples 48000 --batch_size 512 --sigma_max 80 --sigma_min 0.002 --s_churn 0 --steps 40 --sampler heun --model_path /home/checkpoints/Group-Diffusion/rot_mnist_600/model037000.pt --attention_resolutions 32,16,8 --class_cond True --dropout 0.0 --image_size 28 --num_channels 64 --num_head_channels 32 --num_res_blocks 1 --resblock_updown True --use_fp16 False --use_scale_shift_norm False --weight_schedule karras
