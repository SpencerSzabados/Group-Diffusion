"""
    Author: Spencer Szabados
    Date: 2024-01-02

    Given a list of images (categorized) compute the negative log-likelihood of generating 
    each image and save summary statistics based on the categories.

    Images are assumed to follow the naming convention "<class_label>_<image_index>.JPEG"

    Implementaion based on that provided in: 
    (https://github.com/yang-song/score_sde_pytorch/blob/main/run_lib.py)

    Launch command:
    OPENAI_LOGDIR=/home/checkpoints/temp/ python image_euler_likelihood.py --g_equiv False --g_input Z2_K --g_output Z2_K --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 100 --batch_size 100 --image_size 28 --lr 0.0001 --num_channels 64 --num_head_channels 32 --num_res_blocks 1 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --save_interval 500 --model_path /home/checkpoints/Group-Diffusion/c4toy_example_non_eqv/model016000.pt --data_dir /home/datasets/c4_toy --num_samples 1 --steps 1000 --repeats 2 --sde VESDE 
    OPENAI_LOGDIR=/home/checkpoints/temp/ python image_euler_likelihood.py --g_equiv True --g_input Z2_K --g_output C4_K --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 100 --batch_size 100 --image_size 28 --lr 0.0001 --num_channels 64 --num_head_channels 32 --num_res_blocks 1 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --save_interval 500 --model_path /home/checkpoints/Group-Diffusion/c4toy_example/model014000.pt --data_dir /home/datasets/c4_toy_rot270 --num_samples 1 --steps 1000 --repeats 2 --sde VESDE 

    CUDA_VISIBLE_DEVICES=0 OPENAI_LOGDIR=/home/checkpoints/Group-Diffusion/temp/ python image_euler_likelihood.py --g_equiv True --g_input Z2_K --g_output C4_K --diff_type ddim --sampler ddim --data_augment 0 --eqv_reg None --pred_type x --attention_resolutions 32,16,8 --class_cond True --channel_mult 1,2,4 --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 128 --batch_size 128 --image_size 28 --lr 0.0001 --num_channels 64 --num_head_channels 32 --num_res_blocks 2 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --save_interval 1000 --sampling_interval 1000 --model_path /home/checkpoints/Group-Diffusion/c4_mnist_3000_eqv_ddim_ch124_r2/model020000.pt --data_dir /home/datasets/c4_mnist_6000 --num_samples 1 --steps 100 --repeats 1 --sde VPSDE

"""

import io
import os
import argparse
import numpy as np
import torch as th

from model.utils import distribute_util
import torch.distributed as dist

from model import logger
from model.image_dataset_loader import load_data
from model.utils.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from model.utils.random_util import get_generator
from model.karras_diffusion import karras_sample

from evaluations import euler_likelihood


# -----------------------------------------------
## Preliminary function
def create_argparser():
    defaults = dict(
        data_dir="",
        model_path="",
        g_equiv=False,
        g_input=None,
        g_output=None,
        eqv_reg=None,
        pred_type='x',
        sampler="multistep",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=2048,
        steps=1,
        repeats=1,
        batch_size=-1,
        microbatch=-1,      # -1 disables microbatches
        num_samples=-1,
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        sampling_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        sde='VPSDE',
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        seed=42,
        user_id='dummy',
        slurm_id='-1',
        device='cuda:0'
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def get_data_inverse_scaler():
    """
        Inverse data normalizer.
        Data is assumed to be rescaled in the model to [-1,1],
        this rescales [-1, 1] to [0, 1] for use in likelihood computation.
    """
    return lambda x: (x + 1.) / 2.


# -----------------------------------------------
## Setup SDE equation parameters 
# The following parameter values should match those use during training of the model

def main():
    distribute_util.setup_dist()
    logger.configure()

    logger.log("\nCreating argument parser")
    args = create_argparser().parse_args()

    # print(args.user_id, args.slurm_id)
    if args.user_id != '-1':
        os.environ["SLURM_JOB_ID"] = args.slurm_id
        os.environ['USER'] = args.user_id

    # Default parameter values 
    sigma_max = args.sigma_max or 80.0
    sigma_min = args.sigma_min or 0.002
    sampler = args.sampler
    steps = args.steps or 100
    bpd_num_repeats = args.repeats or 1 # Average over the dataset this many times when computing likelihood 

    if args.sde == "VPSDE":
        sde = euler_likelihood.VPSDE(sigma_min=sigma_min, sigma_max=sigma_max, N=steps)
    elif args.sde == "VESDE":
        sde = euler_likelihood.VESDE(sigma_min=sigma_min, sigma_max=sigma_max, N=steps)
    else:
        NotImplementedError(f"SDE not implemented")

    print("sde.T: "+str(sde.T)) # DEBUG
    print("sde.N: "+str(sde.N)) # DEBUG

    likelihood_fn = euler_likelihood.get_likelihood_fn(sde, get_data_inverse_scaler())

    # Create diffusion model and load checkpoint state
    logger.log("\nCreating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        distillation=False,
    )
    model.load_state_dict(
            distribute_util.load_state_dict(args.model_path, map_location="cpu")
        )
    model.to(distribute_util.dev())

    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # Load data
    logger.log("\nCreating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        deterministic=True,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    # Compute negative log-likelihoods (bits/dim) 
    logger.log("\nComputing NLL of data...")
    bpds = []
    nfes = []
    means = []
    vars = []
    for i in range(bpd_num_repeats):
        batch, cond = next(data)
        cond = cond if args.class_cond else None
        batch = batch.to('cuda:0').float()

        bpd, z, nfe = likelihood_fn(model, batch, cond)
        bpd = bpd.detach().cpu().numpy().reshape(-1)

        bpds.extend(bpd)
        nfes.append(nfe)
        
        means.append(np.mean(bpd))
        vars.append(np.var(bpd))

    logger.log("Mean NFE: "+str(np.sum(nfe)/bpd_num_repeats))
    logger.log("NLL mean: "+str(np.mean(means)))
    logger.log("NLL var: "+str(np.var(vars)))

    logger.log("\n...Finished")
            

if __name__=="__main__":
    main()