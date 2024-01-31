"""
    Author: Spencer Szabados
    Date: 2024-01-02

    Given a list of images (categorized) compute the negative log-likelihood of generating 
    each image and save summary statistics based on the categories.

    Images are assumed to follow the naming convention "<class_label>_<image_index>.JPEG"

    Implementaion based on that provided in: 
    (https://github.com/yang-song/score_sde_pytorch/blob/main/run_lib.py)

    Launch command:
    OPENAI_LOGDIR=/home/checkpoints/temp/ python image_likelihood.py --g_equiv True --g_input Z2_K --g_output C4_K --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 128 --batch_size 128 --image_size 28 --lr 0.0001 --num_channels 64 --num_head_channels 32 --num_res_blocks 1 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --save_interval 500 --model_path /home/checkpoints/Group-Diffusion/model014000.pt --data_dir /home/datasets/c4test_rot90 --num_samples 1 --sde VESDE 
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

from evaluations import sde_lib
from evaluations import likelihood


# -----------------------------------------------
## Preliminary function

def create_argparser():
    defaults = dict(
        data_dir="",
        model_path="",
        g_equiv=False,
        g_input=None,
        g_output=None,
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=2048,
        num_samples=None,
        batch_size=-1,
        microbatch=-1,      # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        sde='VPSDE',
        user_id='dummy',
        slurm_id='-1'
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
        sde = likelihood.VPSDE(sigma_min=sigma_min, sigma_max=sigma_max, N=steps)
    elif args.sde == "VESDE":
        sde = likelihood.VESDE(sigma_min=sigma_min, sigma_max=sigma_max, N=steps)
    else:
        NotImplementedError(f"SDE not implemented")

    print("sde.T: "+str(sde.T)) # DEBUG
    print("sde.N: "+str(sde.N)) # DEBUG

    likelihood_fn = likelihood.get_likelihood_fn(sde, get_data_inverse_scaler(), method='RK23', eps=0.01) # DEBUG: remove esp argument

    # Create diffusion model and load checkpoint state
    logger.log("\nCreating model and diffusion...")
    model, _ = create_model_and_diffusion(
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
    for i in range(bpd_num_repeats):
        for batch_id in range(int(args.num_samples)):
            batch, cond = next(data)
            batch = batch.to('cuda:0').float()
            bpd, z, nfe = likelihood_fn(model, batch)
            bpd = bpd.detach().cpu().numpy().reshape(-1)
            bpds.extend(bpd)
            # logger.info("ckpt: %d, repeat: %d, batch: %d, mean bpd: %6f" % (ckpt, repeat, batch_id, np.mean(np.asarray(bpds))))
            bpd_round_id = batch_id+int(args.num_samples)*i
            # Save bits/dim to disk
            ## DEBUG
            print("NFE: "+str(nfe))
            print("NLL mean: "+str(np.mean(bpd)))
            print("NLL var: "+str(np.var(bpd)))
            

if __name__=="__main__":
    main()