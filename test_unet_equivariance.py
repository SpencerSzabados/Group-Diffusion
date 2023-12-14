
import numpy as np
import torch as pt

import argparse

from model import logger
from model.model_modules.equivariant_layers import *
from model.model_modules.equivariant_blocks import *
from model.utils import distribute_util
from model.resample import create_named_schedule_sampler
from model.utils.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import os

def create_argparser():
    defaults = dict(
        data_dir="",
        g_equiv=False,
        g_input=None,
        g_output=None,
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=1,
        batch_size=-1,
        microbatch=-1,      # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        user_id='dummy',
        slurm_id='-1'
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def test_unet_pooling_equivariance():
    """
    Function for testing if the implemented unet is equivariant.

    Shell launch command:
    python test_unet_equivariance.py --g_equiv True --g_input Z2 --g_output C4 --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False --dropout 0.0 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 1 --image_size 28 --lr 0.00001 --num_channels 64 --num_head_channels 32 --num_res_blocks 1 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras
    """
    args = create_argparser().parse_args()

    device = pt.device("cuda:0")

    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    # model.to(device)

    logger.log("Generating data...")
    # im = np.random.randn(1, 1, 11, 11)
    im = pt.rand(1,3,28,28)
    imT = pt.rot90(im)
    timesteps=pt.from_numpy(np.arange(0,1,0.1))
    
    print("Image : "+str(im))
    print("Image.T : "+str(im))
    
    logger.log("Passing data through unet...")
    im.to(device)
    imT.to(device)
    timesteps.to(device)

    y = model(im, timesteps)
    y = pt.mean(y, dim=2)

    yT = model(imT, timesteps)
    yT = pt.mean(yT, dim=2)

    logger.log("Comparing unet output...")
    difference = y-yT
    error = pt.sum(difference)

    print("Error : "+str(error))
    print("Difference: "+str(difference))

if __name__=="__main__":
    test_unet_pooling_equivariance()