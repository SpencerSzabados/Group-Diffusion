"""
    Author: Spencer Szabados
    Date: 2024-01-02

    Given a list of images (categorized) compute the negative log-likelihood of generating 
    each image and save summary statistics based on the categories.

    Images are assumed to follow the naming convention "<class_label>_<image_index>.JPEG"

    Implementaion based on that provided in: 
    (https://github.com/yang-song/score_sde_pytorch/blob/main/run_lib.py)

    Launch command:
    OPENAI_LOGDIR=/home/checkpoints/temp/ python image_euler_likelihood.py --g_equiv True --g_input Z2_K --g_output C4_K --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 3 --batch_size 3 --image_size 28 --lr 0.0001 --num_channels 64 --num_head_channels 32 --num_res_blocks 1 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --save_interval 500 --model_path /home/checkpoints/Group-Diffusion/model014000.pt --data_dir /home/datasets/rot_mnist_600 --num_samples 1 --sde VESDE 
    OPENAI_LOGDIR=/home/checkpoints/temp/ python image_euler_likelihood.py --g_equiv True --g_input Z2_K --g_output C4_K --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 3 --batch_size 3 --image_size 28 --lr 0.0001 --num_channels 64 --num_head_channels 32 --num_res_blocks 1 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --save_interval 500 --model_path /home/checkpoints/Group-Diffusion/model014000.pt --data_dir /home/datasets/c4test --num_samples 1 --sde VESDE 
"""

import io
import os
import argparse
import numpy as np
import torch as th
import torchvision

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
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=2048,
        microbatch=-1,      # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        sde='VESDE',
        user_id='dummy',
        slurm_id='-1',
        training_mode="edm",
        generator="determ",
        clip_denoised=True,
        num_samples=16,
        batch_size=16,
        sampler="euler",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        steps=100,
        seed=42,
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
    sigma_max = 80.0 or args.sigma_max
    sigma_min = 0.002 or args.sigma_min
    num_timesteps = args.steps or 100
    sampling_eps = 1e-5 or args.sampling_eps
    bpd_num_repeats = 1 # Average over the dataset this many times when computing likelihood 

    if args.sde == "VESDE":
        # Varaince exploding SDE
        sde = euler_likelihood.VESDE(sigma_min=sigma_min, sigma_max=sigma_max, N=num_timesteps)
        print("sde.T: "+str(sde.T)) # DEBUG
        print("sde.N: "+str(sde.N)) # DEBUG
    else:
        NotImplementedError(f"SDE not implemented")

    likelihood_fn = euler_likelihood.get_likelihood_fn(sde, get_data_inverse_scaler()) # DEBUG: remove esp argument

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
        deterministic=False,
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

    exit()

    # Samples images using z prior
    logger.log("sampling...")
    if args.sampler == "multistep":
        assert len(args.ts) > 0
        ts = tuple(int(x) for x in args.ts.split(","))
    else:
        ts = None

    all_images = []
    all_labels = []
    generator = get_generator(args.generator, args.num_samples, z)
    grid_img = torchvision.utils.make_grid(z, nrow = 1, normalize = True)
    torchvision.utils.save_image(grid_img, f'tmp_imgs/z_sample.pdf')

    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=distribute_util.dev()
            )
            model_kwargs["y"] = classes

        sample = karras_sample(
            diffusion,
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            steps=args.steps,
            model_kwargs=model_kwargs,
            device=distribute_util.dev(),
            clip_denoised=args.clip_denoised,
            sampler=args.sampler,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            s_churn=args.s_churn,
            s_tmin=args.s_tmin,
            s_tmax=args.s_tmax,
            s_noise=args.s_noise,
            generator=generator,
            ts=ts,
        )

        grid_img = torchvision.utils.make_grid(sample, nrow = 8, normalize = True)
        torchvision.utils.save_image(grid_img, f'tmp_imgs/z_samples.pdf')
        

    dist.barrier()
    logger.log("sampling complete")
            

if __name__=="__main__":
    main()