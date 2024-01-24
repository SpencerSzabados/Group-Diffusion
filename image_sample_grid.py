"""
    Image (generation) sampling script used for generating a batch of image samples
    from a model and save them as image grid.
"""

import os
import argparse

import numpy as np
from model.utils import distribute_util
import torch as th
import torch.distributed as dist
import torchvision

from model import logger
from model.utils.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from model.utils.random_util import get_generator
from model.karras_diffusion import karras_sample


def create_argparser():
    defaults = dict(
        training_mode="edm",
        generator="determ",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        sampler="heun",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        steps=40,
        model_path="",
        seed=42,
        ts="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    args = create_argparser().parse_args()

    if args.batch_size > args.num_samples:
        logger.log("batch_size > num_samples; reducing batch_size.")
        args.batch_size = args.num_samples

    distribute_util.setup_dist()
    logger.configure()

    if "consistency" in args.training_mode:
        distillation = True
    else:
        distillation = False

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        distillation=distillation,
    )
    model.load_state_dict(
        distribute_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(distribute_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    if args.sampler == "multistep":
        assert len(args.ts) > 0
        ts = tuple(int(x) for x in args.ts.split(","))
    else:
        ts = None

    all_images = []
    all_labels = []
    generator = get_generator(args.generator, args.num_samples, args.seed)

    # Code for performing incremental image sampling during training.
    # TODO: Make this function more general and accept model paramters during sampling 
    #       rather than the hard coded values used currently.
    #       This sould be modified if training on a dataset of different resolution.
    logger.log("generating samples...")

    while len(all_images)*args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.arange(start=0, end=9, dtype=int, device=distribute_util.dev())
            i = 0
            while len(classes) < args.batch_size and i < args.num_samples:
                classes = classes.append[classes[i]]
                i += 1
            classes = classes.reshape(args.batch_size,)
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

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    # Save the generated sample images
    logger.log("sampled tensor shape: "+str(sample.shape))
    grid_img = torchvision.utils.make_grid(sample, nrow = 10, normalize = True)
    torchvision.utils.save_image(grid_img, f"tmp_imgs/generated_sample.pdf")

    logger.log("sampling complete")


if __name__ == "__main__":
    main()
