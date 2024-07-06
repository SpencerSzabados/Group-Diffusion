"""
    Script for training a diffusion model on image data using EDM (https://github.com/NVlabs/edm)
    methodology. 
"""

import argparse

from model import logger
from model.image_dataset_loader import load_data
from model.utils import distribute_util
from model.resample import create_named_schedule_sampler
from model.utils.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from model.utils.train_util import TrainLoop
import torch.distributed as dist
import os

def create_argparser():
    defaults = dict(
        data_dir="",
        g_equiv=False,
        g_input=None,
        g_output=None,
        self_cond=False,
        diff_type='pfode',
        pred_type='x',
        eqv_reg=None,
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=2048,
        batch_size=-1,
        microbatch=-1,      # -1 disables microbatches
        start_ema=0.95,
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        user_id='dummy',
        slurm_id='-1',
        data_augment=0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    args = create_argparser().parse_args()

    # print(args.user_id, args.slurm_id)
    if args.user_id != '-1':
        os.environ["SLURM_JOB_ID"] = args.slurm_id
        os.environ['USER'] = args.user_id

    distribute_util.setup_dist()
    logger.configure()


    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.to(distribute_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    if args.batch_size == -1:
        batch_size = args.global_batch_size // dist.get_world_size()
        if args.global_batch_size % dist.get_world_size() != 0:
            logger.log(
                f"warning, using smaller global_batch_size of {dist.get_world_size()*batch_size} instead of {args.global_batch_size}"
            )
    else:
        batch_size = args.batch_size

    data = load_data(
        data_dir=args.data_dir,
        batch_size=batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        self_cond=args.self_cond,
        diff_type=args.diff_type,
        pred_type=args.pred_type,
        eqv_reg=args.eqv_reg,
        data=data,
        data_augment=args.data_augment,
        batch_size=batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        start_ema=args.start_ema if hasattr(args, 'start_ema') else None,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


if __name__ == "__main__":
    main()