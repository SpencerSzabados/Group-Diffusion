"""
    Script for training a diffusion model on image data using EDM (https://github.com/NVlabs/edm)
    methodology. This script is modified to pause training after a selected number of steps and 
    compute the current fid score then resume training. 
"""

import os
from pathlib import Path
from PIL import Image

import argparse
from model import logger
from model.image_dataset_loader import load_data
from model.utils import distribute_util
from model.resample import create_named_schedule_sampler
from model.utils.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from model.utils.train_util import TrainLoop
import torch.distributed as dist
from model.utils.random_util import get_generator
from model.karras_diffusion import *
from evaluations.fid_score import calculate_fid_given_paths

def create_argparser():
    defaults = dict(
        data_dir="",
        ref_dir="",
        sampling_dir="",
        g_equiv=False,
        g_input=None,
        g_output=None,
        channel_mult="",
        diff_type='pfode',
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=2048,
        batch_size=-1,
        microbatch=-1,      # -1 disables microbatches
        start_ema=None,
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        num_samples=50000,
        sampling_interval=0,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        user_id='dummy',
        slurm_id='-1',
        data_augment=0,
        generator="determ",
        sampler='euler',
        pred_type='x',
        eqv_reg=None,
        schedule_sampler="uniform",
        clip_denoised=True,
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        steps=40,
        model_path="",
        seed=42,
        ts="",
        save_as="npy"
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def calculate_fid(diffusion, model, args, step):
    with th.no_grad():
        model.eval()
        logger.log("Called calculate_fid.")
        logger.log("Sampling images...")
        Path(args.sampling_dir).mkdir(parents=True, exist_ok=True)
        if args.sampler == "multistep":
            assert len(args.ts) > 0
            ts = tuple(int(x) for x in args.ts.split(","))
        else:
            ts = None

        all_images = []
        all_labels = []
        generator = get_generator(args.generator, args.num_samples, args.seed)

        # TODO: Determine a better way to set this parameter
        if args.batch_size < 0:
            args.batch_size = args.global_batch_size

        while len(all_images) * args.batch_size < args.num_samples:
            model_kwargs = {}
            if args.class_cond:
                classes = th.randint(
                    low=0, high=10, size=(args.batch_size,), device=distribute_util.dev()
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
                pred_type=args.pred_type,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                s_churn=args.s_churn,
                s_tmin=args.s_tmin,
                s_tmax=args.s_tmax,
                s_noise=args.s_noise,
                generator=generator,
                ts=ts,
            )
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

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

        arr = np.concatenate(all_images, axis=0)
        arr = arr[: args.num_samples]
        if args.class_cond:
            label_arr = np.concatenate(all_labels, axis=0)
            label_arr = label_arr[: args.num_samples]
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(args.sampling_dir, f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)
        logger.log("sampling complete.")

        logger.log("extracting images...")
        filename = Path(args.sampling_dir).stem # TODO: The filename and dir2img create a directoy in the wrong location currently. 
        dir2img = f"{filename}/images"
        Path(dir2img).mkdir(parents=True, exist_ok=True)
        imgs = dict(np.load(out_path))['arr_0']
        num_img = len(imgs)
        for i in range(num_img):
            im = Image.fromarray(np.squeeze(imgs[i]))
            im.save(os.path.join(dir2img, f'{i}.JPEG'))
        logger.info(f'Image extraction completed (Total: {num_img})')

        logger.log("Computing current fid...")
        fid_value = 0
        try:
            fid_value = calculate_fid_given_paths(
                paths=[args.ref_dir, dir2img],
                batch_size=args.batch_size,
                device='cuda',
                dims=2048,
                img_size=args.image_size,
                num_workers=dist.get_world_size(),
                eqv=args.g_output.split('_')[0]
            )
        except ValueError:
            fid_value = np.inf
        logger.log(f"Steps: {step}, FID: {fid_value}")
        
    return fid_value

def main():
    logger.log("Creating argparser...")
    args = create_argparser().parse_args()

    # print(args.user_id, args.slurm_id)
    if args.user_id != '-1':
        os.environ["SLURM_JOB_ID"] = args.slurm_id
        os.environ['USER'] = args.user_id

    distribute_util.setup_dist()
    logger.configure()


    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.to(distribute_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("Creating data loader...")
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
    logger.log("Creating trainloop object...")
    trainloop = TrainLoop(
        model=model,
        diffusion=diffusion,
        diff_type=args.diff_type,
        pred_type=args.pred_type,
        data=data,
        batch_size=batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        start_ema=args.start_ema if hasattr(args, 'start_ema') else None,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        sampling_interval=args.sampling_interval,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        data_augment=args.data_augment
    )

    # Training and sampling loop
    logger.log("Training...")
    while True:
        if args.sampling_interval > 0:
            step, ema_rate = trainloop.run_loop()
            # Compute fid of model 
            calculate_fid(diffusion, model, args, step=step)
        else:
            trainloop.run_loop()


if __name__ == "__main__":
    main()
