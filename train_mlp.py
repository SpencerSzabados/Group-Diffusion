"""
    Script for training a mlp diffusion model on point data.

    Example launch command:
    CUDA_VISIBLE_DEVICES=0 OPENAI_LOGDIR=/home/sszabados/models/Group-Diffusion/logger_dir NCCL_P2P_LEVEL=NVL mpiexec -n 1 python train_mlp.py --experiment_name mlp_fa --g_equiv True --g_input C4
"""

import os
import argparse
from model.utils import distribute_util
import torch.distributed as dist
from model.utils.point_dataset_loader import load_data
from model.mlp import MLP
from model.mlp_diffusion import NoiseScheduler

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from tqdm import tqdm
from model import logger
from datetime import datetime

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def create_argparser():
    defaults = dict(
        experiment_name="mlp",
        data_dir="/home/sszabados/datasets/checkerboard/radial_checkerboard_density_dataset.npz",
        g_equiv=False,
        g_input=None,
        diff_type='pfode',
        pred_type='eps',
        eqv_reg=None,
        hidden_layers=3,
        hidden_size=128,
        emb_size=128,
        time_emb="sinusoidal",
        input_emb="sinusoidal",
        num_timesteps=80,
        beta_schedule="linear",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        ema=0.994,
        lr_anneal_steps=0,
        global_batch_size=10000,
        global_sample_size=10000,
        batch_size=-1,
        log_interval=2000,
        sample_interval=10000,
        save_interval=1000000,
        training_steps=1000000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        user_id='dummy',
        slurm_id='-1',
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def rot_fn(points, k):
    """
    Rotates a batch of [x, y] points around (0, 0) by k * 90 degrees.

    Parameters:
        points (torch.Tensor): A tensor of shape (batch_size, 2) containing [x, y] points.
    
    Returns:
        torch.Tensor: The rotated points.
    """
     # Ensure k is between 0 and 3 (for multiples of 90 degrees)
    k = k % 4
    
    # Rotation matrices for 90-degree increments
    if k == 1:  # 90 degrees counterclockwise
        rotation_matrix = th.tensor([[0, -1], [1, 0]], dtype=points.dtype, device=points.device)
    elif k == 2:  # 180 degrees
        rotation_matrix = th.tensor([[-1, 0], [0, -1]], dtype=points.dtype, device=points.device)
    elif k == 3:  # 270 degrees counterclockwise (or 90 degrees clockwise)
        rotation_matrix = th.tensor([[0, 1], [-1, 0]], dtype=points.dtype, device=points.device)
    else:  # k == 0, no rotation
        return points
    
    # Apply the rotation matrix to the batch of points
    return th.matmul(points, rotation_matrix)


def inv_rot_fn(points, k):
    """
    Rotates a batch of [x, y] points around (0, 0) by -k * 90 degrees (inverse of the rotation).

    Parameters:
        points (torch.Tensor): A tensor of shape (batch_size, 2) containing [x, y] points.
    
    Returns:
        torch.Tensor: The points rotated backward by k * 90 degrees.
    """
    # Inverse rotation is equivalent to rotating in the opposite direction by (4 - k) * 90 degrees
    return rot_fn(points, -k)
    

def update_ema(model, ema_model, ema):
    """
    Updates model weights using exponential moving average.

    Paramters:
        model (nn.Module): Current model being trained
        ema_model (nn.Module): No gradient copy of model
        ema (th.Tensor): EMA decay value

    """
    with th.no_grad():
        model_params = list(model.parameters())
        ema_params = list(ema_model.parameters())
        
        for model_param, ema_param in zip(model_params, ema_params):
            ema_param.data.mul_(ema).add_(model_param.data, alpha=(1 - ema))


def main():
    args = create_argparser().parse_args()

    # print(args.user_id, args.slurm_id)
    if args.user_id != '-1':
        os.environ["SLURM_JOB_ID"] = args.slurm_id
        os.environ['USER'] = args.user_id

    outdir = f"exps/{args.experiment_name}"
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(f"{outdir}/images/", exist_ok=True)

    distribute_util.setup_dist()
    logger.configure()

    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.log(f"[{time}]"+"="*20+"\nJob started.")
    logger.log(f"Experiment: {args.experiment_name}\n")

    logger.log("creating model and noise scheduler...")
    model = MLP(hidden_layers=args.hidden_layers, 
                hidden_size=args.hidden_size,
                emb_size=args.emb_size,
                time_emb=args.time_emb,
                input_emb=args.input_emb)
    
    noise_scheduler = NoiseScheduler(num_timesteps=args.num_timesteps,
                                     beta_schedule=args.beta_schedule)

    model = model.to(distribute_util.dev())

    logger.log("creating data loader...")
    if args.batch_size == -1:
        batch_size = args.global_batch_size // dist.get_world_size()
        sample_size = args.global_sample_size // dist.get_world_size()
        if args.global_batch_size % dist.get_world_size() != 0:
            logger.log(
                f"warning, using smaller global_batch_size of {dist.get_world_size()*batch_size} instead of {args.global_batch_size}"
            )
    else:
        batch_size = args.batch_size

    dataloader, dataset = load_data(
        data_dir=args.data_dir,
        batch_size=batch_size,
    )

    # Set up the optimizer
    optimizer = th.optim.AdamW(model.parameters(), lr=args.lr)

    # Create a deepcopy of the model to store the EMA weights
    ema_model = deepcopy(model)

    # Disable gradient tracking for the EMA model
    for param in ema_model.parameters():
        param.requires_grad = False

    global_step = 0
    frames = []
    losses = []

    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.log(f"[{time}]"+"="*20+"\nTraining model...\n")
    while global_step < args.training_steps:
        model.train()
        for step, batch in enumerate(dataloader):
            batch = batch[0]
            noise = th.randn(batch.shape)
            timesteps = th.randint(0, noise_scheduler.num_timesteps, (batch.shape[0],)).long()
            noisy = noise_scheduler.add_noise(batch, noise, timesteps)

            # put data on gpu
            batch = batch.to(distribute_util.dev())
            noise = noise.to(distribute_util.dev())
            noisy = noisy.to(distribute_util.dev())
            timesteps = timesteps.to(distribute_util.dev())

            # Compute loss
            optimizer.zero_grad()
            if args.pred_type == 'eps':
                if args.g_equiv and args.g_input == "C4":
                    loss = 0
                    for k in range(0,4):
                        noisy_rot = rot_fn(noisy, k)
                        noise_pred = inv_rot_fn(model(noisy_rot, timesteps), k)
                        loss += F.mse_loss(noise_pred, noise)
                    loss = loss/4.0
                else:
                    noise_pred = model(noisy, timesteps)
                    loss = F.mse_loss(noise_pred, noise)
            elif args.pred_type == "x":
                if args.g_equiv and args.g_input == "C4":
                    loss = 0
                    for k in range(0,4):
                        noisy_rot = rot_fn(noisy, k)
                        x_pred = inv_rot_fn(model(noisy_rot, timesteps), k)
                        loss += F.mse_loss(x_pred, batch)
                    loss = loss/4.0
                else:
                    x_pred = model(noisy, timesteps)
                    loss = F.mse_loss(x_pred, batch)
            loss.backward()

            # nn.utils.clip_grad_norm_(model.parameters(), 1.0) # TODO: removed this line to speed up convergence.
            optimizer.step()
            
            if args.ema > 0:
                # Update the EMA model after the optimizer step
                update_ema(model, ema_model, args.ema)

            global_step += 1

            if global_step % args.log_interval == 0 and global_step > 0:
                time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logger.log(f"[{time}]"+"-"*20)
                logger.log(f"Step: {global_step}")
                logger.log(f"Loss: {loss.detach().item()}")

            if global_step % args.sample_interval == 0 and global_step > 0:
                logger.log("Logging (saving) sample plot...")
                # generate data with the model to later visualize the learning process
                model.eval()
                with th.no_grad():
                    sample = th.randn(sample_size, 2)
                    timesteps = list(range(len(noise_scheduler)))[::-1]

                    for i, t in enumerate(tqdm(timesteps)):
                        sample = sample.to(distribute_util.dev())
                        t = th.from_numpy(np.repeat(t, sample_size)).long().to(distribute_util.dev())
                        residual = model(sample, t).to(distribute_util.dev())
                        sample = noise_scheduler.step(residual, t[0], sample)

                    frame = sample.detach().cpu().numpy()
                    frames.append(frame)

                    logger.log("Saving plot...")
                    plt.figure(figsize=(8, 8))
                    plt.scatter(frame[:, 0], frame[:, 1], alpha=0.5, s=1)
                    plt.axis('off')
                    plt.savefig(f"{outdir}/images/sample_{global_step}.png")
                    plt.close()
                model.train()

            if global_step % args.save_interval == 0 and global_step > 0:
                logger.log("Saving model...")
                th.save(model.state_dict(), f"{outdir}/model_{global_step}.pth")

    # print("Saving images...")
    # imgdir = f"{outdir}/images"
    # os.makedirs(imgdir, exist_ok=True)
    # frames = np.stack(frames)
    # xmin, xmax = -6, 6
    # ymin, ymax = -6, 6
    # for i, frame in enumerate(frames):
    #     plt.figure(figsize=(10, 10))
    #     plt.scatter(frame[:, 0], frame[:, 1])
    #     plt.xlim(xmin, xmax)
    #     plt.ylim(ymin, ymax)
    #     plt.savefig(f"{imgdir}/{i:04}.png")
    #     plt.close()

    print("Saving loss as numpy array...")
    np.save(f"{outdir}/loss.npy", np.array(losses))

    print("Saving frames...")
    np.save(f"{outdir}/frames.npy", frames)


if __name__ == "__main__":
    main()
