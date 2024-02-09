import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import RAdam
import torchvision
from . import distribute_util
from .. import logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from ..resample import LossAwareSampler, UniformSampler

from ..karras_diffusion import karras_sample
from .random_util import get_generator

from .fp16_util import (
    get_param_groups_and_shapes,
    make_master_params,
    master_params_to_model_params,
)
import numpy as np

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        diff_type,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        pred_type='pfode',
        sampling_interval=0,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        start_ema=0.0,
        self_cond=False,
        eqv_reg=None,
        data_augment=0,
    ):
        self.model = model
        self.diffusion = diffusion
        self.pred_type = pred_type
        self.eqv_reg = eqv_reg
        self.self_cond = self_cond
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.global_batch = self.batch_size * dist.get_world_size()
        self.lr = lr
        self.lr_anneal_steps = lr_anneal_steps
        self.weight_decay = weight_decay
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.target_ema = start_ema
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.step = 0
        self.resume_step = 0
        if sampling_interval > 0:
            if sampling_interval < save_interval:
                logger.log("Sampling_interval < save_interval, setting sampling_interval=save_interval.")
                self.sampling_interval = self.save_interval
            else:
                self.sampling_interval = sampling_interval
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
    
        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = RAdam(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]
            
        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[distribute_util.dev()],
                output_device=distribute_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

        if self.eqv_reg is not None:
            self.target_model = copy.deepcopy(self.model)
            if self.resume_step:
                self._load_and_sync_target_parameters()

            self.target_model.requires_grad_(False)
            self.target_model.train()
            # self.target_model_master_params = list(self.target_model.parameters())
            self.target_model_param_groups_and_shapes = get_param_groups_and_shapes(
                self.target_model.named_parameters()
            )
            self.target_model_master_params = make_master_params(
                self.target_model_param_groups_and_shapes
            )

        self.step = self.resume_step
        self.resume_step = 0

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or find_resume_checkpoint_aux(self.resume_checkpoint)

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    distribute_util.load_state_dict(
                        resume_checkpoint, map_location=distribute_util.dev()
                    ),
                )

        distribute_util.sync_params(self.model.parameters())
        distribute_util.sync_params(self.model.buffers())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = distribute_util.load_state_dict(
                    ema_checkpoint, map_location=distribute_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        distribute_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = distribute_util.load_state_dict(
                opt_checkpoint, map_location=distribute_util.dev()
            )
            self.opt.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"Cannot find checkpoint file at {opt_checkpoint}")

    def run_loop(self):
        self.model.train()
        saved = False
        while (
            not self.lr_anneal_steps
            or self.step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            saved = False
            if (self.save_interval != -1
                and self.step % self.save_interval == 0
            ):
                self.save()
                saved = True
                th.cuda.empty_cache()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return

            if self.step % self.log_interval == 0:
                logger.dumpkvs()

            if hasattr(self, 'sampling_interval'):
                if self.step % self.sampling_interval == 0:
                    self.model.eval()
                    return self.step, self.ema_rate

        # Save the last checkpoint if it wasn't already saved.
        if not saved:
            self.save()

    def run_step(self, batch, cond):
        self._anneal_lr()
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
            if self.eqv_reg:
                self._update_target_ema()
            self.step += 1
        self._anneal_lr()
        self.log_step()
        return took_step
    
    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()

        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(distribute_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(distribute_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], distribute_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                pred_type=self.pred_type,
                model_kwargs=micro_cond,
                target_model=self.target_model if self.eqv_reg else None,
                eqv_reg=self.eqv_reg,
                self_cond=self.self_cond
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _update_target_ema(self):
        with th.no_grad():
            if self.use_fp16: # TODO: combine into one case.
                update_ema(
                    self.target_model_master_params,
                    self.mp_trainer.master_params,
                    rate=self.target_ema
                )
                master_params_to_model_params(
                    self.target_model_param_groups_and_shapes,
                    self.target_model_master_params,
                )
            else: 
                update_ema(
                    list(self.target_model.parameters()),
                    self.mp_trainer.master_params,
                    rate=self.target_ema
                )

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

            if self.eqv_reg is not None:
                logger.log("Saving target model state...")
                filename = f"target_model{self.step:06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(self.target_model.state_dict(), f)

        # Save model parameters last to prevent race conditions where a restart
        # loads model at step N, but opt/ema state isn't saved for step N.
        save_checkpoint(0, self.mp_trainer.master_params)
        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def obtain_slurm_ckpt_dir():
    SLURM_JOB_ID = os.environ['SLURM_JOB_ID']
    USER = os.environ['USER']
    ckpt_dir=f'/checkpoint/{USER}/{SLURM_JOB_ID}'
    return ckpt_dir


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    try:
        ckpt_dir = obtain_slurm_ckpt_dir()
        ckpt_file_path = find_resume_checkpoint_aux(ckpt_dir)
        logger.info(f'DETECT SLURM_JOB_ID; ckpt_file_path is set to {ckpt_file_path}')
        return ckpt_file_path

    except KeyError:
        logger.info('CANNOT DETECT SLURM_JOB_ID; NOT RUN ON A CLUSTER')
        return None


def find_resume_checkpoint_aux(ckpt_dir):
    # search the lastest checkpoint in ckpt_dir

    if not os.path.exists(ckpt_dir):
        logger.warn(f'CANNOT FIND: {ckpt_dir}')
        return None

    last_resume_step = -1

    for filename in os.listdir(ckpt_dir):
        if ('model' in filename) and ('.pt' in filename):
            resume_step = parse_resume_step_from_filename(filename)
            if resume_step > last_resume_step:
                last_resume_step = resume_step

    if last_resume_step == -1:
        return
    else:
        last_resume_step = str(last_resume_step).zfill(6)
        return os.path.join(ckpt_dir, f'model{last_resume_step}.pt')


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
