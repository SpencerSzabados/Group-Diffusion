"""
    File implements group invariant/equivariant unet architecture using the equivariant layers and blocks 
    defined in the model_modeules folder.

    Implementation is a modified version of the Consistency Model from (https://github.com/openai/consistency_models)
"""

from abc import abstractmethod

import math
import torch as th
from pathlib import Path
from functools import partial
from multiprocessing import cpu_count
import torchvision

from .utils.fp16_util import convert_module_to_f16, convert_module_to_f32
from .utils.nn import (
    checkpoint,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

from .model_modules.equivariant_layers import *
from .model_modules.equivariant_blocks import *


### ---[ Consistency model unet ]----------------
class UNetModel(nn.Module):
    """
    The full group equivariant UNet model with attention and timestep embedding.

    :param in_channels: number of channels in the input Tensor.
    :param model_channels: number of out_channels for first ResBlock following time 
        embedding of input. This value is later scaled by the entreis of channel_mult 
        to build subsequent layers.
    :param out_channels: number of channels in the returned output Tensor.
    :param num_res_blocks: number of residual blocks per up-down-sample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple. For example, if this 
        contains 4, then at 4x downsampling, attention will be used.
    :param dropout: dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet, used to scale
        layer input and output channels.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
        a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
        of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for
        potentially increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        g_equiv=False,
        g_input=None,
        g_output=None,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        data_augment=0,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.g_equiv = g_equiv
        self.g_input = g_input
        self.g_output = g_output
        self.kernel_size = 3 # TODO: Define these values using arguments to the model
        self.padding = 1     # TODO: Define these values using arguments to the model
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = 4*model_channels
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels) # Number in_channels in first ResBlock after time embedding
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(gconv_nd(dims, g_equiv=self.g_equiv, g_input=self.g_input, g_output=self.g_output, in_channels=in_channels, out_channels=ch, kernel_size=self.kernel_size, padding=self.padding))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    GResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        g_equiv=self.g_equiv,
                        g_input=self.g_output,
                        out_channels=int(mult*model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        GAttentionBlock(
                            ch,
                            g_equiv=self.g_equiv,
                            g_input=self.g_output,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        GResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            g_equiv=self.g_equiv,
                            g_input=self.g_output,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else GDownsample(
                                ch,
                                use_conv=self.conv_resample,
                                g_equiv=self.g_equiv,
                                g_input=self.g_output,
                                dims=dims, 
                                out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            # GSymmetrize(
            #     g_output=self.g_output
            # ),
            GResBlock(
                ch,
                time_embed_dim,
                dropout,
                g_equiv=self.g_equiv,
                g_input=self.g_output,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            GAttentionBlock(
                ch,
                g_equiv=self.g_equiv,
                g_input=self.g_output,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            GResBlock(
                ch,
                time_embed_dim,
                dropout,
                g_equiv=self.g_equiv,
                g_input=self.g_output,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    GResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        g_equiv=self.g_equiv,
                        g_input=self.g_output,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        GAttentionBlock(
                            ch,
                            g_equiv=self.g_equiv,
                            g_input=self.g_output,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        GResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            g_equiv=self.g_equiv,
                            g_input=self.g_output,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else GUpsample(
                            ch,
                            use_conv=self.conv_resample,
                            g_equiv=self.g_equiv,
                            g_input=self.g_output,
                            dims=dims, 
                            out_channels=out_ch
                        )
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(gconv_nd(dims, self.g_equiv, self.g_output, self.g_output, input_ch, out_channels, kernel_size=self.kernel_size, padding=self.padding)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N, C, H, W] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N, C, H, W] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y) 

        h = x.type(self.dtype)

        for module in self.input_blocks:
            h_org = module(h, emb)
            h = h_org
            hs.append(h)     

        h = self.middle_block(h, emb)

        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)

        h = h.type(x.dtype)
        return self.out(h)
    

