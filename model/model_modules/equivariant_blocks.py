"""
    File contains function definitons for group (mostly C_4) equivariant (or invariant) 
    convolution, pooling, and transformation blocks. Makes use of 'equivariant_layers.py'.
     
    These are needed as standard convolutional and pooling layers may destory the
    invariant properties of the generator or disciminator process due to flattening.
"""
from abc import abstractmethod
import math
import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
#
from .equivariant_layers import GUpsample, GDownsample, gconv_nd
from .unet_layers import Upsample, Downsample
from ..utils.nn import (
    checkpoint,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

### ---[ Time step embedding blocks ]------------

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x
    

### ---[ Generator blocks ]------------------

class GResBlock(TimestepBlock):
    """
    A group equivariant residual block.
    """

    def __init__(
        self,
        in_channels, 
        emb_channels,
        dropout,
        g_equiv,
        g_input,
        out_channels=None,          # Default to in_channels if not set
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,                     # Select Conv1D,Conv2d,Conv3D
        kernel_size=5,
        padding=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
    
        self.g_equiv = g_equiv or False
        self.g_input = g_input or None
        
        # Detect input group and scale number of channels based on corresponding scaling factor
        # See 'equivariant_layers.py' for the origin of these values.
        nti = 1
        if self.g_equiv:
            if self.g_input == 'Z2':
                nti = 1
            elif self.g_input == 'H' or self.g_input == 'V':
                nti = 2
            elif self.g_input == 'C4':
                nti = 4
            elif self.g_input == 'D4':
                nti = 8
            else:    
                raise ValueError(f"unsupported g_input in GResBock(): {g_input}")
        
        self.in_channels = in_channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or in_channels
        
        self.dropout = dropout
        self.use_conv = use_conv
        self.kernel_size = kernel_size
        self.padding = padding
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(in_channels),
            nn.SiLU(),
            gconv_nd(dims, g_equiv=self.g_equiv, g_input=self.g_input, g_output=self.g_input, in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, padding=self.padding),
        )

        self.updown = up or down

        if up:
            self.h_upd = GUpsample(self.in_channels, use_conv=False, g_equiv=True, g_input=self.g_input, dims=dims)
            self.x_upd = GUpsample(self.in_channels, use_conv=False, g_equiv=True, g_input=self.g_input, dims=dims)
        elif down:
            self.h_upd = GDownsample(self.in_channels, use_conv=False, g_equiv=True, g_input=self.g_input, dims=dims)
            self.x_upd = GDownsample(self.in_channels, use_conv=False, g_equiv=True, g_input=self.g_input, dims=dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, 2*self.out_channels if use_scale_shift_norm else self.out_channels)
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                gconv_nd(dims, g_equiv=self.g_equiv, g_input=self.g_input, g_output=self.g_input, in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, padding=1)
            )
        )

        if self.out_channels == self.in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = gconv_nd(dims, self.g_equiv, self.g_input, self.g_input, self.in_channels, self.out_channels, kernel_size=self.kernel_size, padding=1)
        else:
            self.skip_connection = gconv_nd(dims, self.g_equiv, self.g_input, self.g_input, self.in_channels, self.out_channels, kernel_size=1)

    def _forward(self, x, emb):

        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            
            h = self.in_layers(x)

            # h_rot90 = self.in_layers[-1](pt.rot90(x, 1, dims = [-1, -2]))

            # h = x

            # h_rot90 = pt.rot90(x, 1, dims = [-1, -2])


        #     print('eqv:', pt.abs(h_rot90 - pt.rot90(h, 1, dims=[-1,-2])).max())

        # exit()





        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

            
        if self.use_scale_shift_norm:

            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = pt.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        
        return self.skip_connection(x) + h

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )
    

### ---[ Attention blocks ]----------------------

### Function:
# Supporting function for attention layers 
def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += pt.DoubleTensor([matmul_ops])

class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/output heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads
        from einops import rearrange
        self.rearrange = rearrange

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        qkv = qkv.half()

        qkv =   self.rearrange(
            qkv, "b (three h d) s -> b s three h d", three=3, h=self.n_heads
        ) 
        q, k, v = qkv.transpose(1, 3).transpose(3, 4).split(1, dim=2)
        q = q.reshape(bs*self.n_heads, ch, length)
        k = k.reshape(bs*self.n_heads, ch, length)
        v = v.reshape(bs*self.n_heads, ch, length)

        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = pt.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = pt.softmax(weight, dim=-1).type(weight.dtype)
        a = pt.einsum("bts,bcs->bct", weight, v)
        a = a.float()
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

class QKVFlashAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        batch_first=True,
        attention_dropout=0.0,
        causal=False,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        from einops import rearrange
        from flash_attn.flash_attention import FlashAttention

        assert batch_first
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal

        assert (
            self.embed_dim % num_heads == 0
        ), "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim in [16, 32, 64], "Only support head_dim == 16, 32, or 64"

        self.inner_attn = FlashAttention(
            attention_dropout=attention_dropout, **factory_kwargs
        )
        self.rearrange = rearrange

    def forward(self, qkv, attn_mask=None, key_padding_mask=None, need_weights=False):
        qkv = self.rearrange(
            qkv, "b (three h d) s -> b s three h d", three=3, h=self.num_heads
        )
        qkv, _ = self.inner_attn(
            qkv,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            causal=self.causal,
        )
        return self.rearrange(qkv, "b s h d -> b (h d) s")

class GAttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        g_equiv=False,
        g_input=None,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        attention_type="reg", # Default value is "flash" all other values default to regular
        encoder_channels=None,
        dims=2,
        channels_last=False,
        use_new_attention_order=False,
    ):
        super().__init__()

        self.g_equiv = g_equiv or False
        self.g_input = g_input or None

        # Detect input group and scale number of channels based on corresponding scaling factor
        # See 'equivariant_layers.py' for the origin of these values.
        nti = 1
        if self.g_equiv:
            if self.g_input == 'Z2':
                nti = 1
            elif self.g_input == 'H' or self.g_input == 'V':
                nti = 2
            elif self.g_input == 'C4':
                nti = 4
            elif self.g_input == 'D4':
                nti = 8
            else:
                raise ValueError(f"unsupported g_input in GAttentionBlcok(): {g_input}")

        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = gconv_nd(dims, g_equiv, g_input, g_input, channels, 3*channels, 1)
        self.attention_type = attention_type
        if attention_type == "flash":
            self.attention = QKVFlashAttention(channels, self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.use_attention_checkpoint = not (
            self.use_checkpoint or self.attention_type == "flash"
        )
        if encoder_channels is not None:
            assert attention_type != "flash"
            self.encoder_kv = gconv_nd(1, g_equiv=False, in_channels=encoder_channels, out_channels=2*channels, kernel_size=1) # TODO - Implement 1D group equivariant convolution 
        self.proj_out = zero_module(gconv_nd(dims, g_equiv=self.g_equiv, g_input=self.g_input, g_output=self.g_input, in_channels=channels, out_channels=channels, kernel_size=1))

    def forward(self, x, encoder_out=None):
        if encoder_out is None:
            return checkpoint(
                self._forward, (x,), self.parameters(), self.use_checkpoint
            )
        else:
            return checkpoint(
                self._forward, (x, encoder_out), self.parameters(), self.use_checkpoint
            )

    def _forward(self, x, encoder_out=None):
        b, _, *spatial = x.shape
        qkv = self.qkv(self.norm(x)).view(b, -1, np.prod(spatial))
        if encoder_out is not None:
            encoder_out = self.encoder_kv(encoder_out)
            h = checkpoint(
                self.attention, (qkv, encoder_out), (), self.use_attention_checkpoint
            )
        else:
            h = checkpoint(self.attention, (qkv,), (), self.use_attention_checkpoint)
        h = h.view(b, -1, *spatial)
        h = self.proj_out(h)
        return x + h
