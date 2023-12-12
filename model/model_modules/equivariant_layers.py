"""
    File contains function definitons for group (mostly C_4) equivariant (or invariant) 
    convolution, pooling, and transformation layers.
     
    These are needed as standard convolutional and pooling layers may destory the
    invariant properties of the generator or disciminator process due to flattening.
"""

### ---[ library imports ]-----------------------
import os
import math
import numpy as np
import torch as pt
import torch.nn as nn
# from torch.nn.modules.utils import ut
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
# 
from .. import logger
#
from ..utils.nn import (
    checkpoint,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
# Libraries for group convolutional layers
from groupy.gconv.make_gconv_indices import make_c4_z2_indices, make_c4_p4_indices, make_d4_z2_indices, make_d4_p4m_indices
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling


### ---[ Transformation layers ]-----------------

class GUpsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv=False, g_equiv=False, g_input=None, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.g_input = g_input
        self.g_output = g_input
        self.dims = dims

        assert use_conv == True or use_conv == False
        assert isinstance(g_input, str)

        if use_conv:
            self.conv = gconv_nd(dims, g_equiv=g_equiv, g_input=self.g_input, g_output=self.g_output, in_channels=self.channels, out_channels=self.out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class GDownsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv=False, g_equiv=False, g_input=None, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.g_equiv = g_equiv
        self.g_input = g_input
        self.g_output = g_input
        self.dims = dims

        assert use_conv == True or use_conv == False
        assert isinstance(g_input, str)

        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            #print("g_input, g_output, in_channels, out_channels: "+str(g_input)+", "+str(g_output)+", "+str(out_channels), flush=True)
            self.op = gconv_nd(dims=self.dims, g_equiv=self.g_equiv, g_input=self.g_input, g_output=self.g_output, in_channels=self.channels, out_channels=self.out_channels, kernel_size=3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels, f' {x.shape[1]} || {self.channels}'
        return self.op(x)

# ---[ Equivariant transformations ]-------------
"""
    These layers implement group equivariant operations under the Special Euclidean group
    SE(2), which in general SE(n) is homeomorphic to R^n X SO(n) where SO(n) is the 
    Special Orthogonal group. The cyclic groups C_n are subgroups of SO(2). 

    The FiLM layers are extentions of those provided in (https://github.com/caffeinism/FiLM-pytorch)
"""


### ---[ Group equivariant convolutions ]--------
"""
    Implementation of the equivariant convoltional layers is based on that in
    "Structure Preserving GANs by Birrell et.al. (2022)".
"""

class SplitGConv2D(nn.Module):
    """
    This class is based on the implemention given by Adam Bielski from:
    (https://github.com/adambielski/GrouPy/blob/master/groupy/gconv/pytorch_gconv/splitgconv2d.py),
    however, formatting was changed to minimic the style of:
    (https://github.com/basveeling/keras-gcnn/blob/master/keras_gcnn/layers/convolutional.py#L114)
    """

    def __init__(self, 
                g_input, 
                g_output, 
                in_channels, 
                out_channels, 
                kernel_size=3, 
                stride=1,
                padding=0, 
                bias=True) -> None:
        
        super(SplitGConv2D, self).__init__()

        # Tranform kernel size argument 
        self.ksize = kernel_size
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)

        self.g_input = g_input
        self.g_output = g_output

        # Convert g_input, g_output to integer keys
        # sets values for nit, nto paramters
        self.nti, self.nto, self.inds = self.make_indices()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        
        self.weight = Parameter(pt.Tensor(
            out_channels, in_channels, self.nti, *kernel_size))
        if bias:
            self.bias = Parameter(pt.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def make_indices(self):
        indices = None
        nti = 1
        nto = 1
        if self.g_input == 'Z2' and self.g_output == 'C4':
            nti = 1
            nto = 4
            indices = make_c4_z2_indices(self.ksize)
        elif self.g_input == 'C4' and self.g_output == 'C4':
            nti = 4
            nto = 4
            indices = make_c4_p4_indices(self.ksize)
        elif self.g_input == 'Z2' and self.g_output == 'D4':
            nti = 1
            nto = 8
            indices = make_d4_z2_indices(self.ksize)
        elif self.g_input == 'D4' and self.g_output == 'D4':
            nti = 8
            nto = 8
            indices = make_d4_p4m_indices(self.ksize)
        else:
            raise ValueError(f"unsupported g_input g_output pair in make_indices(): {self.g_input, self.g_output}")
        return nti, nto, indices
    
    # Tranform filter output to be of the form [batch_size, nto*out_channels, input_shape[1], input_shape[0]]
    def transform_filter_2d_ncchw(self, w, inds):
        inds_reshape = inds.reshape((-1, inds.shape[-1])).astype(np.int32)
        w_indexed = w[:, :, inds_reshape[:, 0].tolist(), inds_reshape[:, 1].tolist(), inds_reshape[:, 2].tolist()]
        w_indexed = w_indexed.view(w_indexed.size()[0], 
                                   w_indexed.size()[1],
                                   inds.shape[0], 
                                   inds.shape[1], 
                                   inds.shape[2], 
                                   inds.shape[3])
        w_transformed = w_indexed.permute(0, 2, 1, 3, 4, 5)
        return w_transformed.contiguous()
    
    def transform_filter_2d_nchw(self, y, shape):
        # DEBUG
        print("tranform_filter shape: "+str(shape))
        return y.view(shape[0], shape[1]*shape[2], shape[3], shape[4])

    def forward(self, input):
        tw = self.transform_filter_2d_ncchw(self.weight, self.inds)
        tw_shape = (self.out_channels*self.nto,
                    self.in_channels,
                    self.ksize, self.ksize)
        tw = tw.view(tw_shape)

        # DEBUG
        print("gconv_input_shape: "+str(input.size()), flush=True)
        print("gconv in_channels*nti: "+str(self.in_channels*self.nti), flush=True)

        input_shape = input.size()
        input = input.view(input_shape[0], self.in_channels, input_shape[-2], input_shape[-1])

        y = F.conv2d(input, weight=tw, bias=None, stride=self.stride,
                        padding=self.padding)
        
        batch_size, _, ny_out, nx_out = y.size()
        y = y.view(batch_size, self.out_channels, self.nto, ny_out, nx_out)

        print('='*10)
        print(y.shape)

        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1, 1) # Applies bias to out_channels and not out_channels*nto
            y = y + bias

        y = pt.mean(y, dim=2)
        # y = self.transform_filter_2d_nchw(y, [batch_size, self.out_channels, self.nto, ny_out, nx_out])

        return y

class GConv2D(SplitGConv2D):
    """
    
    """
    def __init__(self, g_input, g_output, *args, **kwargs):
        super(GConv2D, self).__init__(g_input, g_output, *args, **kwargs)


def gconv_nd(dims, g_equiv=False, g_input=None, g_output=None, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D group equivariant convolution layer.
    """
    if g_equiv == True:
        if dims == 2:
            # Deal we special case 
            if g_input == 'Z2' and g_output == 'Z2':
                return nn.Conv2d(*args, **kwargs)
            else:
                return GConv2D(g_input, g_output, *args, **kwargs)
        raise ValueError(f"unsupported dimensions for equivariant in gconv_nd: {dims}")
    elif g_equiv == False:
        if dims == 1:
            return nn.Conv1d(*args, **kwargs)
        elif dims == 2:
            return nn.Conv2d(*args, **kwargs)
        elif dims == 3:
            return nn.Conv3d(*args, **kwargs)
        raise ValueError(f"unsupported dimensions in gconv_ng: {dims}")
    raise ValueError(f"unsupported group equivariance boolean value in gconv_ng: {g_equiv}")



### ---[ Pooling layers ]------------------------
"""

"""

class GMaxPool2D(nn.Module):
    """
        Max pool over all orientations.
    """
    def __init__(self, g_input, **kwargs):
        super(GMaxPool2D, self).__init__(**kwargs)
        self.g_input = g_input
        self.scale = 1

    def compute_scale(self):
        if self.g_input == 'C4':
            self.scale *= 4
        elif self.g_input == 'D4':
            self.scale *= 8

    def _forward(self, x):
        # Rshape input tensor and scale dimention by scale
        input_shape = x.shape()
        input_reshaped = x.reshape([-1,input_shape[1],input_shape[2],input_shape[3]//self.scale, self.scale])
        max_per_group = pt.max(input_reshaped, -1)

        return max_per_group

    def forward(self, x):
        return self._forward(self, x)
    
# Note, in the reference code the authors define GlobalSumPooling2D. In pytorch the layer
#  nn.AvgPool2d supplies this functionality (https://discuss.pytorch.org/t/global-average-pooling-in-pytorch/6721).


# ---[ Symetrize operator blocks ]---------------
"""
    As discussed in "Structure Preserving GANs by Birrell et.al. (2022)"
    objectives (e.g., probability distributions on images) can be symmetrized, 
    that is, reduced to a set of equivalance classes induced by desired group symmetry
    properties. The following blocks implement operations that gauarantee this behaviour.
"""

# ---[ ]

def symmetrize_block(features, num_filters, kerenl_size, h_):
    """
    TODO - Implement symmetrization block that averages over all orientations of group actions and
           returns unscaled number of channels opposed to a multiple of the input channels as done
           by GrouPy.
    """