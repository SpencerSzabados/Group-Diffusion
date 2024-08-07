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
import torch as th
import torch.nn as nn
import torch.nn.init as init
# from torch.nn.modules.utils import ut
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import torch.nn.utils.parametrize as parametrize
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


# ---[ Symetrize operator blocks ]---------------
"""
    As discussed in "Structure Preserving GANs by Birrell et.al. (2022)"
    objectives (e.g., probability distributions on images) can be symmetrized, 
    that is, reduced to a set of equivalance classes induced by desired group symmetry
    properties. The following blocks implement operations that gauarantee this behaviour.
"""
class Vertical_Symmetric(nn.Module):
    def forward(self, X):
        _, _, h, w = X.shape
        upper_channel = h//2
        if not hasattr(self, 'upper_mask'):
            self.upper_mask = nn.Parameter(th.tensor([1.0]* upper_channel + [0.0] * (h - upper_channel), device = X.device)[None, None, :, None], requires_grad = False)

        return X * self.upper_mask + th.flip(X, dims=[-2]) * (1 - self.upper_mask)

    def right_inverse(self, A):
        return A
    

class Horizontal_Symmetric(nn.Module):
    def forward(self, X):
        _, _, h, w = X.shape
        left_channel = w//2
        if not hasattr(self, 'left_mask'):
            self.left_mask = nn.Parameter(th.tensor([1.0]* left_channel + [0.0] * (w - left_channel), device = X.device)[None, None, None, :], requires_grad = False)
        return X * self.left_mask + th.flip(X, dims=[-1]) * (1 - self.left_mask)
    
    def right_inverse(self, A):
        return A


class C4_Symmetric(nn.Module):
    def forward(self, X):
        _, _, h, w = X.shape
        assert h == w, 'the initialization assumes h == w'
        upper_channel = h//2
        if h % 2 == 0:
            
            tmp_ = th.tensor([[1]*upper_channel + [0] * ( h - upper_channel)], device = X.device)
            up_left_mask = nn.Parameter((tmp_.T @ tmp_)[None, None, :, :], requires_grad = False)
            
            X_ = X * up_left_mask
            X__ = None
            for rot_ in range(3):
                X__ = th.rot90(X_, 1, [-1, -2]) if X__ is None else  th.rot90(X__, 1, [-1, -2])
                X_ = X_ + X__
            return X_
        else:
            tmp_A = th.tensor([[1.0]*upper_channel + [0.0] * ( h - upper_channel)], device = X.device)
            tmp_B = th.tensor([[1.0]*(upper_channel + 1) + [0.0] * ( h - (upper_channel + 1))], device = X.device)
            up_left_mask = nn.Parameter((tmp_A.T @ tmp_B)[None, None, :, :], requires_grad=False)

            center_elem_mask = th.zeros(h, w, device = X.device)
            center_elem_mask[h//2, h//2] = 1.0
            center_elem_mask = nn.Parameter(center_elem_mask, requires_grad=False)

            X_ = X * center_elem_mask.to(X.device)
            X__ = None
            for rot_ in range(4):
                X__ = th.rot90(X * up_left_mask.to(X.device), 1, [-1, -2]) if X__ is None else th.rot90(X__, 1, [-1, -2])
                X_ = X_ + X__
            return X_
        
    def right_inverse(self, A):
        return A
        

class D4_Symmetric(nn.Module):
    def forward(self, X):
        # make the weights symmetric 
        X = X.triu() + X.triu(1).transpose(-1, -2)

        _, _, h, w = X.shape
        assert h == w, 'the initialization assumes h == w'
        upper_channel = h//2
        if h % 2 == 0:
            
            tmp_ = th.tensor([[1]*upper_channel + [0] * ( h - upper_channel)], dtype = X.dtype, device = X.device)
            up_left_mask = (tmp_.T @ tmp_)[None, None, :, :]
            
            X_ = X * self.up_left_mask
            X__ = None
            for rot_ in range(3):
                X__ = th.rot90(X_, 1, [-1, -2]) if X__ is None else  th.rot90(X__, 1, [-1, -2])
                X_ = X_ + X__
            return X_
        else:
            tmp_A = th.tensor([[1.0]*upper_channel + [0.0] * ( h - upper_channel)], dtype = X.dtype, device = X.device)
            tmp_B = th.tensor([[1.0]*(upper_channel + 1) + [0.0] * ( h - (upper_channel + 1))], dtype = X.dtype, device = X.device)
            up_left_mask =(tmp_A.T @ tmp_B)[None, None, :, :]

            center_elem_mask = th.zeros(h, w, dtype = X.dtype, device = X.device)
            center_elem_mask[h//2, h//2] = 1.0

            X_ = X * center_elem_mask
            X__ = None
            for rot_ in range(4):
                X__ = th.rot90(X * up_left_mask, 1, [-1, -2]) if X__ is None else th.rot90(X__, 1, [-1, -2])
                X_ = X_ + X__
            return X_
    
    def right_inverse(self, A):
        return A


### ---[ Group equivariant convolutions ]--------
"""
    Implementation of the equivariant convoltional layers is based on that in
    "Structure Preserving GANs by Birrell et.al. (2022)".

    This is based on the implemention given by Adam Bielski from:
    (https://github.com/adambielski/GrouPy/blob/master/groupy/gconv/pytorch_gconv/splitgconv2d.py),
    however, formatting was changed to minimic the style of:
    (https://github.com/basveeling/keras-gcnn/blob/master/keras_gcnn/layers/convolutional.py#L114)
"""

class SplitGConv2d(nn.Module):
    """
    Group equivariant convolution layer.
    
    :parm g_input: One of ('Z2', 'C4', 'D4'). Use 'Z2' for the first layer. Use 'C4' or 'D4' for later layers.
        The parameter value 'Z2' specifies the data being convolved is from the Z^2 plane (discrete mesh).
    :parm g_output: One of ('C4', 'D4'). What kind of transformations to use (rotations or roto-reflections).
        The value of g_input of the subsequent layer should match the value of g_output from the previous.
    :parm in_channels: The number of input channels. Based on the input group action the number of channels 
        used is equal to nti*in_channels.
    :parm out_channels: The number of output channels. Based on the output group action the number of channels
        used is equal to nto*out_channels.
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
        
        super(SplitGConv2d, self).__init__()

        # Transform kernel size argument 
        self.ksize = kernel_size
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)

        self.g_input = g_input
        self.g_output = g_output

        # Convert g_input, g_output to integer keys
        # sets values for nit, nto paramters
        self.nti, self.nto, self.inds = self.make_filter_indices()

        self.in_channels = in_channels//self.nti
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        
        # Construct convolution kernel weights 
        self.weight = Parameter(pt.Tensor(out_channels, self.in_channels, self.nti, *kernel_size))
        if bias:
            self.bias = Parameter(pt.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # Initialize convolution kernel weights
        init.xavier_normal_(self.weight)

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def make_filter_indices(self):
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
    
    def transform_filter_2d_nncchw(self, w, inds):
        """
        Transform filter output to be of the form [ksize, ksize, out_channels, nto, input_shape[1], input_shape[0]]
        """
        inds_reshape = inds.reshape((-1, inds.shape[-1])).astype(np.int32)
        w_indexed = w[:, :, inds_reshape[:, 0].tolist(), inds_reshape[:, 1].tolist(), inds_reshape[:, 2].tolist()]
        w_indexed = w_indexed.view(w_indexed.size()[0], 
                                   w_indexed.size()[1],
                                   inds.shape[0], 
                                   inds.shape[1], 
                                   inds.shape[2], 
                                   inds.shape[3])
        w_transformed = w_indexed.permute(0, 1, 3, 2, 4, 5) # Previously w_transformed = w_indexed.permute(0, 2, 1, 3, 4, 5)
        return w_transformed.contiguous()                   #                                             (0, 1, 3, 2, 4, 5)
    
    def transform_filter_2d_nnchw(self, y, shape):
        """
        Transform filter output to be of the form [ksize, ksize, out_channels*nto, input_shape[1], input_shape[0]]
        """
        return y.view(shape[0], shape[1]*shape[2], shape[3], shape[4])

    def forward(self, input):
        tw = self.transform_filter_2d_nncchw(self.weight, self.inds)
        tw_shape = (self.out_channels*self.nto,
                    self.in_channels*self.nti,
                    self.ksize, self.ksize)
        tw = tw.view(tw_shape)

        input_shape = input.shape
        input = input.reshape(input_shape[0], self.in_channels*self.nti, input_shape[-2], input_shape[-1])

        y = F.conv2d(input, weight=tw, bias=None, stride=self.stride,
                        padding=self.padding)
        
        batch_size, _, ny_out, nx_out = y.size()
        y = y.view(batch_size, self.out_channels, self.nto, ny_out, nx_out)

        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1, 1) # Applies bias to out_channels and not out_channels*nto
            y = y + bias

        # TODO - remove this line and add GAvgPool2D layers at each occurance of gconv
        y_pooled = th.zeros(size=(batch_size,self.out_channels,ny_out,ny_out)).to(input.device)
        for i in range(self.nto):
            y_pooled += y[:,:,i,:,:]
        # y = self.transform_filter_2d_nchw(y, [batch_size, self.out_channels, self.nto, ny_out, nx_out])
        y = y_pooled

        return y


def gconv2d(g_input, g_output, *args, **kwargs):
    """
    Wrapper function for creating group equivariant layers.
    """
    return SplitGConv2d(g_input, g_output, *args, **kwargs)


class KernelGConv2d(nn.Module):
    """
    Group equivariant convolution layer. Implemetation is based on manipulating the convolutional kernel.
    
    :parm g_input: One of ('Z2', 'C4', 'D4'). Use 'Z2' for the first layer. Use 'C4' or 'D4' for later layers.
        The parameter value 'Z2' specifies the data being convolved is from the Z^2 plane (discrete mesh).
    :parm g_output: One of ('C4', 'D4'). What kind of transformations to use (rotations or roto-reflections).
        The value of g_input of the subsequent layer should match the value of g_output from the previous.
    :parm in_channels: The number of input channels. 
    :parm out_channels: The number of output channels.
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
        super(KernelGConv2d, self).__init__()

        self.g_output = g_output
        self.g_input = g_input
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        if self.g_output == 'H' or self.g_output == 'V':
            self.num_group = 2
        elif self.g_output == 'C4':
            self.num_group = 4
        elif self.g_output == "D4":
            self.num_group = 8
        else:
            raise NotImplementedError

        self.conv_weight = nn.Parameter(th.randn(out_channels, in_channels, kernel_size, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(th.randn(out_channels)[None, :,  None, None])

        # init.xavier_normal_(self.conv_weight)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in _pair(self.kernel_size):
            n *= k
        stdv = 1. / math.sqrt(n)
        self.conv_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        out = self.bias
        # print(self.g_output)
        # exit()

        if self.g_output == 'V':
            out = out + F.conv2d(
                input = input, 
                weight = self.conv_weight, 
                bias = None, 
                stride = self.stride, 
                padding = self.padding,
                dilation = 1,
                groups = 1
            )

            out = out + F.conv2d(
                input = input, 
                weight = th.flip(self.conv_weight, dims = [-2]),
                bias = None, 
                stride = self.stride, 
                padding = self.padding,
                dilation = 1,
                groups = 1
            )
           
        elif self.g_output == 'H':
            out = out + F.conv2d(
                input = input, 
                weight = self.conv_weight, 
                bias = None, 
                stride = self.stride, 
                padding = self.padding,
                dilation = 1,
                groups = 1
            )

            out = out + F.conv2d(
                input = input, 
                weight = th.flip(self.conv_weight, dims = [-1]),
                bias = None, 
                stride = self.stride, 
                padding = self.padding,
                dilation = 1,
                groups = 1
            )
        elif self.g_output == 'C4':
            for k in range(4):
                out = out + F.conv2d(
                    input = input, 
                    weight = th.rot90(self.conv_weight, k = k, dims = [-1, -2]),
                    bias = None, 
                    stride = self.stride, 
                    padding = self.padding,
                    dilation = 1,
                    groups = 1
                )
        elif self.g_output == "D4":
            kernel = self.conv_weight
            for k in range(4):
                kernel = th.rot90(kernel, k=1, dims=[-1, -2])
                out = out + F.conv2d(
                    input = input, 
                    weight = kernel,
                    bias = None, 
                    stride = self.stride, 
                    padding = self.padding,
                    dilation = 1,
                    groups = 1
                )
            kernel = th.flip(self.conv_weight, dims=[-2])
            out += out
            for k in range(4):
                kernel = th.rot90(kernel, k=1, dims=[-1, -2])
                out = out + F.conv2d(
                    input = input, 
                    weight = kernel,
                    bias = None, 
                    stride = self.stride, 
                    padding = self.padding,
                    dilation = 1,
                    groups = 1
                )
        else:
            raise NotImplementedError
        
        return out


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
                if len(g_output.split('_')) > 1:
                    g_output, suffix = g_output.split('_')

                    if len(g_input.split('_')) > 1:
                        g_input, suffix_ = g_input.split('_')
                else:
                    suffix = None
                # If simple kernel symmetric layer  is desired
                if suffix == 'K':
                    logger.info(f'Initializing K weight tied equiv. layer: {g_input, g_output}')
                    return KernelGConv2d(g_input, g_output, *args, **kwargs)
                # If masking kernel symmetric kerel layer is desired 
                elif suffix == 'S':
                    logger.info(f'Initializing G weight tied equiv. layer: {g_input, g_output}')
                    if g_output == 'H':
                        layer = nn.Conv2d(*args, **kwargs)
                        parametrize.register_parametrization(layer, "weight", Horizontal_Symmetric())  
                        return layer 
                    elif g_output == 'V':
                        layer = nn.Conv2d(*args, **kwargs)
                        parametrize.register_parametrization(layer, "weight", Vertical_Symmetric())   
                        return layer
                    elif g_output == 'C4':
                        layer = nn.Conv2d(*args, **kwargs)
                        parametrize.register_parametrization(layer, "weight", C4_Symmetric())
                        return layer   
                    elif g_output == 'D4':
                        layer = nn.Conv2d(*args, **kwargs)
                        # layer.weight.data = layer.weight.data.half()
                        parametrize.register_parametrization(layer, "weight", D4_Symmetric())  
                        return layer
                # If GrouPy equivariant layer is desired
                elif suffix == None:
                    logger.info(f'Initializing splitgconv equiv. layer: {g_input, g_output}')
                    return SplitGConv2d(g_input, g_output, *args, **kwargs)
                else:
                    raise NotImplementedError(f"unsupported g_input g_ouput combination in gconv_nd: {g_input, g_output}\n or unsupported suffix: {suffix}")
        raise ValueError(f"unsupported dimensions for equivariant in gconv_nd: {dims}")
    elif g_equiv == False:
        if dims == 1:
            return nn.Conv1d(*args, **kwargs)
        elif dims == 2:
            return nn.Conv2d(*args, **kwargs)
        elif dims == 3:
            return nn.Conv3d(*args, **kwargs)
        raise ValueError(f"unsupported dimensions in gconv_ng: {dims}")
    else:
        raise ValueError(f"unsupported group equivariance boolean value in gconv_ng: {g_equiv}")


### ---[ Pooling layers ]------------------------
"""

"""

class GMaxPool2D(nn.Module):
    """
        Max pool over all orientations.

        TODO: Correct pooling axis
    """
    def __init__(self, g_input, **kwargs):
        super(GMaxPool2D, self).__init__(**kwargs)
        self.g_input = g_input
        self.scale = 1

    def compute_scale(self):
        if self.g_input == 'C4':
            self.scale = 4
        elif self.g_input == 'D4':
            self.scale = 8

    def _forward(self, x):
        # Rshape input tensor and scale dimention by scale
        input_shape = x.shape
        input_reshaped = x.reshape([-1,input_shape[1],input_shape[2],input_shape[3]//self.scale, self.scale])
        max_per_group = pt.max(input_reshaped, -1)

        return max_per_group

    def forward(self, x):
        return self._forward(self, x)


class GAvgPool2D(nn.Module):
    """
        Average pool over all orientations.

        TODO: Correct pooling axis
    """
    def __init__(self, g_input, **kwargs):
        super(GMaxPool2D, self).__init__(**kwargs)
        self.g_input = g_input
        self.scale = 1

    def compute_scale(self):
        if self.g_input == 'C4':
            self.scale = 4
        elif self.g_input == 'D4':
            self.scale = 8

    def _forward(self, x):
        # Rshape input tensor and scale dimention by scale
        input_shape = x.shape
        input_reshaped = x.reshape([-1,input_shape[1],input_shape[2],input_shape[3]//self.scale, self.scale])
        mean_per_group = pt.mean(input_reshaped, -1)

        return mean_per_group

    def forward(self, x):
        return self._forward(self, x)
    
# Note, in the reference code the authors define GlobalSumPooling2D. In pytorch the layer
#  nn.AvgPool2d supplies this functionality (https://discuss.pytorch.org/t/global-average-pooling-in-pytorch/6721).


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