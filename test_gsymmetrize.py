
import torch as th
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from model.model_modules.equivariant_layers import *

# Constants
batch_size = 2
channels = 2
height = 5
width = 5

## odd height and width
print('='*10)
print('test symmetrization layer')

x = th.randn(batch_size, channels, height, width)
x_rot = th.rot90(x, dims=[2,3])
print("x:\n "+str(x))
print("x_rot:\n "+str(x_rot))

layer = GSymmetrize(g_output='C4')

y = layer(x)
y_rot = layer(x_rot)
print("y:\n "+str(y))
print("y_rot:\n "+str(y_rot))
