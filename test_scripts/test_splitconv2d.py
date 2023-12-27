"""
    File contains various test for evaluating the functionality of the given 
    splitgconv2d.py implementations.
"""

import numpy as np

### ---[ Test pytorch implementation ]-----------
import torch as pt
from torch.nn.modules.utils import _pair
from torch.autograd import Variable
from groupy.gconv.make_gconv_indices import *
from groupy.gconv.pytorch_gconv.splitgconv2d import gconv2d
from groupy.gfunc import Z2FuncArray, P4FuncArray
import groupy.garray.C4_array as c4a
from PIL import Image

def test_p4_net_equivariance():
    im = np.random.randn(1, 1, 11, 11)
    inds = make_c4_z2_indices(ksize=3)
    print("make c4_z2 indices: "+str(inds))
    inds_flat = flatten_indices(inds)
    print("inds_flat: "+str(inds_flat))

    check_equivariance(
        im=im,
        layers=[
            gconv2d(g_input='Z2', g_output='C4', in_channels=1, out_channels=1, kernel_size=3, padding=1)
        ],
        input_array=Z2FuncArray,
        output_array=P4FuncArray,
        point_group=c4a,
    )

def check_equivariance(im, layers, input_array, output_array, point_group):
    # Transform the image
    print("Input: "+str(im), flush=True)
    f = input_array(im)
    print("Network output: "+str(f), flush=True)
    g = point_group.rand()
    gf = g*f # Default g*f
    im1 = gf.v
    # Apply layers to both images
    im = Variable(pt.Tensor(im))
    im1 = Variable(pt.Tensor(im1))

    fmap = im
    fmap1 = im1
    for layer in layers:
        fmap = layer(fmap)
        fmap1 = layer(fmap1)

    # Transform the computed feature maps
    fmap1_garray = output_array(fmap1.data.numpy())
    r_fmap1_data = (g.inv() * fmap1_garray).v

    fmap_data = fmap.data.numpy()
    assert np.allclose(fmap_data, r_fmap1_data, rtol=1e-5, atol=1e-3)

def test_p4_net_pooling_equivariance():
    out_channels=1
    in_channels=1
    nti=1
    ksize=3
    kernel_size = _pair(ksize)
    print("kernel size:\n "+str(kernel_size))

    w = pt.nn.Parameter(pt.Tensor(out_channels, in_channels, nti, *kernel_size))
    print("w:\n "+str(w))

    inds = make_c4_z2_indices(ksize=ksize)
    print("make c4_z2 indices:\n "+str(inds))
    inds_flat = flatten_indices(inds)
    print("inds_flat:\n "+str(inds_flat))

    inds_reshape = inds.reshape((-1, inds.shape[-1])).astype(np.int32)
    print("inds_reshape:\n "+str(inds_reshape))
    print("inds_reshaped shape:\n "+str(inds_reshape.shape))
    w_indexed = w[:, :, inds_reshape[:, 0].tolist(), inds_reshape[:, 1].tolist(), inds_reshape[:, 2].tolist()]
    print("w_indexed:\n "+str(w_indexed))
    w_indexed = w_indexed.view(w_indexed.size()[0], 
                                w_indexed.size()[1],
                                inds.shape[0], 
                                inds.shape[1], 
                                inds.shape[2], 
                                inds.shape[3])
    print("w_indexed:\n "+str(w_indexed))
    w_transformed = w_indexed.permute(0, 2, 1, 3, 4, 5)
    print("w_tranformed:\n "+str(w_transformed))

    im = pt.randn(in_channels,out_channels,ksize,ksize)
    imT = pt.rot90(im, dims=[2,3])
    layers=[
            gconv2d(g_input='Z2', g_output='C4', in_channels=in_channels, out_channels=out_channels, kernel_size=ksize, padding=1)
        ]
    
    print("Image:\n "+str(im))
    print("Image.T:\n "+str(imT))
    
    y = im
    for layer in layers:
        y = layer(y)
        print("y: "+str(y))
    y = pt.mean(y, dim=1)

    yT = imT
    for layer in layers:
        yT = layer(yT)
        print("yT: "+str(yT))
    yT = pt.mean(yT, dim=1)

    print("y_pooled:\n "+str(y))
    print("yT_pooled:\n "+str(yT))
    difference = pt.abs(pt.rot90(y, dims=[1,2])-yT)
    error = pt.sum(difference)

    print("Error:\n "+str(error))
    print("Difference:\n "+str(difference))


### ---[ Test tensforflow implementation ]-------
# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
# import tensorflow.keras.layers as KL
# # from tensorflow.keras.models import Model
# # from keras_gcnn.layers import GConv2D, GBatchNorm

# from groupy.gconv.tensorflow_gconv.splitgconv2d import gconv2d_util, gconv2d
# from groupy.gfunc.z2func_array import Z2FuncArray
# from groupy.gfunc.p4func_array import P4FuncArray
# from groupy.gfunc.p4mfunc_array import P4MFuncArray
# import groupy.garray.C4_array as C4a
# import groupy.garray.D4_array as D4a

# def check_c4_z2_conv_equivariance():
#     out_channels=1
#     in_channels=1
#     nti=1
#     ksize=3

#     print("kernel size:\\n "+str(ksize))

#     im = np.random.randn(in_channels, ksize, ksize, out_channels)
#     imT = np.rot90(im, axes=(1,2))
    
#     print("Image: "+str(im))
#     print("Image.T: "+str(imT))

#     x, y = make_graph('Z2', 'C4', ksize)

#     print("x: "+str(x))
#     print("y: "+str(y))

#     inds = make_c4_z2_indices(ksize=3)
#     print("make c4_z2 indices:\n "+str(inds))

#     inds_util, inds_shape_util, w_shape = gconv2d_util(h_input='Z2', h_output='C4', in_channels=in_channels, out_channels=out_channels, ksize=ksize)
#     print("inds_util:\n "+str(inds_util))
#     print("inds_shape_util:\n "+str(inds_shape_util))
#     print("w_shape:\n "+str(w_shape))

#     w = tf.Variable(tf.compat.v1.truncated_normal(w_shape, stddev=1.))
    
#     # Compute
#     input = tf.compat.v1.placeholder(dtype=tf.float32, shape=[out_channels, ksize, ksize, in_channels * nti])
#     output = gconv2d(input=input, filter=w, strides=[1, 1, 1, 1], padding='SAME',
#                 gconv_indices=inds_util, gconv_shape_info=inds_shape_util)
#     init = tf.compat.v1.global_variables_initializer()
#     sess = tf.compat.v1.Session()
#     sess.run(init)
#     y = sess.run(output, feed_dict={input: im})
#     yT = sess.run(output, feed_dict={input: imT})
#     sess.close()

#     print("y:\n "+str(y))
#     print("yT:\n "+str(yT))

#     differece = np.abs(np.rot90(y, axes=(1,2))-yT)
#     print("Difference:\n "+str(differece))

#     y_pooled = KL.AveragePooling2D(y)


#     check_equivariance(im, x, y, Z2FuncArray, P4FuncArray, C4a)


# def make_graph(h_input, h_output, ksize):
#     gconv_indices, gconv_shape_info, w_shape = gconv2d_util(
#         h_input=h_input, h_output=h_output, in_channels=1, out_channels=1, ksize=3)
#     nti = gconv_shape_info[-2]
#     x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, ksize, ksize, 1 * nti])
#     w = tf.Variable(tf.compat.v1.truncated_normal(shape=w_shape, stddev=1.))
#     y = gconv2d(input=x, filter=w, strides=[1, 1, 1, 1], padding='SAME',
#                 gconv_indices=gconv_indices, gconv_shape_info=gconv_shape_info)
#     return x, y


# def check_equivariance(im, input, output, input_array, output_array, point_group):

#     # Transform the image
#     f = input_array(im.transpose((0, 3, 1, 2)))
#     g = point_group.rand()
#     gf = g * f
#     im1 = gf.v.transpose((0, 2, 3, 1))

#     # Compute
#     init = tf.global_variables_initializer()
#     sess = tf.Session()
#     sess.run(init)
#     yx = sess.run(output, feed_dict={input: im})
#     yrx = sess.run(output, feed_dict={input: im1})
#     sess.close()

#     # Transform the computed feature maps
#     fmap1_garray = output_array(yrx.transpose((0, 3, 1, 2)))
#     r_fmap1_data = (g.inv() * fmap1_garray).v.transpose((0, 2, 3, 1))

#     print (np.abs(yx - r_fmap1_data).sum())
#     assert np.allclose(yx, r_fmap1_data, rtol=1e-5, atol=1e-3)



### ---[ Main ]----------------------------------

if __name__=="__main__":
    print("\n"+"="*10+"[ PYTROCH ]"+"="*10+"\n")
    test_p4_net_pooling_equivariance()
    # print("\n"+"="*10+"[ TENSORFLOW ]"+"="*10)
    # check_c4_z2_conv_equivariance()


