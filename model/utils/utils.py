import torch
import numpy as np


# convert ints of base 10 to the ones of base 2
def int_to_bits(x, bits=None, dtype=torch.uint8):
    assert not(x.is_floating_point() or x.is_complex()), "x isn't an integer type"
    print( x.element_size())
    if bits is None: bits = x.element_size() * 8
    mask = 2**torch.arange(bits-1,-1,-1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).to(dtype=dtype)


def int_to_bits_np(x, bits=None, dtype=np.uint8):
    # assert not(x.is_floating_point() or x.is_complex()), "x isn't an integer type"
    mask = 2**np.arange(bits-1,-1,-1)
    return np.not_equal(np.bitwise_and(np.expand_dims(x, axis = [-1]), mask), 0).astype(dtype)


# convert ints of base 2 to the ones of base 10
def bits_to_int(x, idx_D = 1, dtype=torch.uint8):
    # get number of (D)igits (bits)
    D = x.size(idx_D)

    # create an array of weights
    shape_ = (1,) * (idx_D - 1) + (D,) + (1,) * (len(x.shape) - idx_D - 1)
    W = (2 ** torch.arange(D-1, -1, -1)).view(*shape_)

    # convert
    return torch.sum(x * W, dim = idx_D)


def bits_to_int_np(x, idx_D = 1, dtype=np.uint8):
    # get number of (D)igits (bits)
    D = x.shape[idx_D]

    # create an array of weights
    shape_ = (1,) * (idx_D - 1) + (D,) + (1,) * (len(x.shape) - idx_D - 1)
    W = (2 ** np.arange(D-1, -1, -1)).reshape(*shape_)

    # convert
    return np.sum(x * W, axis = idx_D)


if __name__ == '__main__':

    a = torch.tensor([[
        [191.,  95., 255.],
        [ 95., 111., 255.],
        [144., 144.,  96.]],
        [[ 63., 159., 191.],
        [ 63.,  95., 255.],
        [112.,   8.,  16.]],
        [[ 63.,  95., 127.],
        [ 95., 239., 127.],
        [144.,  80., 160.]]])
    a = a.to(torch.uint8)
    
    bits = int_to_bits(a, dtype=torch.float32)

    bits_np = int_to_bits_np(a.numpy(), bits = 8)

    bits_np = bits_np.transpose(0,3,1,2)

    bits_np = bits_to_int_np(bits_np, idx_D = 1)

    print(bits_np - a.numpy())

    # bits = bits.permute(0,3,1,2)
    # b = bits_to_int(bits, idx_D = 1)
    
    # print(a-b)