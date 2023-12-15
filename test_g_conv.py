# Idea is based on: https://pytorch.org/tutorials/intermediate/parametrizations.html

import torch as th
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize


dims = 3
ch=1
# layer = gconv_nd(dims, g_equiv=True, g_input=g_input, g_output=g_output,  in_channels=ch, out_channels=ch, kernel_size=3, padding=1)



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Vertical_Symmetric(nn.Module):
    def forward(self, X):
        _, _, h, w = X.shape
        upper_channel = h//2
        if not hasattr(self, 'upper_mask'):
            self.upper_mask = nn.Parameter(th.tensor([1.0]* upper_channel + [0.0] * (h - upper_channel), device = X.device)[None, None, :, None], requires_grad = False)

        return X * self.upper_mask + th.flip(X, dims=[-2]) * (1 - self.upper_mask)
    

class Horizontal_Symmetric(nn.Module):
    def forward(self, X):
        _, _, h, w = X.shape
        left_channel = w//2
        if not hasattr(self, 'left_mask'):
            self.left_mask = nn.Parameter(th.tensor([1.0]* left_channel + [0.0] * (w - left_channel), device = X.device)[None, None, None, :], requires_grad = False)
        return X * self.left_mask + th.flip(X, dims=[-1]) * (1 - self.left_mask)

class C4_Symmetric(nn.Module):
    def forward(self, X):
        _, _, h, w = X.shape
        assert h == w, 'the initialization assumes h == w'
        upper_channel = h//2
        if h % 2 == 0:
            
            if not hasattr(self, 'up_left_mask'):
                tmp_ = th.tensor([[1.0]*upper_channel + [0.0] * ( h - upper_channel)], device = X.device)
                self.up_left_mask = nn.Parameter((tmp_.T @ tmp_)[None, None, :, :], requires_grad = False)
            
            X_ = X * self.up_left_mask
            X__ = None
            for rot_ in range(3):
                X__ = th.rot90(X_, 1, [-1, -2]) if X__ is None else  th.rot90(X__, 1, [-1, -2])
                X_ = X_ + X__
            return X_
        else:
            if not hasattr(self, 'up_left_mask'):
                tmp_A = th.tensor([[1.0]*upper_channel + [0.0] * ( h - upper_channel)], device = X.device)
                tmp_B = th.tensor([[1.0]*(upper_channel + 1) + [0.0] * ( h - (upper_channel + 1))], device = X.device)
                self.up_left_mask = nn.Parameter((tmp_A.T @ tmp_B)[None, None, :, :], requires_grad=False)

            if not hasattr(self, 'center_elem_mask'):
                center_elem_mask = th.zeros(h, w, device = X.device)
                center_elem_mask[h//2, h//2] = 1.0
                self.center_elem_mask = nn.Parameter(center_elem_mask, requires_grad=False )

            X_ = X * self.center_elem_mask.to(X.device)
            X__ = None
            for rot_ in range(4):
                X__ = th.rot90(X * self.up_left_mask.to(X.device), 1, [-1, -2]) if X__ is None else th.rot90(X__, 1, [-1, -2])
                X_ = X_ + X__
            return X_
        



# ## test layer_h
# print('='*10)
# print('test layer_h')

# layer_h = th.nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=4, padding=1)

# with th.no_grad():
#     layer_h.weight *= 10000

# parametrize.register_parametrization(layer_h, "weight", Horizontal_Symmetric())    

# x = th.randn(1, ch, 4, 4)
# print( th.abs(layer_h(th.flip(x, dims = [-1])) - th.flip(layer_h(x), dims = [-1])) )
# print( th.abs(layer_h(th.flip(x, dims = [-1])) - th.flip(layer_h(x), dims = [-1])) < 1e-6)

# ## test layer_v
# print('='*10)
# print('test layer_v')
# layer_v = th.nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=4, padding=1)
# parametrize.register_parametrization(layer_v, "weight", Vertical_Symmetric())    
# x = th.randn(1, ch, 4, 4)
# print(th.abs(layer_v(th.flip(x, dims = [-2])) - th.flip(layer_v(x), dims = [-2])) < 1e-6)


# ## test layer_c4

# ## even height and width
# print('='*10)
# print('test layer_c4 (even)')

# layer_c4 = th.nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=4, padding=1)
# parametrize.register_parametrization(layer_c4, "weight", C4_Symmetric())    

# org_output = layer_c4(x)
# for rot_times in range(5):
#     print(f"rot_times: {rot_times}")
#     print(th.abs( layer_c4(th.rot90(x, rot_times, dims = [-1, -2])) - th.rot90(org_output, rot_times, dims = [-1, -2])) < 1e-6)


## odd height and width
print('='*10)
print('test layer_c4 (odd)')

x = th.randn(1, ch, 4, 4)

layer_c4 = th.nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, padding=1)
parametrize.register_parametrization(layer_c4, "weight", C4_Symmetric())    

org_output = layer_c4(x)
for rot_times in range(5):
    print(f"rot_times: {rot_times}")
    # print(th.abs( layer_c4(th.rot90(x, rot_times, dims = [-1, -2])) - th.rot90(org_output, rot_times, dims = [-1, -2])) < 1e-6)
    print(layer_c4(th.rot90(x, rot_times, dims = [-1, -2])))
