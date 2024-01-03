import torch
import os

## This method also exists in models/nn.py
# will fix this duplication issue afterwards
def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]

def checkpoint_model(model, ckpt_dir = 'checkpoint/tmp.pth'):
    p, filename = os.path.split(ckpt_dir)
    if not os.path.exists(p):
        os.makedirs(p)
    torch.save(model.state_dict(), ckpt_dir)
    print(f'{ckpt_dir} saved')

## return arccos(cos(x))
def x2v(x):
    x_ = x % (2 * torch.pi)
    return torch.pi - torch.abs(x_ - torch.pi)

def x2v_c(x):
    x = x2v(x)
    # centering the input from [0, pi] to [-1, 1]
    x = (2 * x / torch.pi) - 1
    return x

def x2v_sin(x):
    return 1 - torch.abs(2 - (x+1)%4)

def get_time_step(t_idx, T = 20, rho = 7, N = 1000, eps = 0.001):
    # t schedule follows the one mentioned in the consistency model  
    return ( eps**(1/rho) + (t_idx - 1) / (N-1) * ( T**(1/rho) - eps**(1/rho) ) )**rho

def dir_switcher(x):
    # returns a mask that has the same dim as x
    # mask's elem is True if the corresponding elem in x
    # need to be flipped
    return ( x % (2 * torch.pi) ) > torch.pi


def dir_switcher_sin(x):
    # returns a mask that has the same dim as x
    # mask's elem is True if the corresponding elem in x
    # need to be flipped
    return ( (x + 1) % 4 ) > 2


def get_weightings(weight_schedule, snrs, sigma_data):
    if weight_schedule == "snr":
        weightings = snrs
    elif weight_schedule == "snr+1":
        weightings = snrs + 1
    elif weight_schedule == "karras":
        weightings = snrs + 1.0 / sigma_data**2
    elif weight_schedule == "truncated-snr":
        weightings = torch.clamp(snrs, min=1.0)
    elif weight_schedule == "uniform":
        weightings = torch.ones_like(snrs)
    else:
        raise NotImplementedError()
    return weightings


class Bit2Real:
    def __init__(self, num_bits, use_fp16, device):
        if use_fp16:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        self.num_bits = num_bits
        M = torch.tensor([2 ** i for i in range(num_bits-1, -1, -1)], dtype=self.dtype, device = device).repeat(num_bits, 1).T
        self.M = torch.triu(M, diagonal=1).to(self.dtype)
        maximum = M.sum(0)
        maximum[0] = 1
        self.maximum = maximum[None, :, None, None, None]


    def __call__(self, x):
        # assert torch.is_floating_point(x) is False
        assert x.size(1) == self.num_bits

        x_int = torch.einsum('bncwh,nj->bjcwh', x, self.M)

        # scale integers from 0 .. (2 ** num_bits - 1) to [-1, 1]
        x_real = x_int * 2 / self.maximum  - 1

        # zero the left most bit
        x_real[:, 0] = 0

        return x_real
    

class Bit2Int:
    def __init__(self, num_bits, device):
        self.num_bits = num_bits
        M = torch.tensor([2 ** i for i in range(num_bits-1, -1, -1)], dtype=self.dtype, device = device).repeat(num_bits, 1).T
        self.M = torch.triu(M, diagonal=1).to(self.dtype)
        maximum = M.sum(0)
        maximum[0] = 1
        self.maximum = maximum[None, :, None, None, None]

    def __call__(self, x):
        # assert torch.is_floating_point(x) is False
        assert x.size(1) == self.num_bits

        x_int = torch.einsum('bncwh,nj->bjcwh', x, self.M)

        # scale integers from 0 .. (2 ** num_bits - 1) to [-1, 1]
        x_real = x_int * 2 / self.maximum  - 1

        # zero the left most bit
        x_real[:, 0] = 0

        raise NotImplementedError
        return x_real
    

class Denoiser:
    def __init__(self) -> None:
        # self.sigma_data = sigma_data
        pass

    def denoise(self, model, x_t, sigmas, **model_kwargs):
        rescaled_t = 1000 * 0.25 * torch.log(sigmas + 1e-44)
        denoised = model(x_t, rescaled_t, **model_kwargs)
        return denoised
    
    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in


class LogNormalSampler:
    def __init__(self, p_mean=-1.2, p_std=1.3, even=False):
        self.p_mean = p_mean
        self.p_std = p_std
        self.even = even
        if self.even:
            self.inv_cdf = lambda x: norm.ppf(x, loc=p_mean, scale=p_std)
            self.rank, self.size = dist.get_rank(), dist.get_world_size()

    def sample(self, bs, device):
        if self.even:
            # buckets = [1/G]
            start_i, end_i = self.rank * bs, (self.rank + 1) * bs
            global_batch_size = self.size * bs
            locs = (torch.arange(start_i, end_i) + torch.rand(bs)) / global_batch_size
            log_sigmas = torch.tensor(self.inv_cdf(locs), dtype=torch.float32, device=device)
        else:
            log_sigmas = self.p_mean + self.p_std * torch.randn(bs, device=device)
        sigmas = torch.exp(log_sigmas)
        weights = torch.ones_like(sigmas)
        return sigmas, weights