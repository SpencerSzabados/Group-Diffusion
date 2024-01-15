"""
    Author: Spencer Szabados
    Date: 2024-01-01

    This file contains scripts for computing the negative log-likelihood of
    image samples. 

    This implementaion is bsaed on that from:
    (https://github.com/yang-song/score_sde_pytorch/blob/main/likelihood.py)
"""


import torch as th
import numpy as np
from scipy import integrate
from . import sde_lib

from piq import LPIPS


class Karras_Score:
    """
        Class wrapper for computing the score of a karras edm diffusion model
        to enable the computation of the log-likelihood of images.
    """
    def __init__(
        self,
        model=None,
        model_kwargs=None,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        num_timesteps=100,
        rho=7.0,
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        weight_schedule="karras",
        device='cpu'
    ):
        self.model = model
        assert(model != None)
        self.model.eval()
        self.device = 'cuda:0' # TODO: make this a command line argument and distribute

        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.num_timesteps = num_timesteps
        self.rho = rho
        self.s_churn = s_churn
        self.s_tmax = s_tmax
        self.s_tmin = s_tmin
        self.s_noise = s_noise
        self.min_inv_rho = self.sigma_min ** (1 / self.rho)
        self.max_inv_rho = self.sigma_max ** (1 / self.rho)

        # self.sigmas = self.get_sigmas_karras(self.num_timesteps)
        self.sigmas = th.exp(th.linspace(np.log(self.sigma_min), np.log(self.sigma_max), self.num_timesteps)).to(self.device)

    def append_dims(self, x, target_dims):
        """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
        dims_to_append = target_dims - x.ndim
        if dims_to_append < 0:
            raise ValueError(
                f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
            )
        return x[(...,)+(None,)*dims_to_append]
    
    def append_zero(self, x):
        return th.cat([x, x.new_zeros([1])])
    
    def get_sigmas_karras(self, n):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = th.linspace(0, 1, n)
        sigmas = (self.max_inv_rho + ramp * (self.min_inv_rho - self.max_inv_rho)) ** self.rho
        return self.append_zero(sigmas).to(self.device)
    
    def calculate_sigma_karras(self, t):
        sigma = (self.max_inv_rho + t*(self.min_inv_rho-self.max_inv_rho))**self.rho
        return sigma

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2/(sigma**2+self.sigma_data**2)
        c_out = sigma*self.sigma_data/(sigma**2+self.sigma_data**2)**0.5
        c_in = 1/(sigma**2+self.sigma_data**2)**0.5
        return c_skip, c_out, c_in

    def to_d(self, x, sigma, denoised):
        """Converts a denoiser output to a Karras ODE derivative."""
        # print("x shape: "+str(x.shape)) # DEBUG
        # print("sigma shape: "+str(sigma.shape))
        # print("denoised shape: "+str(denoised.shape)) 
        return (x - denoised) / self.append_dims(sigma, x.ndim)

    def denoise(self, x_t, sigmas):
        c_skip, c_out, c_in = [
            self.append_dims(x, x_t.ndim) for x in self.get_scalings(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        model_output = self.model(c_in * x_t, rescaled_t)
        denoised = c_out * model_output + c_skip * x_t
        
        return model_output, denoised

    @th.no_grad()
    def euler_step(self, x_t, t):
        """Adds noise to x using sigma[t] and then denoises and computes the gradident of score
           to perform euler update step.
        """
        s_in = x_t.new_ones([x_t.shape[0]])
        sigma_hat = self.sigmas[t]
        noise = th.randn_like(x_t)
        x_t1 = x_t + noise*(sigma_hat**2 - self.sigmas[t] ** 2) ** 0.5
        _, denoised = self.denoise(x_t1, sigma_hat*s_in)
        d = self.to_d(x_t1, sigma_hat*s_in, denoised)
        
        dt = self.sigmas[t-1]-sigma_hat
        x_t1 = x_t1 + d*dt

        return x_t1, dt

    def denoise_score(self, x_t, t):
        s_in = x_t.new_ones([x_t.shape[0]])
        sigma_hat = self.sigmas[t]
        noise = th.randn_like(x_t)
        x_t1 = x_t + noise*(sigma_hat**2 - self.sigmas[t] ** 2) ** 0.5
        _, denoised = self.denoise(x_t1, sigma_hat*s_in)
        d = self.to_d(x_t1, sigma_hat*s_in, denoised)
       
        return d

    def get_score(self, x, label):
    #    print("get_score label: "+str(label)) # DEBUG
       return self.denoise_score(x, label[0])
    

def get_model_fn(model, train=False):
  """Create a function to give the output of the score-based model.

  Args:
    model: The score model.
    train: `True` for training and `False` for evaluation.

  Returns:
    A model function which returns the score of the model at a given datapoint.
  """

  def model_fn(x, labels):
    """Compute the output of the score-based model.

    Args:
      x: A batch of input data.
      labels: A batch of conditioning variables for time steps. Should be interpreted differently
        for different models.

    Returns:
      A tuple of (model output, new mutable states)
    """
    if not train:
        karras_score = Karras_Score(model=model)
        print("Karras score norm: "+str(th.norm(karras_score.get_score(x, labels))), flush=True) # DEBUG
        return karras_score.get_score(x, labels)
    else:
        raise NotImplementedError(f"NLL computation is only implemented for inference not during training.")

  return model_fn


def get_score_fn(sde, model, train=False, continuous=False):
    """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

    Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A score model.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.

    Returns:
    A score function.
    """
    model_fn = get_model_fn(model, train=train)

    def score_fn(x, t):
        if continuous: # DEBUG what is the mean of the marginal in terms of a label here? Should this not be the time?
            labels = sde.marginal_prob(th.zeros_like(x), t)[1]
        else:
            # For VE-trained models, t=0 corresponds to the highest noise level
            labels = sde.T - t
            labels *= sde.N - 1
            labels = th.round(labels).long() 
        # print("label: "+str(labels)) # DEBUG
        # print("X: "+str(x[0][0][0])) # DEBUG
        score = model_fn(x, labels)

        return score
    
    return score_fn


def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return th.from_numpy(x.reshape(shape))


def get_div_fn(fn):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

    def div_fn(x, t, eps):
        with th.enable_grad():
            x.requires_grad_(True)
            fn_eps = th.sum(fn(x, t) * eps)
            grad_fn_eps = th.autograd.grad(fn_eps, x)[0]
        x.requires_grad_(False)
        return th.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))

    # def div_fn(x, t, noise):
    #     with th.enable_grad():
    #         sigma = (80. ** (1 / 7.) + t*(0.002 ** (1 / 7.)-80. ** (1 / 7.)))**7.
    #         x_t = x + noise * sigma
    #         x_t.requires_grad_(True)
    #         fn_noise = th.sum(fn(x_t, t)*noise)
    #         grad_fn_noise = th.autograd.grad(fn_noise, x_t)[0]
    #     x_t.requires_grad_(False)
    #     return th.sum(grad_fn_noise * noise, dim=tuple(range(1, len(x_t.shape))))

    return div_fn


def get_likelihood_fn(sde, inverse_scaler, hutchinson_type='Rademacher',
                      rtol=1e-5, atol=1e-5, method='RK45', eps=1e-5):
    """Create a function to compute the unbiased log-likelihood estimate of a given data point.

    Args:
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    inverse_scaler: The inverse data normalizer.
    hutchinson_type: "Rademacher" or "Gaussian". The type of noise for Hutchinson-Skilling trace estimator.
    rtol: A `float` number. The relative tolerance level of the black-box ODE solver.
    atol: A `float` number. The absolute tolerance level of the black-box ODE solver.
    method: A `str`. The algorithm for the black-box ODE solver.
        See documentation for `scipy.integrate.solve_ivp`.
    eps: A `float` number. The probability flow ODE is integrated to `eps` for numerical stability.

    Returns:
    A function that takes a batch of data points and returns the log-likelihoods in bits/dim,
        the latent code, and the number of function evaluations cost by computation.
    """

    def drift_fn(model, x, t):
        """The drift function of the reverse-time SDE."""
        score_fn = get_score_fn(sde, model, train=False, continuous=False) # Default: continuous=True
        # Probability flow ODE is a special case of Reverse SDE
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t)[0]


    def div_fn(model, x, t, noise):
        return get_div_fn(lambda xx, tt: drift_fn(model, xx, tt))(x, t, noise)


    def likelihood_fn(model, data):
        """Compute an unbiased estimate to the log-likelihood in bits/dim.

        Args:
            model: A score model.
            data: A PyTorch tensor.

        Returns:
            bpd: A PyTorch tensor of shape [batch size]. The log-likelihoods on `data` in bits/dim.
            z: A PyTorch tensor of the same shape as `data`. The latent representation of `data` under the
            probability flow ODE.
            nfe: An integer. The number of function evaluations used for running the black-box ODE solver.
        """
        with th.no_grad():
            shape = data.shape
            if hutchinson_type == 'Gaussian':
                epsilon = th.randn_like(data)
            elif hutchinson_type == 'Rademacher':
                epsilon = th.randint_like(data, low=0, high=2).float() * 2 - 1.
            else:
                raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")
            
            def ode_func(t, x):
                print("ode_func t: "+str(t)) # DEBUG
                sample = from_flattened_numpy(x[:-shape[0]], shape).to(data.device).type(th.float32)
                vec_t = th.ones(sample.shape[0], device=sample.device) * t
                drift = to_flattened_numpy(drift_fn(model, sample, vec_t))
                logp_grad = to_flattened_numpy(div_fn(model, sample, vec_t, epsilon))
                return np.concatenate([drift, logp_grad], axis=0)

            init = np.concatenate([to_flattened_numpy(data), np.zeros((shape[0],))], axis=0)
            solution = integrate.solve_ivp(ode_func, (eps, sde.T-eps), init, rtol=rtol, atol=atol, method=method)
            nfe = solution.nfev
            zp = solution.y[:, -1]
            z = from_flattened_numpy(zp[:-shape[0]], shape).to(data.device).type(th.float32)
            delta_logp = from_flattened_numpy(zp[-shape[0]:], (shape[0],)).to(data.device).type(th.float32)
            prior_logp = sde.prior_logp(z)
            bpd = -(prior_logp + delta_logp) / np.log(2)
            N = np.prod(shape[1:])
            bpd = bpd / N
            # A hack to convert log-likelihoods to bits/dim
            offset = 7. - inverse_scaler(-1.)
            bpd = bpd + offset
            return bpd, z, nfe

    return likelihood_fn