"""
    Author: Spencer Szabados
    Date: 2024-01-01

    This file contains scripts for computing the negative log-likelihood of
    image samples. 

    This implementaion is bsaed on that from:
    (https://github.com/yang-song/score_sde_pytorch/blob/main/likelihood.py)
"""

import abc
import torch as th
import numpy as np
from tqdm import tqdm
from piq import LPIPS
import torchvision

class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, N):
        """Construct an SDE.

        Args:
        N: number of discretization time steps.
        """
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, t):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        """Generate one sample from the prior distribution, $p_T(x)$."""
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
        z: latent code
        Returns:
        log probability density
        """
        pass

    def discretize(self, x, t):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
        x: a torch tensor
        t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
        f, G
        """
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * th.sqrt(th.tensor(dt, device=t.device))
        return f, G

    def reverse(self, score_fn, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
        score_fn: A time-dependent score-based model that takes x and t and returns the score.
        probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = self.N
        T = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize

        # Build the class for reverse-time SDE.
        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                drift, diffusion = sde_fn(x, t)
                score = score_fn(x, t)
                drift = drift - diffusion[:, None, None, None]**2 * score * (0.5 if self.probability_flow else 1.)
                # Set the diffusion function to zero for ODEs.
                diffusion = 0. if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x, y, t):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, t)
                rev_f = f - G**2 * score_fn(x, y, t)*(0.5 if self.probability_flow else 1.)
                rev_G = th.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G
        
        return RSDE()
    

class VPSDE(SDE):
    def __init__(self, sigma_min=0.1, sigma_max=20, N=1000):
        """Construct a Variance Preserving SDE.

        Default values are configured for DDIM sde paramters.

        Args:
            beta_min: value of beta(0)
            beta_max: value of beta(1)
            N: number of discretization steps
        """
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = self.get_sigmas_ddim(N)
        self.N = N

    def get_sigmas_ddim(self, steps, ns=0.0002, ds=0.00025):
        dt = 1 / steps
        times = th.linspace(1., 0., steps+1)
        times = th.stack((times[:-1], (times[1:] - dt).clamp_(min=0)), dim = 0)
        times = times.unbind(dim = -1)
        return times
    
    def gamma(self, t, ns=0.0002, ds=0.00025):
        """Given time t [0,1] compute DDIM sigma time parameterization values."""
        return  (th.cos((t+ns)/(1+ds)*th.pi/2))**2
    
    def f(self, t, ns=0.0002, ds=0.00025):
        """Returns value of dlog(gamma(t))/dt"""
        return -(th.pi*th.sin((t+ns)/(1+ds)*th.pi/2)) / (2*(1+ds)*th.cos((t+ns)/(1+ds)*th.pi/2))

    @property
    def T(self):
        return 1
    
    @property
    def type(self):
        return 'VPSDE'

    def sde(self, x, t):
        beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = th.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        mean = th.exp(log_mean_coeff[:, None, None, None]) * x
        std = th.sqrt(1. - th.exp(2. * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape):
        return th.randn(*shape)

    def prior_logp(self, z):
        shape = z.shape
        d = np.prod(shape[1:])
        # logp = -d/2.*np.log(2*np.pi) - th.sum(z**2, dim=(1, 2, 3))/2.
        gamma = self.gamma(self.discrete_sigmas[-1][0])
        logp = d/2.*np.log(2.*np.pi*gamma**2) - th.sum(z**2, dim=(1, 2, 3))/(2*gamma**2) 
        return logp

    def discretize(self, x, t):
        """DDIM discretization."""
        s_in = x.new_ones([x.shape[0]])

        times = self.discrete_sigmas[t]

        f = 0.5*th.sqrt(self.f(times[0]))*x
        G = th.sqrt(-self.f(times[0]))
        return f, G


class VESDE(SDE):
    def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
        """Construct a Variance Exploding SDE.

        Args:
        sigma_min: smallest sigma.
        sigma_max: largest sigma.
        N: number of discretization steps
        """
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = th.exp(th.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
        self.N = N

    @property
    def T(self):
        return 1
    
    @property
    def type(self):
        return 'VESDE'

    def sde(self, x, t):
        sigma = self.sigma_min*(self.sigma_max/self.sigma_min)**t
        drift = th.zeros_like(x)
        diffusion = sigma * th.sqrt(th.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                    device=t.device))
        return drift, diffusion

    def marginal_prob(self, x, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t # Standard deviation at time t 
        mean = x
        return mean, std

    def prior_sampling(self, shape):
        return th.randn(*shape) * self.sigma_max

    def prior_logp(self, z):
        shape = z.shape
        d = np.prod(shape[1:])
        logp = -d/2.*np.log(2.*np.pi*self.sigma_max**2) - th.sum(z**2, dim=(1, 2, 3))/(2*self.sigma_max**2) 
        return logp

    def discretize(self, x, t):
        """SMLD(NCSN) discretization."""
        sigma = self.discrete_sigmas.to(x.device)[t]
        # adjacent_sigma = th.where(t==0, th.zeros_like(t), self.discrete_sigmas.to(x.device)[t-1])
        adjacent_sigma = self.discrete_sigmas.to(x.device)[t-1] if t-1 >= 0 else 0.0
        f = th.zeros_like(x)
        # G = th.sqrt(sigma**2 - adjacent_sigma**2)
        G =  self.discrete_sigmas[t] - (0 if t-1 < 0 else self.discrete_sigmas[t-1])
        return f, G


class Karras_Score:
    """
        Class wrapper for computing the score of a karras edm diffusion model
        to enable the computation of the log-likelihood of images.
    """
    def __init__(
        self,
        model=None,
        model_kwargs=None,
        sampler='euler',
        sde=None,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        steps=1000,
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
        self.device = device

        self.sampler = sampler

        self.sde = sde 
        assert(sde != None)
        self.steps = sde.N

        self.sigma_data = sigma_data
        self.sigma_max = sde.sigma_max
        self.sigma_min = sde.sigma_min
        self.sigmas = sde.discrete_sigmas
        self.rho = rho
        self.s_churn = s_churn
        self.s_tmax = s_tmax
        self.s_tmin = s_tmin
        self.s_noise = s_noise
        self.min_inv_rho = self.sigma_min**(1/self.rho)
        self.max_inv_rho = self.sigma_max**(1/self.rho)

        self.sigmas = sde.discrete_sigmas

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

    def get_scalings(self, sigma):
        """Calculates EDM SDE/ODE C_skip,C_in,C_out paramters at time sigma."""
        c_skip = self.sigma_data**2/(sigma**2+self.sigma_data**2)
        c_out = sigma*self.sigma_data/(sigma**2+self.sigma_data**2)**0.5
        c_in = 1/(sigma**2+self.sigma_data**2)**0.5
        return c_skip, c_out, c_in

    def to_d(self, x, sigma, denoised):
        """Converts a denoiser output to a Karras ODE derivative."""
        return (x-denoised)/self.append_dims(sigma, x.ndim)

    def denoise(self, x_t, y, sigmas):
        """Denoises x_t given time steps sigmas."""
        c_skip, c_out, c_in = [
            self.append_dims(x, x_t.ndim) for x in self.get_scalings(sigmas)
        ]
        rescaled_t = 1000*0.25*th.log(sigmas + 1e-44)
        if y==None:
            model_output = self.model(c_in*x_t, rescaled_t)
        else:
            model_output = self.model(c_in*x_t, rescaled_t, y)
        denoised = c_out*model_output + c_skip*x_t  
        return model_output, denoised

    @th.no_grad()
    def euler_step(self, x, y, t):
        """Adds noise to x using sigma[t] and then denoises and computes the gradident of score
           to perform euler update step.
        """
        s_in = x.new_ones([x.shape[0]])

        if self.sde.type == 'VPSDE':
            t = 0 if t-1<0 else t
            t_now = self.sigmas[t][0]
            t_next = self.sigmas[t][1]
            gamma_now = self.sde.gamma(t_now)   
            gamma_next = self.sde.gamma(t_next)
            if t+1<len(self.sigmas):
                t_now_ = self.sigmas[t+1][0]
                t_next_ = self.sigmas[t+1][1]
                gamma_now_ = self.sde.gamma(t_now_)   
                gamma_next_ = self.sde.gamma(t_next_)
            else:
                gamma_now_ = gamma_now
                gamma_next_ = gamma_next
            
            _, denoised = self.denoise(x, y, t_now*s_in)
            denoised = denoised.clamp(-1,1)
            pred_noise = (x-th.sqrt(gamma_now)*denoised)/th.sqrt(1-gamma_now)

            normalizer = th.sqrt((1-gamma_next)/gamma_next)-th.sqrt((1-gamma_now)/gamma_now)
            d = self.to_d(x/th.sqrt(gamma_now), normalizer*s_in, denoised/th.sqrt(gamma_next))
            dt = th.sqrt((1-gamma_next_)/gamma_next_)-th.sqrt((1-gamma_now_)/gamma_now_)
            # dt = th.sqrt((1-gamma_next)/gamma_next)-th.sqrt((1-gamma_now)/gamma_now)

            print(str(t_now)+", "+str(t_next))
            print(str(gamma_now)+", "+str(gamma_next))
            print(dt)
            print(d*dt)

            x_t = th.sqrt(gamma_now_)*(x/th.sqrt(gamma_next_) + d*dt)

        else:
            _, denoised = self.denoise(x, y, self.sigmas[t]*s_in)
            denoised = denoised.clamp(-1,1)
            d = self.to_d(x, self.sigmas[t]*s_in, denoised)
            dt = self.sigmas[t] - (0 if t-1 < 0 else self.sigmas[t-1])

            x_t = x + d*dt

        return x, x_t, dt

    def pfode_score(self, x_t, y, t):
        """Computes the denoising score (gradient of logp) of x_t at time sigma[t]."""
        s_in = x_t.new_ones([x_t.shape[0]])
        _, denoised = self.denoise(x_t, y, self.sigmas[t]*s_in)
        d = self.to_d(x_t, self.sigmas[t]*s_in, denoised)
        return d
    
    def ddim_score(self, x_t, y, t):
        """Computes the denoising score of ddim given x_t in time window sigma[t]."""
        s_in = x_t.new_ones([x_t.shape[0]])
        t_now = self.sigmas[t][0]
        t_next = self.sigmas[t][1]
        gamma_now = self.sde.gamma(t_now)
        gamma_next = self.sde.gamma(t_next)

        _, denoised = self.denoise(x_t, y, t_now*s_in)

        pred_noise = (x_t-th.sqrt(gamma_now)*denoised)/th.sqrt(1-gamma_now)
        x = th.sqrt(gamma_next)*denoised+th.sqrt(1-gamma_next)*pred_noise

        normalizer = th.sqrt((1-gamma_next)/gamma_next)-th.sqrt((1-gamma_now)/gamma_now)

        d = self.to_d(x_t, t_now*s_in, denoised)
        return d

    def get_score(self, x, y, t):
        if self.sde.type == 'VPSDE':
            return self.ddim_score(x, y, t)
        else:
            return self.pfode_score(x, y, t)


def get_likelihood_fn(sde, inverse_scaler, hutchinson_type='Rademacher'):
    """Create a function to compute the unbiased log-likelihood estimate of a given data point.

    Args:
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    inverse_scaler: The inverse data normalizer.
    hutchinson_type: "Rademacher" or "Gaussian". The type of noise for Hutchinson-Skilling trace estimator.
    rtol: A `float` number. The relative tolerance level of the black-box ODE solver.
    atol: A `float` number. The absolute tolerance level of the black-box ODE solver.
    method: A `str`. The algorithm for the black-box ODE solver.
        See documentation for `scipy.integrate.solve_ivp`.

    Returns:
    A function that takes a batch of data points and returns the log-likelihoods in bits/dim,
        the latent code, and the number of function evaluations cost by computation.
    """
    def likelihood_fn(model, data, y):
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
        score_cl = Karras_Score(model=model, sde=sde)
        shape = data.shape

        if hutchinson_type == 'Gaussian':
            epsilon = th.randn_like(data)
        elif hutchinson_type == 'Rademacher':
            epsilon = th.randint_like(data, low=0, high=2).float()*2-1.
        else:
            raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

        def get_score_fn(score_cl):
            """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

            Args:
            sde: An `sde_lib.SDE` object that represents the forward SDE.
            model: A score model.
            train: `True` for training and `False` for evaluation.
            continuous: If `True`, the score-based model is expected to directly take continuous time steps.

            Returns:
            A score function.
            """

            def score_fn(x, y, t):
                score = score_cl.get_score(x, y, t)
                return score
            
            return score_fn

        def drift_fn(score_cl, x, y, t):
            """The drift function of the reverse-time SDE."""
            score_fn = get_score_fn(score_cl) # Default: continuous=True
            rsde = score_cl.sde.reverse(score_fn, probability_flow=True) # Probability flow ODE is a special case of Reverse SDE
            return rsde.discretize(x, y, t)[0]
            # return score_fn(x, y, t)

        def div_fn(score_cl, x, y, t, epsilon):
            with th.enable_grad():
                x.requires_grad_(True)
                fn_eps = th.sum(drift_fn(score_cl, x, y, t)*epsilon)
                grad_fn_eps = th.autograd.grad(fn_eps, x)[0]
            x.requires_grad_(False)
            return th.sum(grad_fn_eps*epsilon, dim=tuple(range(1, len(x.shape))))
        
        def ode_func(x,y,t):
            drift = drift_fn(score_cl, x, y, t)
            logp_grad = div_fn(score_cl, x, y, t, epsilon)
            return (drift, logp_grad)
    
        with th.no_grad():
            x_t = data+th.randn_like(data)*score_cl.sigma_min # Need to add noise to data as T='inf'
            print("x shape: "+str(x_t.shape))
            # print("noise variance: "+str(score_cl.sde.gamma(score_cl.sigmas[0][0]))
            if y!=None:
                y = y['y'].to(data.device)
            logp = th.zeros((shape[0],)).to(data.device)
    
            # Run euler steps
            for t in tqdm(range(0,int(score_cl.steps))): 
                # DEBUG
                grid_img = torchvision.utils.make_grid(x_t, nrow=min((len(y) or 1),10), normalize=True)
                torchvision.utils.save_image(grid_img, f'tmp_imgs/x_{t}_sample.pdf')

                _, x_t, dt = score_cl.euler_step(x_t, y, t)
                
                drift, logp_grad = ode_func(x_t, y, t) 
                
                # print(f"dt: {dt}") # DEBUG
                # print(drift.abs().max(), dt) # DEBUG

                # x_t = x_t + drift*dt 
                logp = logp + logp_grad*dt

                # print("norm of drift: "+str(th.norm(drift))) # DEBUG
                # print("logp: "+str(logp)) # DEBUG
          
            nfe = score_cl.steps
            z = x_t
            delta_logp = logp
            print("delta_logp: "+str(logp)) # DEBUG
            prior_logp = sde.prior_logp(z)
            print("prior_logp: "+str(prior_logp)) # DEBUG
            bpd = -(prior_logp-delta_logp)/np.log(2)
            # Normalize by resolution and number of channels 
            bpd = bpd/np.prod(shape[1:]) 
            # A hack to convert log-likelihoods to bits/dim
            offset = 7. - inverse_scaler(-1.)
            bpd = bpd + offset
            return bpd, z, nfe

    return likelihood_fn
