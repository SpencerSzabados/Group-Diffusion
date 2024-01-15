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
from scipy import integrate
import torchvision

from piq import LPIPS

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
                drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
                # Set the diffusion function to zero for ODEs.
                diffusion = 0. if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x, t):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, t)
                rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
                rev_G = th.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G
        
        return RSDE()
    

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
        # self.discrete_sigmas = th.linspace(self.sigma_min, self.sigma_max, N)
        self.N = N

    @property
    def T(self):
        return 'inf'

    def sde(self, x, t):
        # t = timestep/(self.sigma_max-self.sigma_min) # TODO: verify this scaling 
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
        N = np.prod(shape[1:])
        return -N/2.*np.log(2*np.pi*self.sigma_max**2) - th.sum(z**2, dim=(1, 2, 3))/(2*self.sigma_max**2) # Default
        # return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=0) / (2 * self.sigma_max ** 2)

    def discretize(self, x, t):
        """SMLD(NCSN) discretization."""
        sigma = self.discrete_sigmas.to(x.device)[t]
        adjacent_sigma = th.where(t==0, th.zeros_like(t), self.discrete_sigmas.to(x.device)[t-1])
        f = th.zeros_like(x)
        G = th.sqrt(sigma**2 - adjacent_sigma**2)
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
        sde=None,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        num_timesteps=1000,
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

        self.sde = sde 
        assert(sde != None)
        self.num_timesteps = sde.N

        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.rho = rho
        self.s_churn = s_churn
        self.s_tmax = s_tmax
        self.s_tmin = s_tmin
        self.s_noise = s_noise
        self.min_inv_rho = self.sigma_min**(1/self.rho)
        self.max_inv_rho = self.sigma_max**(1/self.rho)

        # self.sigmas = self.get_sigmas_karras(self.num_timesteps)
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
        print(f"sigma: {sigma}")
        return (x - denoised) / self.append_dims(sigma, x.ndim)

    def denoise(self, x_t, sigmas):
        c_skip, c_out, c_in = [
            self.append_dims(x, x_t.ndim) for x in self.get_scalings(sigmas)
        ]
        rescaled_t = 1000*0.25*th.log(sigmas + 1e-44)
        model_output = self.model(c_in*x_t, rescaled_t)
        denoised = c_out*model_output + c_skip*x_t

        print("c_out, c_skip: ", c_out.abs().max(), c_skip.abs().max())        
        print("x_t.max: ", x_t.abs().max())      


        return model_output, denoised

    @th.no_grad()
    def euler_step(self, x_t, t):
        """Adds noise to x using sigma[t] and then denoises and computes the gradident of score
           to perform euler update step.
        """
        s_in = x_t.new_ones([x_t.shape[0]])
        noise = th.randn_like(x_t)
        x_t = x_t + noise*(self.sigmas[t]**2)**0.5
        _, denoised = self.denoise(x_t, self.sigmas[t]*s_in)
        denoised = denoised.clamp(-1, 1)

        print('1:', denoised.abs().max())
        d = self.to_d(x_t, self.sigmas[t]*s_in, denoised)
        
        dt = self.sigmas[t] - (0 if t - 1 < 0 else self.sigmas[t-1])


        

        print('-'* 100)
        # # print(self.sigmas)

        print(self.sigmas[t], self.sigmas[t-1], dt)

        # print(t, t-1)

        print('-'* 100)

        
        x = x_t + d*dt

        return x, x_t, dt

    def denoise_score(self, x_t, t):
        s_in = x_t.new_ones([x_t.shape[0]])
        model_output, denoised = self.denoise(x_t, self.sigmas[t]*s_in)

        print(f"denoise abs max {denoised.abs().max()} mean {denoised.abs().mean()}")
        print(f"model_output {model_output.abs().max()}")

        d = self.to_d(x_t, self.sigmas[t]*s_in, denoised)
        return d

    def get_score(self, x, t):
       # print("get_score label: "+str(label[0])) # DEBUG
       return self.denoise_score(x, t[0])


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
    eps: A `float` number. The probability flow ODE is integrated to `eps` for numerical stability.

    Returns:
    A function that takes a batch of data points and returns the log-likelihoods in bits/dim,
        the latent code, and the number of function evaluations cost by computation.
    """
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

            def score_fn(x, t):
                score = score_cl.get_score(x, t)
                return score
            
            return score_fn

        def drift_fn(score_cl, x, t):
            """The drift function of the reverse-time SDE."""
            score_fn = get_score_fn(score_cl) # Default: continuous=True
            # rsde = score_cl.sde.reverse(score_fn, probability_flow=True) # Probability flow ODE is a special case of Reverse SDE
            # return rsde.discretize(x,t)[0]
            return score_fn(x,t)

        def div_fn(score_cl, x, t, epsilon):
            fn = lambda xx, tt: drift_fn(score_cl, xx, tt)
            with th.enable_grad():
                x.requires_grad_(True)
                fn_eps = th.sum(fn(x, t)*epsilon)
                grad_fn_eps = th.autograd.grad(fn_eps, x)[0]
            x.requires_grad_(False)
            return th.sum(grad_fn_eps*epsilon, dim=tuple(range(1, len(x.shape))))
        
        def ode_func(x, t):
            print("ode_func t: "+str(t)) # DEBUG
            vec_t = th.ones(x.shape[0], device=x.device, dtype=int)*t
            drift = drift_fn(score_cl, x, vec_t)
            # print("drift: "+str(drift)) # DEBUG
            logp_grad = div_fn(score_cl, x, vec_t, epsilon)
            return (drift, logp_grad)
    
        with th.no_grad():
            grid_img = torchvision.utils.make_grid(data, nrow = 1, normalize = True)
            torchvision.utils.save_image(grid_img, f'tmp_imgs/data_input_samples.pdf')


            x_t = data+th.randn_like(data)*score_cl.sigma_min
            logp = th.zeros((shape[0],)).to(data.device)
            z = th.zeros(size=(score_cl.num_timesteps,shape[0],shape[1],shape[2],shape[3])).to(data.device)

            dist_t = 0

            # Run euler steps
            for t in range(0,int(score_cl.num_timesteps)): 
                grid_img = torchvision.utils.make_grid(x_t, nrow = 1, normalize = True)
                torchvision.utils.save_image(grid_img, f'tmp_imgs/{t}_samples.pdf')

                _, _, dt = score_cl.euler_step(data,t)

                diff = th.norm(data-x_t)
                print("data diff: "+str(diff))
                # print(dt)
                # exit()

                # dt = dt/score_cl.sigma_max # Normalize time scale 
                dist_t += dt
                print("dist_t: "+str(dist_t)) # DEBUG
                
                drift, logp_grad = ode_func(x_t,t) 
                
                print(f"dt: {dt}")

                x_t = x_t + drift*dt 

                print(drift.abs().max(), dt)

                # if t == 101:
                #     exit()

                # exit()



                z[t,:,:,:,:] = x_t
               
                logp += logp_grad*dt
                print("norm of drift: "+str(th.norm(drift))) # DEBUG
                print("logp: "+str(logp))
          
            nfe = score_cl.num_timesteps
            z = z[0]
            print("x-x_T diff: ")
            delta_logp = logp
            print("delta_logp: "+str(logp)) # DEBUG
            prior_logp = sde.prior_logp(z)
            prior_logp_z = sde.prior_logp(th.zeros_like(data))
            print(prior_logp_z)
            print("prior_logp: "+str(prior_logp)) # DEBUG
            bpd = -(prior_logp + delta_logp)/np.log(2)
            # Normalize by resolution and number of channels 
            bpd = bpd/np.prod(shape[1:]) 
            # A hack to convert log-likelihoods to bits/dim
            offset = 7. - inverse_scaler(-1.)
            bpd = bpd + offset
            return bpd, z, nfe

    return likelihood_fn
