import torch
import torch.nn.functional as F

from respace import space_timesteps


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class Schedule:
    def __init__(self, betas):
        self.timesteps = len(betas)
        # define alphas
        self.alphas = 1. - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef1 = betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)


class ScheduleDDIM(Schedule):
    def __init__(self, ddim_timesteps, base, **kwargs):
        timesteps_for_ddim = space_timesteps(base.timesteps, f"ddim{ddim_timesteps}")
        timestep_map = []
        new_betas = []
        last_alpha_cumprod = 1.0
        for i, alpha_cumprod in enumerate(base.alphas_cumprod):
            if i in timesteps_for_ddim:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                timestep_map.append(i)
        self.timestep_map = timestep_map
        self.use_timesteps = timesteps_for_ddim
        self.base = base
        kwargs["betas"] = torch.tensor(new_betas)
        super().__init__(**kwargs)

    def transform_times(self, ts, t_index=None):
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        if t_index is not None:
            return map_tensor[ts], map_tensor[t_index]
        return map_tensor[ts]


def save_schedule_kwargs(path, sched_kwargs):
    torch.save({
        "sched_kwargs": sched_kwargs,
    }, path)

def load_schedule_kwargs(path):
    loaded = torch.load(path, weights_only=True)
    return loaded["sched_kwargs"]
