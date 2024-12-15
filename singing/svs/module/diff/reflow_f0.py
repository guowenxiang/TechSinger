import math
import random
from functools import partial
from inspect import isfunction
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from einops import rearrange

from utils.commons.hparams import hparams
from collections import deque
from torchdyn.core import NeuralODE
sigma = 1e-4

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

class Wrapper(nn.Module):
    def __init__(self, net, cond, num_timesteps, dyn_clip):
        super(Wrapper, self).__init__()
        self.net = net
        self.cond = cond
        self.num_timesteps = num_timesteps
        self.dyn_clip = dyn_clip

    def forward(self, t, x, args):
        # print(t)
        t = torch.tensor([t * self.num_timesteps] * x.shape[0], device=t.device).long()
        # print(t.shape, x.shape)
        ut = self.net.denoise_fn(x, t, self.cond)
        if hparams['f0_sample_clip']=='clip':
            x_recon = (1 - t / self.num_timesteps) * ut + x
            if self.dyn_clip is not None:
                x_recon.clamp_(self.dyn_clip[0], self.dyn_clip[1])
            else:
                x_recon.clamp_(-1., 1.)
            ut = (x_recon - x) / (1 - t / self.num_timesteps)
        return ut

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            Mish()
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            Mish(),
            nn.Linear(time_emb_dim, dim_out)
        )

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        h += self.mlp(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)

def linear_beta_schedule(timesteps, max_beta=hparams.get('max_beta', 0.01)):
    """
    linear schedule
    """
    betas = np.linspace(1e-4, max_beta, timesteps)
    return betas

class ReflowF0(nn.Module):
    def __init__(self, out_dims, denoise_fn,
                 timesteps=1000, f0_K_step=1000, loss_type='l1', betas=None, spec_min=None, spec_max=None):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.fs2 = None
        self.mel_bins = out_dims
        self.K_step = f0_K_step
        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = linear_beta_schedule(timesteps, max_beta=hparams['f0_max_beta'])
            # betas = cosine_beta_schedule(timesteps) # 之前错误的使用了cosine scheduler！这对100步有用，但对1000不好用

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_sample(self, x_start, t, noise=None):
        if noise==None:
            noise = default(noise, lambda: torch.randn_like(x_start))
        x1 = x_start
        x0 = noise
        t_unsqueeze = t.unsqueeze(1).unsqueeze(1).unsqueeze(1).float() / self.num_timesteps
        if hparams['flow_qsample'] == 'sig':
            epsilon = torch.randn_like(x0)
            xt = t_unsqueeze * x1 + (1. - t_unsqueeze) * x0 + sigma * epsilon
        else:
            xt = t_unsqueeze * x1 + (1. - t_unsqueeze) * x0 
        return xt

    def p_losses(self, x_start, t, cond, noise=None, nonpadding=None):
        if noise==None:
            noise = default(noise, lambda: torch.randn_like(x_start))
        xt = self.q_sample(x_start=x_start, t=t, noise=noise)
        # print(x_start.shape, x_start.min(), x_start.max())
        x1 = x_start
        x0 = noise
        
        # print(x_start.shape)
        # print(xt.shape, t.shape, cond.shape)
        v_pred = self.denoise_fn(xt, t, cond)
        ut = x1 - x0 
        t_unsqueeze = t.float() / self.num_timesteps
        t_cont = t_unsqueeze.squeeze().clamp(1e-5, 1. - 1e-5)
        lognorm_weights = 0.398942 / t_cont / (1 - t_cont) * torch.exp(-0.5 * torch.log(t_cont / ( 1 - t_cont)) ** 2)
        if self.loss_type == 'l1':
            # print(lognorm_weights[:, None, None, None].shape, ut.shape, v_pred.shape, nonpadding.unsqueeze(1).shape)
            if nonpadding is not None:
                if hparams['f0_loss_scale'] == 'lognorm':   
                    loss = (lognorm_weights[:, None, None, None] * (ut - v_pred).abs() * nonpadding.unsqueeze(1)).sum() / (nonpadding.unsqueeze(1) + 1e-8).sum()
                else:
                    loss = ((ut - v_pred).abs() * nonpadding.unsqueeze(1)).sum() / (nonpadding.unsqueeze(1) + 1e-8).sum()
            else:
                # print('are you sure w/o nonpadding?')
                if hparams['f0_loss_scale'] == 'lognorm':
                    loss = (lognorm_weights[:, None, None, None] * (ut - v_pred).abs()).mean()
                else:
                    loss = ((ut - v_pred).abs()).mean()
        elif self.loss_type == 'l2':
            if nonpadding is not None:
                if hparams['f0_loss_scale'] == 'lognorm':
                    loss = (lognorm_weights[:, None, None, None] * F.mse_loss(ut, v_pred,  reduction='none') * nonpadding.unsqueeze(1)).sum() / (nonpadding.unsqueeze(1) + 1e-8).sum()
                else:
                    loss = (F.mse_loss(ut, v_pred,  reduction='none') * nonpadding.unsqueeze(1)).sum() / (nonpadding.unsqueeze(1) + 1e-8).sum()
            else:
                loss_simple = F.mse_loss(ut, v_pred,  reduction='none')
                # print('are you sure w/o nonpadding?')
                if hparams['f0_loss_scale'] == 'lognorm':
                    loss = torch.mean(lognorm_weights[:, None, None, None] * loss_simple)
                else:
                    loss = torch.mean(loss_simple)
        else:
            raise NotImplementedError()
        
        return loss

    def forward(self, cond, f0=None, nonpadding=None, ret=None, infer=False,dyn_clip=None,solver='euler'):
        b = cond.shape[0]
        device = cond.device
        if not infer:
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
            x = f0.unsqueeze(1).unsqueeze(1)# [B, 1, M, T]
            return self.p_losses(x, t, cond, nonpadding=nonpadding)
        else:
            num = 1
            cond = cond.expand(num, -1, -1)
            shape = (cond.shape[0], 1, self.mel_bins, cond.shape[2])
            x0 = torch.randn(shape, device=device)
            neural_ode = NeuralODE(self.ode_wrapper(cond, self.num_timesteps, dyn_clip), solver=solver, sensitivity="adjoint", atol=1e-4, rtol=1e-4)
            t_span = torch.linspace(0, 1, self.K_step + 1)
            eval_points, traj = neural_ode(x0, t_span)
            x = traj[-1]
            x = x[:, 0].transpose(1, 2)
            x = x.view(-1, num, x.shape[1], x.shape[2]).mean(dim=1)
            assert x.shape == (b, x.shape[1], x.shape[2])
        return x

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min

    def cwt2f0_norm(self, cwt_spec, mean, std, mel2ph):
        return self.fs2.cwt2f0_norm(cwt_spec, mean, std, mel2ph)

    def out2mel(self, x):
        return x
    
    def ode_wrapper(self, cond, num_timesteps, dyn_clip):
        # self.estimator receives x, mask, mu, t, spk as arguments
        return Wrapper(self, cond, num_timesteps, dyn_clip)