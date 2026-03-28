"""
Lévy noise utilities for DLPM (Denoising Lévy Probabilistic Models).
Implements the CMS sampler and DLPM cosine noise schedule.
Reference: https://arxiv.org/abs/2407.18609
"""
import math
import torch
from torch.distributions import Exponential

CLAMP_A = 2000  # clip skewed-Lévy samples to prevent extremes


# ── Noise schedule ──────────────────────────────────────────────────────────────

def levy_noise_schedule(alpha: float, T: int, s: float = 0.008):
    """
    DLPM scale-preserving cosine noise schedule.

    Identical to DDPM cosine schedule when alpha=2.
    Returns (gammas, bargammas, sigmas, barsigmas), all shape (T,).

    Forward process:  x_t = bargammas[t] * x_0 + barsigmas[t] * eps
    where eps ~ SαS (symmetric alpha-stable).
    """
    t          = torch.arange(T, dtype=torch.float32)
    schedule   = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
    baralphas  = schedule / schedule[0]
    betas      = 1 - baralphas / torch.cat([baralphas[:1], baralphas[:-1]])
    alphas_t   = 1 - betas

    gammas    = alphas_t ** (1.0 / alpha)
    bargammas = torch.cumprod(gammas, dim=0)
    sigmas    = (1 - gammas    ** alpha) ** (1.0 / alpha)
    barsigmas = (1 - bargammas ** alpha) ** (1.0 / alpha)
    return gammas, bargammas, sigmas, barsigmas


# ── Lévy samplers ───────────────────────────────────────────────────────────────

def _cms_sample(alpha: float, beta: float, n: int) -> torch.Tensor:
    """
    Chambers-Mallows-Stuck algorithm for S(alpha, beta) stable distribution.
    Returns float32 tensor of shape (n,).
    alpha != 1 required.
    """
    TH    = (torch.rand(n) * math.pi - math.pi / 2).double()
    W     = Exponential(torch.ones(n)).sample().squeeze().double()

    val0  = beta * math.tan(math.pi * alpha / 2)
    th0   = math.atan(val0) / alpha

    aTH   = alpha * TH
    cos_t = torch.cos(TH)
    tan_t = torch.tan(TH)

    denom = cos_t / torch.tan(alpha * (th0 + TH)) + torch.sin(TH)
    val3  = W / denom
    num   = (torch.cos(aTH) + torch.sin(aTH) * tan_t
             - val0 * (torch.sin(aTH) - torch.cos(aTH) * tan_t))

    # abs() before ** (1/alpha) for numerical safety; result is positive for beta=1
    res = val3 * (num / W).abs() ** (1.0 / alpha)
    return res.float()


def sample_skewed_levy(alpha: float, shape: tuple, device=None) -> torch.Tensor:
    """
    Sample a_t ~ S(alpha/2, beta=1): positive right-skewed stable r.v.
    Used as the variance multiplier:  Sigma_t = a_t * barsigmas[t]^2

    Oversample by 2x then take the first n finite-positive values.
    """
    n   = math.prod(shape)
    raw = _cms_sample(alpha / 2.0, 1.0, n * 2)
    raw = torch.nan_to_num(raw, nan=1.0, posinf=CLAMP_A, neginf=0.0)
    raw = raw.clamp(0.0, CLAMP_A)
    a   = raw[:n].reshape(shape)
    return a.to(device) if device is not None else a


def sample_sas(shape: tuple, a: torch.Tensor) -> torch.Tensor:
    """
    Sample symmetric alpha-stable noise via the Gaussian-Lévy scale mixture:
        eps = sqrt(a) * N(0, I),   a ~ S(alpha/2, beta=1)

    Conditioned on a, this is just rescaled Gaussian — no special sampler needed.
    """
    return torch.sqrt(a) * torch.randn_like(a)
