"""
Code Borrowed from https://github.com/darioShar/DLPM
"""
import math
import torch
from torch.distributions import Exponential

def levy_noise_schedule(alpha: float, T: int, s: float = 0.008):
    """
    Cosine noise scheduler
    :param alpha: control the tail heaviness of the noise distribution; alpha=2 corresponds to Gaussian, smaller alpha means heavier tails
    :param T: total number of time steps
    :param s: cosine schedule offset https://arxiv.org/abs/2102.09672
    :return: (gammas, bargammas, sigmas, barsigmas), each of shape (T,) s.t.
             x_t = bargammas[t] * x_0 + barsigmas[t] * eps, eps ~ SaS,
             with scale-preserving constraint bargammas[t]^alpha + barsigmas[t]^alpha = 1
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

def sample_skewed_levy(alpha: float, shape: tuple, device=None) -> torch.Tensor:
    """
    Sample a ~ S(alpha/2, beta=1): positive right-skewed stable r.v. via the
    Chambers-Mallows-Stuck algorithm.
    Special case: alpha=2 (Gaussian) returns constant 1 to get N(0,1) after mixture
    :param alpha: stability index of the target SaS distribution; a ~ S(alpha/2, 1)
    :param shape: output shape
    :param device: target device
    :return: tensor of shape `shape` with positive stable samples, clamped to [0, 2000]
    """
    if alpha == 2.0:
        raw = torch.ones(shape)
        return raw.to(device) if device is not None else raw

    CLAMP_A = 2000
    n     = math.prod(shape)
    a     = alpha / 2.0
    TH    = (torch.rand(n) * (math.pi - 0.3) - (math.pi - 0.3) / 2).double()
    W     = Exponential(torch.ones(n)).sample().squeeze().double()

    val0  = math.tan(math.pi * a / 2)       # beta=1
    th0   = math.atan(val0) / a

    aTH   = a * TH
    cos_t = torch.cos(TH)
    tan_t = torch.tan(TH)

    denom = cos_t / torch.tan(a * (th0 + TH)) + torch.sin(TH)
    num   = (torch.cos(aTH) + torch.sin(aTH) * tan_t
             - val0 * (torch.sin(aTH) - torch.cos(aTH) * tan_t))
    raw   = (W / denom) * (num / W) ** (1.0 / a)

    raw   = raw.float().clamp(0.0, CLAMP_A).reshape(shape)
    return raw.to(device) if device is not None else raw


def sample_sas(a: torch.Tensor) -> torch.Tensor:
    """
    Sample symmetric alpha-stable noise S(alpha, beta) via the Gaussian scale mixture:
        eps = sqrt(a) * N(0, I),   a ~ S(alpha/2, beta=1)

    Conditioned on a, this is just rescaled Gaussian — no special sampler needed.
    """
    return torch.sqrt(a) * torch.randn_like(a)
