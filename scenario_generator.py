from __future__ import annotations
import warnings
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import torch
from factor_diffusion_sample import generate_uncond, generate_path, _reverse
from factor_diffusion_train import FactorDenoiser
from factor_model import FactorModel, reconstruct_returns
from collections.abc import Callable


class FactorSampler(ABC):
    @abstractmethod
    def sample_crossectional(self, num_generate: int) -> np.ndarray:
        # (N,F,T)
        pass

    @abstractmethod
    def sample_temporal(self, num_generate: int, seq_len: int,
                        cond_fn: Callable[[torch.Tensor], torch.Tensor] = None,
                        vix_cond: float = None) -> np.ndarray:
        # vix_cond: optional constant VIX regime to condition on; ignored by samplers
        # that have no VIX notion (ResampleSampler, GaussianSampler).
        pass

    @abstractmethod
    def cond_generate(self, num_generate: int, cond_fn: Callable[[torch.Tensor], torch.Tensor]) -> np.ndarray:
        pass

class ResampleSampler(FactorSampler):
    def __init__(self, train_factors: pd.DataFrame, scaler, guidance_scale: float = 1.0,
                 rng: np.random.Generator = None):
        """
        :param seq_len: required for sample_temporal; path length per draw.
        :param p:       stationary-bootstrap block restart probability
        """
        self.factors = train_factors.drop(columns=["alpha"]).values if "alpha" in train_factors.columns else train_factors.values
        self.scaler = scaler
        self.batch_size = 1024
        self.max_batches = 500
        self.guidance_scale = guidance_scale
        self.rng = rng if rng is not None else np.random.default_rng()

    def sample_crossectional(self, num_generate: int) -> np.ndarray:
        idx = self.rng.choice(len(self.factors), size=num_generate, replace=True)
        return self.factors[idx]

    def sample_temporal(self, num_generate: int, seq_len: int,
                        cond_fn: Callable[[torch.Tensor], torch.Tensor] = None,
                        vix_cond: float = None) -> np.ndarray:
        """
        stationary boostrap
        """
        if vix_cond is not None:
            warnings.warn("ResampleSampler ignores vix_cond (no VIX notion).")
        if cond_fn is not None:
            raise NotImplementedError(
                "ResampleSampler.sample_temporal: per-day conditioning is not "
                "supported for stationary bootstrap (non-parametric, no clean "
                "rejection mechanism). Use DiffusionSampler or GaussianSampler.")

        N, F = self.factors.shape
        idx  = self.rng.integers(0, N, size=num_generate)
        out  = np.empty((num_generate, seq_len, F), dtype=self.factors.dtype)

        for t in range(seq_len):
            out[:, t, :] = self.factors[idx]
            jump    = self.rng.random(num_generate) < 0.125
            new_idx = self.rng.integers(0, N, size=num_generate)
            idx     = np.where(jump, new_idx, (idx + 1) % N)

        return out.transpose(0, 2, 1)                                            # (num_generate, F, seq_len)

    def cond_generate(self, num_generate: int, cond_fn: Callable[[torch.Tensor], torch.Tensor]) -> np.ndarray:
        accepted = []
        n_tried  = 0
        _mean  = torch.tensor(self.scaler.mean_,  dtype=torch.float32)
        _scale = torch.tensor(self.scaler.scale_, dtype=torch.float32)
        D = len(_mean)
        for _ in range(self.max_batches):
            if len(accepted) >= num_generate:
                break
            candidates = self.sample_crossectional(self.batch_size)
            n_tried += len(candidates)
            for i in range(len(candidates)):
                xi = torch.tensor(candidates[i:i+1], dtype=torch.float32)
                xi_norm = (xi - _mean) / _scale
                loss_i = float(cond_fn(xi_norm))

                accept = self.rng.random() < np.exp(-self.guidance_scale * loss_i)

                if accept:
                    accepted.append(candidates[i])
                if len(accepted) >= num_generate:
                    break
        if not accepted:
            return np.empty((0, D))
        return np.stack(accepted[:num_generate])

class GaussianSampler(FactorSampler):
    def __init__(self, train_factors: pd.DataFrame, scaler,
                 guidance_scale: float = 1.0, rng: np.random.Generator = None):
        """
        :param seq_len:     path length for sample_temporal; ignored by other methods
        :param ewma_lambda: EWMA decay for time-varying Σ (RiskMetrics default 0.94)
        :param warmup:      tail of training data used to warm-start Σ_0
        """
        factors = train_factors.drop(columns=["alpha"]) if "alpha" in train_factors.columns else train_factors
        self._F = factors.values.astype(np.float64)
        self.mean = factors.mean().values.astype(np.float64)
        self.cov = np.cov(self._F, rowvar=False)
        self.scaler = scaler
        self.batch_size = 1024
        self.max_batches = 500
        self.guidance_scale = guidance_scale
        self.rng = rng if rng is not None else np.random.default_rng()
        self.ewma_lambda = 0.94

    def sample_crossectional(self, num_generate: int) -> np.ndarray:
        return self.rng.multivariate_normal(mean=self.mean, cov=self.cov, size=num_generate)

    def sample_temporal(self, num_generate: int, seq_len: int,
                        cond_fn: Callable[[torch.Tensor], torch.Tensor] = None,
                        vix_cond: float = None) -> np.ndarray:
        """
        Per-path EWMA Gaussian rollout. When `cond_fn` is given, each day's
        draw is soft-rejected (accept prob exp(-self.guidance_scale * energy)).
        """
        if vix_cond is not None:
            warnings.warn("GaussianSampler ignores vix_cond (no VIX notion).")
        F   = self._F.shape[1]
        lam = self.ewma_lambda
        mu  = self.mean
        out = np.empty((num_generate, seq_len, F), dtype=np.float64)

        sc_mean  = self.scaler.mean_
        sc_scale = self.scaler.scale_
        max_retries = 200

        for n in range(num_generate):
            Sigma = self.cov.copy()
            for t in range(seq_len):
                f_t = self.rng.multivariate_normal(mu, Sigma)

                if cond_fn is not None:
                    for _ in range(max_retries):
                        f_norm = ((f_t - sc_mean) / sc_scale).astype(np.float32)
                        energy = float(cond_fn(torch.from_numpy(f_norm[None])))
                        if self.rng.random() < np.exp(-self.guidance_scale * energy):
                            break
                        f_t = self.rng.multivariate_normal(mu, Sigma)

                out[n, t, :] = f_t
                d = f_t - mu
                Sigma = lam * Sigma + (1 - lam) * np.outer(d, d)

        return out.transpose(0, 2, 1).astype(np.float32)

    def cond_generate(self, num_generate: int, cond_fn: Callable[[torch.Tensor], torch.Tensor]) -> np.ndarray:
        accepted = []
        n_tried  = 0
        _mean  = torch.tensor(self.scaler.mean_,  dtype=torch.float32)
        _scale = torch.tensor(self.scaler.scale_, dtype=torch.float32)
        D = len(_mean)

        for _ in range(self.max_batches):
            if len(accepted) >= num_generate:
                break
            candidates = self.sample_crossectional(self.batch_size)
            n_tried += len(candidates)
            for i in range(len(candidates)):
                xi = torch.tensor(candidates[i:i+1], dtype=torch.float32)
                xi_norm = (xi - _mean) / _scale
                loss_i = float(cond_fn(xi_norm))

                accept = self.rng.random() < np.exp(-self.guidance_scale * loss_i)

                if accept:
                    accepted.append(candidates[i])
                if len(accepted) >= num_generate:
                    break
        if not accepted:
            return np.empty((0, D))
        return np.stack(accepted[:num_generate])

class DiffusionSampler(FactorSampler):
    """Markov-1 conditional DLPM (F_t → F_{t+1}) with a learned null token.

    - sample_crossectional: unconditional marginal p(F_0) via the null token.
    - sample_temporal:       self-starting autoregressive rollout.
    - cond_generate:         energy-guided unconditional draw (gradient guidance).
    """

    def __init__(self, checkpoint_path: str, device: str = None,
                 guidance_scale: float = 1.0, guidance_decay_pow: float = 1.0):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model = FactorDenoiser(**ckpt["model_kwargs"]).to(self.device)
        self.model.load_state_dict(ckpt["ema_state"])
        self.model.eval()
        self.scaler = ckpt["scaler"]
        self.vix_scaler = ckpt.get("vix_scaler")          # None if trained without VIX
        self.guidance_scale = guidance_scale
        self.guidance_decay_pow = guidance_decay_pow
        self.cfg = ckpt["cfg"]

    def sample_crossectional(self, num_generate: int) -> np.ndarray:
        """Unconditional cross-section: draw the marginal p(F_0) from the null token."""
        return generate_uncond(self.model, self.scaler, self.cfg, num_generate)   # (N, F)

    def sample_temporal(self, num_generate: int, seq_len: int,
                        cond_fn: Callable[[torch.Tensor], torch.Tensor] = None,
                        vix_cond: float = None) -> np.ndarray:
        """Self-starting autoregressive rollout, returned as (N, F, seq_len).
        cond_fn is applied per-day via gradient guidance through generate_path.
        vix_cond: optional constant VIX level to condition the whole path on; None → VIX-free.
        """
        if vix_cond is not None and self.vix_scaler is None:
            raise ValueError("vix_cond given but checkpoint has no vix_scaler "
                             "(model trained without VIX).")
        paths = generate_path(self.model, self.scaler, self.cfg,
                              horizon=seq_len, num_paths=num_generate,
                              vix_cond=vix_cond, vix_scaler=self.vix_scaler,
                              cond_fn=cond_fn, guidance_scale=self.guidance_scale,
                              guidance_decay_pow=self.guidance_decay_pow)         # (N, seq_len, F)
        return paths.transpose(0, 2, 1)                                           # (N, F, seq_len)

    def cond_generate(self, num_generate: int, cond_fn: Callable[[torch.Tensor], torch.Tensor]) -> np.ndarray:
        """Energy-guided cross-section: guide the null-token marginal p(F_0) by cond_fn."""
        factor_dim = len(self.cfg["factors"])
        x, _, _ = _reverse(self.model, self.cfg, None,
                           n=num_generate, factor_dim=factor_dim,
                           cond_fn=cond_fn, guidance_scale=self.guidance_scale,
                           guidance_decay_pow=self.guidance_decay_pow)
        return self.scaler.inverse_transform(x.cpu().numpy())

class ScenarioGenerator:
    def __init__(self, model: FactorModel, sampler: FactorSampler):
        self.model = model
        self.sampler = sampler

    def factor_generate(self, num_generate: int) -> np.ndarray:
        return self.sampler.sample_crossectional(num_generate)

    def factor_generate_temporal(self, num_generate: int, seq_len: int,
                                  cond_fn: Callable[[torch.Tensor], torch.Tensor] = None,
                                  vix_cond: float = None) -> np.ndarray:
        return self.sampler.sample_temporal(num_generate, seq_len, cond_fn=cond_fn,
                                            vix_cond=vix_cond)

    def cond_factor_generate(self, num_generate: int, cond_fn: Callable[[torch.Tensor], torch.Tensor]) -> np.ndarray:
        return self.sampler.cond_generate(num_generate, cond_fn)

    def stock_generate(self, num_generate: int, cond_fn: Callable[[torch.Tensor], torch.Tensor] = None,
                       return_components: bool = False):
        if cond_fn is not None:
            fs = self.cond_factor_generate(num_generate, cond_fn)
        else:
            fs = self.factor_generate(num_generate)

        fs_full = np.column_stack([np.ones((len(fs), 1)), fs])
        return reconstruct_returns(self.model, fs_full, return_components=return_components)

    def stock_generate_temporal(self, num_generate: int, seq_len: int,
                                 cond_fn: Callable[[torch.Tensor], torch.Tensor] = None,
                                 fs_cache: str = None,
                                 return_components: bool = False,
                                 vix_cond: float = None):
        """
        Generate temporal stock returns by sampling factor paths then mapping through
        the factor model with fresh idiosyncratic noise per (path, day).

        :param fs_cache: optional .npy path for factor paths. If the file exists,
                         load it instead of regenerating (must match shape (N, F, T)).
                         If missing, generate and save there. NOTE: when loading from
                         cache, `vix_cond` has no effect (the cached paths are used as-is).
        :param return_components: if True, return (R, systematic, idiosyncratic) each (N, S, T).
        :param vix_cond: optional constant VIX level to condition factor paths on (passed
                         through to the sampler); None → VIX-free. Ignored by non-diffusion
                         samplers and when loading from `fs_cache`.
        :return: (num_generate, S, seq_len) stock returns; or 3-tuple of same shape.
        """
        import os
        if fs_cache is not None and os.path.isfile(fs_cache):
            fs = np.load(fs_cache)
            assert fs.shape[0] == num_generate and fs.shape[2] == seq_len, \
                f"cached fs shape {fs.shape} != (num_generate={num_generate}, F, seq_len={seq_len})"
        else:
            fs = self.factor_generate_temporal(num_generate, seq_len, cond_fn=cond_fn,
                                               vix_cond=vix_cond)                       # (N, F, T)
            if fs_cache is not None:
                os.makedirs(os.path.dirname(fs_cache) or ".", exist_ok=True)
                np.save(fs_cache, fs)

        N, F, T = fs.shape
        alpha   = np.ones((N, 1, T), dtype=fs.dtype)
        fs_full = np.concatenate([alpha, fs], axis=1)                    # (N, F+1, T)
        fs_flat = fs_full.transpose(0, 2, 1).reshape(N * T, F + 1)       # (N*T, F+1)

        out = reconstruct_returns(self.model, fs_flat, return_components=return_components)

        def _back(flat):
            S = flat.shape[1]
            return flat.reshape(N, T, S).transpose(0, 2, 1)              # (N, S, T)

        if return_components:
            R_flat, sys_flat, idio_flat = out
            return _back(R_flat), _back(sys_flat), _back(idio_flat)
        return _back(out)