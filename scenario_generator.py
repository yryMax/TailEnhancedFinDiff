from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import torch
from factor_diffusion_sample import generate
from factor_diffusion_train import FactorDenoiser
from factor_model import FactorModel, reconstruct_returns
from collections.abc import Callable


class FactorSampler(ABC):
    @abstractmethod
    def sample_crossectional(self, num_generate: int) -> np.ndarray:
        # (N,F,T)
        pass

    @abstractmethod
    def sample_temporal(self, num_generate: int, seq_len: int) -> np.ndarray:
        # (N,F)
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
                        (expected block length = 1/p; default 0.125 → 8 days).
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

    def sample_temporal(self, num_generate: int, seq_len: int) -> np.ndarray:
        """
        stationary boostrap
        """
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

    def sample_temporal(self, num_generate: int, seq_len: int) -> np.ndarray:
        F = self._F.shape[1]
        lam = self.ewma_lambda
        mu = self.mean
        Sigma_init = self.cov

        out = np.empty((num_generate, seq_len, F), dtype=np.float64)

        for n in range(num_generate):
            Sigma = Sigma_init.copy()
            for t in range(seq_len):
                f_t = self.rng.multivariate_normal(mean=mu, cov=Sigma)
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
    def __init__(self, checkpoint_path: str, device: str = None,
                 guidance_scale: float = 1.0, guidance_decay_pow: float = 1.0):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model = FactorDenoiser(**ckpt["model_kwargs"]).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.scaler = ckpt["scaler"]
        self.L = ckpt.get("L_noise")
        self.guidance_scale = guidance_scale
        self.guidance_decay_pow = guidance_decay_pow
        self.cfg = ckpt["cfg"]

    def sample_temporal(self, num_generate: int) -> np.ndarray:
        """Sample paths and return them stacked as (N, F, seq_len)."""
        samples, _, _ = generate(self.model, self.scaler, self.cfg,
                                 num_samples=num_generate, L=self.L)
        return samples

    def sample_crossectional(self, num_generate: int) -> np.ndarray:
        """Generate enough paths to cover num_generate timepoints, flatten across (path, time)."""
        seq_len = self.cfg["seq_len"]
        n_paths = (num_generate + seq_len) // seq_len
        paths, _, _ = generate(self.model, self.scaler, self.cfg,
                                 num_samples=n_paths, L=self.L)        # (N, F, T)
        flat    = paths.transpose(0, 2, 1).reshape(-1, paths.shape[1])           # (N*T, F)
        return flat[:num_generate]

    def cond_generate(self, num_generate: int, cond_fn: Callable[[torch.Tensor], torch.Tensor]) -> np.ndarray:

        from factor_diffusion_sample import generate
        samples, _, _ = generate(
            self.model, self.scaler, self.cfg, num_samples=num_generate,
            cond_fn=cond_fn, guidance_scale=self.guidance_scale,
            guidance_decay_pow=self.guidance_decay_pow, L=self.L,
        )
        return samples

class ScenarioGenerator:
    def __init__(self, model: FactorModel, sampler: FactorSampler):
        self.model = model
        self.sampler = sampler

    def factor_generate(self, num_generate: int) -> np.ndarray:
        return self.sampler.sample_crossectional(num_generate)

    def factor_generate_temporal(self, num_generate: int, seq_len) -> np.ndarray:
        return self.sampler.sample_temporal(num_generate, seq_len)

    def cond_factor_generate(self, num_generate: int, cond_fn: Callable[[torch.Tensor], torch.Tensor]) -> np.ndarray:
        return self.sampler.cond_generate(num_generate, cond_fn)

    def stock_generate(self, num_generate: int, cond_fn: Callable[[torch.Tensor], torch.Tensor] = None) -> np.ndarray:
        if cond_fn is not None:
            fs = self.cond_factor_generate(num_generate, cond_fn)
        else:
            fs = self.factor_generate(num_generate)

        fs_full = np.column_stack([np.ones((len(fs), 1)), fs])
            
        returns = reconstruct_returns(self.model, fs_full)
        return returns