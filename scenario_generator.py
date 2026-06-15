from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import torch
from factor_diffusion_sample import generate_style_transfer
from factor_diffusion_train import FactorDenoiser
from factor_model import FactorModel, reconstruct_returns
from collections.abc import Callable


class FactorSampler(ABC):
    @abstractmethod
    def generate(self, num_generate: int) -> np.ndarray:
        """generate unconditional samples"""
        pass

    @abstractmethod
    def cond_generate(self, num_generate: int, cond_fn: Callable[[torch.Tensor], torch.Tensor]) -> np.ndarray:
        """
        generate condition samples
        :param num_generate: number of samples to generate
        :param cond_fn: a loss function that take a batch sample as input and return a loss score(should be aggregated later)
        if the condition is fullfilled, cond(x) should return all 0
        """
        pass

class ResampleSampler(FactorSampler):
    def __init__(self, train_factors: pd.DataFrame, scaler, guidance_scale: float = 1.0, rng: np.random.Generator = None):
        self.factors = train_factors.drop(columns=["alpha"]).values if "alpha" in train_factors.columns else train_factors.values
        self.scaler = scaler
        self.batch_size = 1024
        self.max_batches = 500
        self.guidance_scale = guidance_scale
        self.rng = rng if rng is not None else np.random.default_rng()

    def generate(self, num_generate: int) -> np.ndarray:
        idx = self.rng.choice(len(self.factors), size=num_generate, replace=True)
        return self.factors[idx]

    def cond_generate(self, num_generate: int, cond_fn: Callable[[torch.Tensor], torch.Tensor]) -> np.ndarray:
        accepted = []
        n_tried  = 0
        _mean  = torch.tensor(self.scaler.mean_,  dtype=torch.float32)
        _scale = torch.tensor(self.scaler.scale_, dtype=torch.float32)
        D = len(_mean)
        for _ in range(self.max_batches):
            if len(accepted) >= num_generate:
                break
            candidates = self.generate(self.batch_size)
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
        rate = len(accepted) / max(n_tried, 1)
        if not accepted:
            return np.empty((0, D))
        return np.stack(accepted[:num_generate])

class GaussianSampler(FactorSampler):
    def __init__(self, train_factors: pd.DataFrame, scaler, guidance_scale: float = 1.0, rng: np.random.Generator = None):
        factors = train_factors.drop(columns=["alpha"]) if "alpha" in train_factors.columns else train_factors
        self.mean = factors.mean().values
        self.cov = np.cov(factors.values, rowvar=False)
        self.scaler = scaler
        self.batch_size = 1024
        self.max_batches = 500
        self.guidance_scale = guidance_scale
        self.rng = rng if rng is not None else np.random.default_rng()

    def generate(self, num_generate: int) -> np.ndarray:
        return self.rng.multivariate_normal(mean=self.mean, cov=self.cov, size=num_generate)

    def cond_generate(self, num_generate: int, cond_fn: Callable[[torch.Tensor], torch.Tensor]) -> np.ndarray:
        accepted = []
        n_tried  = 0
        _mean  = torch.tensor(self.scaler.mean_,  dtype=torch.float32)
        _scale = torch.tensor(self.scaler.scale_, dtype=torch.float32)
        D = len(_mean)

        for _ in range(self.max_batches):
            if len(accepted) >= num_generate:
                break

            candidates = self.generate(self.batch_size)
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
        rate = len(accepted) / max(n_tried, 1)
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

    def generate(self, num_generate: int) -> np.ndarray:
        from factor_diffusion_sample import generate
        samples, _, _ = generate(self.model, self.scaler, self.cfg,
                                 num_samples=num_generate, L=self.L)
        return samples

    def cond_generate(self, num_generate: int, cond_fn: Callable[[torch.Tensor], torch.Tensor]) -> np.ndarray:
        from factor_diffusion_sample import generate
        samples, _, _ = generate(
            self.model, self.scaler, self.cfg, num_samples=num_generate,
            cond_fn=cond_fn, guidance_scale=self.guidance_scale,
            guidance_decay_pow=self.guidance_decay_pow, L=self.L,
        )
        return samples

    def transfer(self, x0_obs: np.ndarray, cond_fn: Callable[[torch.Tensor], torch.Tensor],
                 strength: float = 0.6) -> np.ndarray:
        samples, _, _ = generate_style_transfer(
            self.model, self.scaler, self.cfg, x0_obs, cond_fn, strength=strength,
            guidance_scale=self.guidance_scale, guidance_decay_pow=self.guidance_decay_pow,
        )
        return samples

class ScenarioGenerator:
    def __init__(self, model: FactorModel, sampler: FactorSampler):
        self.model = model
        self.sampler = sampler


    def factor_generate(self, num_generate: int) -> np.ndarray:
        return self.sampler.generate(num_generate)

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

    def style_transfer_oos(self, oos_factors: np.ndarray,
                           cond_fn: Callable[[torch.Tensor], torch.Tensor],
                           strength: float = 0.6):
        """
        Per-day OOS style transfer. Each row of `oos_factors` is one observed OOS day; the whole
        panel is processed as a single batch (the batch holds different days, one counterfactual
        per day). The scenario is carried by `cond_fn` (guidance); each day's identity by the
        SDEdit anchor. Loadings/residuals come from self.model (the train model).

        :param oos_factors: (D, factor_dim) raw observed OOS factors, alpha column dropped,
                            column order matching the diffusion scaler / self.model.F.
        :param cond_fn:     scenario condition cond_fn(x0_hat) -> per-sample loss.
        :param strength:    SDEdit strength in (0, 1].
        :return: (cf_factors (D, factor_dim), cf_returns (D, S)); rows align 1:1 with OOS days.
        """
        x0_obs = np.asarray(oos_factors, dtype=np.float32)
        cf_factors = self.sampler.transfer(x0_obs, cond_fn, strength=strength)

        fs_full = np.column_stack([np.ones((len(cf_factors), 1)), cf_factors])
        cf_returns = reconstruct_returns(self.model, fs_full)
        return cf_factors, cf_returns