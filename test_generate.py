import numpy as np
import pytest
import torch
from unittest.mock import patch

from factor_diffusion_train import FactorDenoiser
from factor_diffusion_levy import levy_noise_schedule
from factor_diffusion_sample import (
    CHECKPOINT, DEVICE, FACTOR_NAMES, FACTOR_DIM,
    LEVY_ALPHA, NUM_TIMESTEPS, generate,
)

N_SAMPLES = 64
VOL_IDX   = FACTOR_NAMES.index("volatility")
THRESHOLD = 0.001


@pytest.fixture(scope="module")
def artifacts():
    ckpt  = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model = FactorDenoiser(**ckpt["model_kwargs"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    gammas, bargammas, sigmas, barsigmas = levy_noise_schedule(LEVY_ALPHA, NUM_TIMESTEPS)
    return model, gammas, bargammas, sigmas, barsigmas, ckpt["scaler"]


def make_threshold_guidance(threshold, scaler):
    dummy = np.zeros((1, FACTOR_DIM))
    dummy[0, VOL_IDX] = threshold
    threshold_norm = float(scaler.transform(dummy)[0, VOL_IDX])

    def cond_fn(x0_hat):
        return torch.relu(threshold_norm - x0_hat[:, VOL_IDX]).pow(2).mean()

    return cond_fn

def _to_norm(scaler, value, idx):
    """Transform a single original-space value to normalized space for factor idx."""
    dummy = np.zeros((1, FACTOR_DIM))
    dummy[0, idx] = value
    return float(scaler.transform(dummy)[0, idx])


def _make_lo_cond(q_norm, idx):
    """Loss: penalise x0_hat[:, idx] > q_norm  (push below q1)."""
    def cond_fn(x0_hat):
        return torch.relu(x0_hat[:, idx] - q_norm).pow(2).mean()
    return cond_fn


def _make_hi_cond(q_norm, idx):
    """Loss: penalise x0_hat[:, idx] < q_norm  (push above q99)."""
    def cond_fn(x0_hat):
        return torch.relu(q_norm - x0_hat[:, idx]).pow(2).mean()
    return cond_fn


def test_conditional_single(artifacts):
    model, gammas, bargammas, sigmas, barsigmas, scaler = artifacts

    with patch("factor_diffusion_sample.NUM_GENERATE", N_SAMPLES):
        uncon = generate(model, gammas, bargammas, sigmas, barsigmas,
                         LEVY_ALPHA, scaler, cond_fn=None)

    q1  = np.percentile(uncon, 1,  axis=0)
    q99 = np.percentile(uncon, 99, axis=0)

    failures = []

    for i, name in enumerate(FACTOR_NAMES):
        q1_norm  = _to_norm(scaler, q1[i],  i)
        q99_norm = _to_norm(scaler, q99[i], i)

        with patch("factor_diffusion_sample.NUM_GENERATE", N_SAMPLES):
            lo = generate(model, gammas, bargammas, sigmas, barsigmas,
                          LEVY_ALPHA, scaler,
                          cond_fn=_make_lo_cond(q1_norm, i), guidance_scale=5.0)
            hi = generate(model, gammas, bargammas, sigmas, barsigmas,
                          LEVY_ALPHA, scaler,
                          cond_fn=_make_hi_cond(q99_norm, i), guidance_scale=5.0)

        lo_mean  = lo[:, i].mean()
        hi_mean  = hi[:, i].mean()
        pass_lo  = lo_mean < q1[i]
        pass_hi  = hi_mean > q99[i]

        print(f"  {name:<18} {q1[i]:>9.4f} {q99[i]:>9.4f} "
              f"{lo_mean:>10.4f} {hi_mean:>10.4f}  ")

        if not pass_lo:
            failures.append(f"{name} lo: mean={lo_mean:.4f} not < q1={q1[i]:.4f}")
        if not pass_hi:
            failures.append(f"{name} hi: mean={hi_mean:.4f} not > q99={q99[i]:.4f}")

    assert not failures, "\nFailed conditions:\n" + "\n".join(failures)
