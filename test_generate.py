import numpy as np
import pytest
import torch
import matplotlib.pyplot as plt
from unittest.mock import patch


def plot_var_grad(var_history, grad_history, title="DLPM reverse-process diagnostics"):
    """Plot posterior variance and guidance gradient histories from generate()."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=13)

    if var_history:
        ts_v   = [r[0] for r in var_history[1:]]
        mean_v = [r[1] for r in var_history[1:]]
        max_v  = [r[2] for r in var_history[1:]]

        ax = axes[0, 0]
        ax.plot(ts_v, mean_v, label="mean var")
        ax.set(title="Posterior variance  —  mean over (n, D)", xlabel="t", ylabel="var")
        ax.invert_xaxis()
        ax.legend()

        ax = axes[0, 1]
        ax.plot(ts_v, max_v, color="tab:orange", label="max var")
        ax.set(title="Posterior variance  —  max over (n, D)", xlabel="t", ylabel="var")
        ax.invert_xaxis()
        ax.legend()

    if grad_history:
        ts_g   = [r[0] for r in grad_history[1:]]
        mean_g = [r[1] for r in grad_history[1:]]
        max_g  = [r[2] for r in grad_history[1:]]

        ax = axes[1, 0]
        ax.plot(ts_g, mean_g, color="tab:green", label="mean |grad|")
        ax.set(title="Guidance gradient  —  mean over (n, D)", xlabel="t", ylabel="grad")
        ax.invert_xaxis()
        ax.legend()

        ax = axes[1, 1]
        ax.plot(ts_g, max_g, color="tab:red", label="max |grad|")
        ax.set(title="Guidance gradient  —  max over (n, D)", xlabel="t", ylabel="grad")
        ax.invert_xaxis()
        ax.legend()

    plt.tight_layout()
    plt.show()
    return fig

from factor_diffusion_train import FactorDenoiser
from factor_diffusion_levy import levy_noise_schedule
from factor_diffusion_sample import (
    CHECKPOINT, DEVICE, FACTOR_NAMES, FACTOR_DIM,
    LEVY_ALPHA, NUM_TIMESTEPS, generate,
)

N_SAMPLES = 128
VOL_IDX   = FACTOR_NAMES.index("volatility")


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
    """Push x0_hat[:, idx] below q_norm. Returns shape (n,)."""
    def cond_fn(x0_hat):
        return torch.relu(x0_hat[:, idx] - q_norm).pow(2)
    return cond_fn


def _make_hi_cond(q_norm, idx):
    """Push x0_hat[:, idx] above q_norm. Returns shape (n,)."""
    def cond_fn(x0_hat):
        return torch.relu(q_norm - x0_hat[:, idx]).pow(2)
    return cond_fn


def test_demo(artifacts):
    model, gammas, bargammas, sigmas, barsigmas, scaler = artifacts

    with patch("factor_diffusion_sample.NUM_GENERATE", N_SAMPLES):
        uncon, _, _ = generate(model, gammas, bargammas, sigmas, barsigmas,
                               LEVY_ALPHA, scaler, cond_fn=None)

    q1      = np.percentile(uncon, 1, axis=0)
    q1_norm = _to_norm(scaler, q1[VOL_IDX], VOL_IDX)
    def cf(x0_hat):
        return torch.relu(x0_hat[:, VOL_IDX] - q1_norm)
    with patch("factor_diffusion_sample.NUM_GENERATE", N_SAMPLES):
        lo, var_history, grad_history = generate(
            model, gammas, bargammas, sigmas, barsigmas,
            LEVY_ALPHA, scaler,
            cond_fn=cf, guidance_scale=5.0,
        )

    print(f"Demo: vol q1={q1[VOL_IDX]:.4f}  lo mean={lo[:, VOL_IDX].mean():.4f}")

    if grad_history:
        print("max grad first 10 steps:", [r[2] for r in grad_history[1:10]])
        print("min grad first 10 steps:", [r[1] for r in grad_history[1:10]])

    plot_var_grad(var_history, grad_history,
                  title="DLPM reverse-process diagnostics (batch-0, guided)")

def test_conditional_single(artifacts):
    model, gammas, bargammas, sigmas, barsigmas, scaler = artifacts

    with patch("factor_diffusion_sample.NUM_GENERATE", N_SAMPLES):
        uncon, _, _ = generate(model, gammas, bargammas, sigmas, barsigmas,
                               LEVY_ALPHA, scaler, cond_fn=None)

    q1  = np.percentile(uncon, 1,  axis=0)
    q99 = np.percentile(uncon, 99, axis=0)

    failures = []

    for i, name in enumerate(FACTOR_NAMES):
        q1_norm  = _to_norm(scaler, q1[i],  i)
        q99_norm = _to_norm(scaler, q99[i], i)

        with patch("factor_diffusion_sample.NUM_GENERATE", N_SAMPLES):
            lo, _, _ = generate(model, gammas, bargammas, sigmas, barsigmas,
                                LEVY_ALPHA, scaler,
                                cond_fn=_make_lo_cond(q1_norm, i), guidance_scale=5.0)
            hi, _, _ = generate(model, gammas, bargammas, sigmas, barsigmas,
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
