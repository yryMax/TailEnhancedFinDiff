import numpy as np
import pytest
import torch
import matplotlib.pyplot as plt
from factor_diffusion_train import FactorDenoiser
from factor_diffusion_levy import levy_noise_schedule
import yaml
from factor_diffusion_sample import generate
from unittest.mock import patch


with open("cfg.yaml") as f:
    _all_cfg = yaml.safe_load(f)
    _tcfg    = _all_cfg["train"]
    _scfg    = _all_cfg["sample"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_TIMESTEPS = _tcfg["num_timesteps"]
LEVY_ALPHA    = _tcfg["levy_alpha"]
PREFIX        = _tcfg["prefix"]
BATCH_SIZE    = _tcfg["batch_size"]
FACTOR_NAMES  = _tcfg["factor_names"]
NUM_GENERATE  = _scfg["num_generate"]
FACTOR_DIM    = len(_tcfg["factor_names"])
CHECKPOINT    = f"{PREFIX}/checkpoints/factor_DLPM_ep0200.pt"
OUT_PATH      = f"{PREFIX}/samples/factor_{NUM_GENERATE}.npy"

N_SAMPLES = 128
VOL_IDX   = FACTOR_NAMES.index("volatility")

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


@pytest.fixture(scope="module")
def artifacts():
    ckpt  = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model = FactorDenoiser(**ckpt["model_kwargs"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    gammas, bargammas, sigmas, barsigmas = levy_noise_schedule(LEVY_ALPHA, NUM_TIMESTEPS)
    return model, gammas, bargammas, sigmas, barsigmas, ckpt["scaler"]


def _to_norm(scaler, value, idx):
    """Transform a single original-space value to normalized space for factor idx."""
    dummy = np.zeros((1, FACTOR_DIM))
    dummy[0, idx] = value
    return float(scaler.transform(dummy)[0, idx])


def _make_lo_cond(q_norm, idx):
    """Push x0_hat[:, idx] below q_norm. Returns shape (n,)."""
    def cond_fn(x0_hat):
        return torch.relu(x0_hat[:, idx] - q_norm)
    return cond_fn


def _make_hi_cond(q_norm, idx):
    """Push x0_hat[:, idx] above q_norm. Returns shape (n,)."""
    def cond_fn(x0_hat):
        return torch.relu(q_norm - x0_hat[:, idx])
    return cond_fn


def _make_interval_cond(lo_norm, hi_norm, idx):
    """Bilateral linear penalty to keep x0_hat[:, idx] in [lo_norm, hi_norm]. Returns shape (n,)."""
    def cond_fn(x0_hat):
        return (torch.relu(x0_hat[:, idx] - hi_norm) +
                torch.relu(lo_norm - x0_hat[:, idx]))
    return cond_fn


def test_demo(artifacts):
    model, gammas, bargammas, sigmas, barsigmas, scaler = artifacts

    with patch("factor_diffusion_sample.NUM_GENERATE", N_SAMPLES):
        uncon, _, _ = generate(model, gammas, bargammas, sigmas, barsigmas,
                               LEVY_ALPHA, scaler, cond_fn=None)

    q1      = np.percentile(uncon, 1, axis=0)
    q1_norm = _to_norm(scaler, q1[VOL_IDX], VOL_IDX)
    qmin = np.min(uncon[:, VOL_IDX])
    qmin_norm = _to_norm(scaler, qmin, VOL_IDX)


    def cf(x0_hat):
        return torch.relu(x0_hat[:, VOL_IDX] - q1_norm) + torch.relu(qmin_norm - x0_hat[:, VOL_IDX])
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

    q1   = np.percentile(uncon, 1,  axis=0)
    q99  = np.percentile(uncon, 99, axis=0)
    qmin = uncon.min(axis=0)
    qmax = uncon.max(axis=0)

    failures = []

    for i, name in enumerate(FACTOR_NAMES):
        qmin_norm = _to_norm(scaler, qmin[i], i)
        q1_norm   = _to_norm(scaler, q1[i],   i)
        q99_norm  = _to_norm(scaler, q99[i],  i)
        qmax_norm = _to_norm(scaler, qmax[i], i)

        with patch("factor_diffusion_sample.NUM_GENERATE", N_SAMPLES):
            lo, _, _ = generate(model, gammas, bargammas, sigmas, barsigmas,
                                LEVY_ALPHA, scaler,
                                cond_fn=_make_interval_cond(qmin_norm, q1_norm, i),
                                guidance_scale=5.0)
            hi, _, _ = generate(model, gammas, bargammas, sigmas, barsigmas,
                                LEVY_ALPHA, scaler,
                                cond_fn=_make_interval_cond(q99_norm, qmax_norm, i),
                                guidance_scale=5.0)

        lo_mean  = lo[:, i].mean()
        hi_mean  = hi[:, i].mean()
        pass_lo  = lo_mean < q1[i]
        pass_hi  = hi_mean > q99[i]

        print(f"  {name:<18} [{qmin[i]:>8.4f}, {q1[i]:>8.4f}]  [{q99[i]:>8.4f}, {qmax[i]:>8.4f}]"
              f"  lo={lo_mean:>8.4f}  hi={hi_mean:>8.4f}")

        if not pass_lo:
            failures.append(f"{name} lo: mean={lo_mean:.4f} not < q1={q1[i]:.4f}")
        if not pass_hi:
            failures.append(f"{name} hi: mean={hi_mean:.4f} not > q99={q99[i]:.4f}")

    assert not failures, "\nFailed conditions:\n" + "\n".join(failures)


@pytest.mark.parametrize("plo,phi", [(1, 25), (25, 75), (75, 99)])
def test_double(artifacts, plo, phi):
    """Interval guidance test: check that guided samples' mean falls within [q_plo, q_phi]."""
    model, gammas, bargammas, sigmas, barsigmas, scaler = artifacts

    with patch("factor_diffusion_sample.NUM_GENERATE", N_SAMPLES):
        uncon, _, _ = generate(model, gammas, bargammas, sigmas, barsigmas,
                               LEVY_ALPHA, scaler, cond_fn=None)

    lo_bound = np.percentile(uncon, plo, axis=0)
    hi_bound = np.percentile(uncon, phi, axis=0)

    failures = []

    print(f"\n  Interval [{plo}th, {phi}th] percentile")
    print(f"  {'factor':<18} {'lo_bound':>10} {'hi_bound':>10} {'guided_mean':>12} {'pass':>6}")

    for i, name in enumerate(FACTOR_NAMES):
        lo_norm = _to_norm(scaler, lo_bound[i], i)
        hi_norm = _to_norm(scaler, hi_bound[i], i)

        with patch("factor_diffusion_sample.NUM_GENERATE", N_SAMPLES):
            guided, _, _ = generate(model, gammas, bargammas, sigmas, barsigmas,
                                    LEVY_ALPHA, scaler,
                                    cond_fn=_make_interval_cond(lo_norm, hi_norm, i),
                                    guidance_scale=5.0)

        mean    = guided[:, i].mean()
        pass_lo = mean >= lo_bound[i]
        pass_hi = mean <= hi_bound[i]
        ok      = pass_lo and pass_hi

        print(f"  {name:<18} {lo_bound[i]:>10.4f} {hi_bound[i]:>10.4f} {mean:>12.4f}")

        if not pass_lo:
            failures.append(f"{name} [{plo},{phi}]: mean={mean:.4f} not > lo={lo_bound[i]:.4f}")
        if not pass_hi:
            failures.append(f"{name} [{plo},{phi}]: mean={mean:.4f} not < hi={hi_bound[i]:.4f}")

    assert not failures, "\nFailed conditions:\n" + "\n".join(failures)
