import numpy as np
import pytest
import torch
import matplotlib.pyplot as plt
from factor_diffusion_train import FactorDenoiser
import yaml
from factor_diffusion_sample import generate

with open("cfg.yaml") as f:
    _exp = yaml.safe_load(f)["experiment_name"]
PREFIX = f"model/{_exp}"
with open(f"{PREFIX}/cfg.yaml") as f:
    _cfg = yaml.safe_load(f)

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_TIMESTEPS = _cfg["num_timesteps"]
LEVY_ALPHA    = _cfg["levy_alpha"]
FACTOR_NAMES  = _cfg["factor_names"]
FACTOR_DIM    = len(FACTOR_NAMES)
CHECKPOINT    = f"{PREFIX}/checkpoints/{_cfg['ckpt_name']}.pt"

N_SAMPLES = 128
VOL_IDX   = FACTOR_NAMES.index("volatility")


def plot_var_grad(var_history, grad_history, title="DLPM reverse-process diagnostics"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=13)

    if var_history:
        ts_v   = [r[0] for r in var_history[1:]]
        mean_v = [r[1] for r in var_history[1:]]
        max_v  = [r[2] for r in var_history[1:]]

        ax = axes[0, 0]
        ax.plot(ts_v, mean_v, label="mean var")
        ax.set(title="Posterior variance  —  mean over (n, D)", xlabel="t", ylabel="var")
        ax.invert_xaxis(); ax.legend()

        ax = axes[0, 1]
        ax.plot(ts_v, max_v, color="tab:orange", label="max var")
        ax.set(title="Posterior variance  —  max over (n, D)", xlabel="t", ylabel="var")
        ax.invert_xaxis(); ax.legend()

    if grad_history:
        ts_g   = [r[0] for r in grad_history[1:]]
        mean_g = [r[1] for r in grad_history[1:]]
        max_g  = [r[2] for r in grad_history[1:]]

        ax = axes[1, 0]
        ax.plot(ts_g, mean_g, color="tab:green", label="mean |grad|")
        ax.set(title="Guidance gradient  —  mean over (n, D)", xlabel="t", ylabel="grad")
        ax.invert_xaxis(); ax.legend()

        ax = axes[1, 1]
        ax.plot(ts_g, max_g, color="tab:red", label="max |grad|")
        ax.set(title="Guidance gradient  —  max over (n, D)", xlabel="t", ylabel="grad")
        ax.invert_xaxis(); ax.legend()

    plt.tight_layout()
    plt.show()
    return fig


@pytest.fixture(scope="module")
def artifacts():
    ckpt  = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model = FactorDenoiser(**ckpt["model_kwargs"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt["scaler"]


def _to_norm(scaler, value, idx):
    dummy = np.zeros((1, FACTOR_DIM))
    dummy[0, idx] = value
    return float(scaler.transform(dummy)[0, idx])


def _make_interval_cond(lo_norm, hi_norm, idx):
    """Penalty for x0_hat[:, idx] outside [lo_norm, hi_norm]. Accepts ±inf bounds."""
    def cond_fn(x0_hat):
        v = x0_hat[:, idx]
        penalty = torch.zeros_like(v)
        if hi_norm is not None:
            penalty = penalty + torch.relu(v - hi_norm) ** 5
        if lo_norm is not None:
            penalty = penalty + torch.relu(lo_norm - v) ** 5
        return penalty
    return cond_fn


def test_demo(artifacts):
    model, scaler = artifacts

    uncon, _, _ = generate(model, scaler, LEVY_ALPHA, NUM_TIMESTEPS, num_samples=N_SAMPLES)

    q1        = np.percentile(uncon, 1, axis=0)
    q1_norm   = _to_norm(scaler, q1[VOL_IDX],   VOL_IDX)

    cf = _make_interval_cond(None, q1_norm, VOL_IDX)
    lo, var_history, grad_history = generate(
        model, scaler, LEVY_ALPHA, NUM_TIMESTEPS,
        cond_fn=cf, guidance_scale=5.0, num_samples=N_SAMPLES
    )

    print(f"Demo: vol q1={q1[VOL_IDX]:.4f}  lo mean={lo[:, VOL_IDX].mean():.4f}")
    plot_var_grad(var_history, grad_history,
                  title="DLPM reverse-process diagnostics (batch-0, guided)")


def test_conditional_single(artifacts):
    model, scaler = artifacts

    uncon, _, _ = generate(model, scaler, LEVY_ALPHA, NUM_TIMESTEPS, num_samples=N_SAMPLES)

    q5  = np.percentile(uncon, 5,  axis=0)
    q95 = np.percentile(uncon, 95, axis=0)
    q50 = np.percentile(uncon, 50, axis=0)

    failures = []
    for i, name in enumerate(FACTOR_NAMES):
        q5_norm  = _to_norm(scaler, q5[i],  i)
        q95_norm = _to_norm(scaler, q95[i], i)

        lo, _, _ = generate(model, scaler, LEVY_ALPHA, NUM_TIMESTEPS,
                            cond_fn=_make_interval_cond(None, q5_norm, i),
                            guidance_scale=5.0, num_samples=N_SAMPLES)
        hi, _, _ = generate(model, scaler, LEVY_ALPHA, NUM_TIMESTEPS,
                            cond_fn=_make_interval_cond(q95_norm, None, i),
                            guidance_scale=5.0, num_samples=N_SAMPLES)

        lo_mean = lo[:, i].mean()
        hi_mean = hi[:, i].mean()
        print(f"  {name:<18}  q5={q5[i]:>8.4f}  q95={q95[i]:>8.4f}"
              f"  lo_mean={lo_mean:>8.4f}  hi_mean={hi_mean:>8.4f}  median={q50[i]:>8.4f}")

        if not lo_mean < q50[i]:
            failures.append(f"{name} lo: mean={lo_mean:.4f} not < median={q50[i]:.4f}")
        if not hi_mean > q50[i]:
            failures.append(f"{name} hi: mean={hi_mean:.4f} not > median={q50[i]:.4f}")

    assert not failures, "\nFailed conditions:\n" + "\n".join(failures)


@pytest.mark.parametrize("plo,phi", [(1, 25), (25, 75), (75, 99)])
def test_double(artifacts, plo, phi):
    model, scaler = artifacts

    uncon, _, _ = generate(model, scaler, LEVY_ALPHA, NUM_TIMESTEPS, num_samples=N_SAMPLES)

    lo_bound = np.percentile(uncon, plo, axis=0)
    hi_bound = np.percentile(uncon, phi, axis=0)

    failures = []
    print(f"\n  Interval [{plo}th, {phi}th] percentile")
    for i, name in enumerate(FACTOR_NAMES):
        lo_norm = _to_norm(scaler, lo_bound[i], i)
        hi_norm = _to_norm(scaler, hi_bound[i], i)

        guided, _, _ = generate(model, scaler, LEVY_ALPHA, NUM_TIMESTEPS,
                                cond_fn=_make_interval_cond(lo_norm, hi_norm, i),
                                guidance_scale=5.0, num_samples=N_SAMPLES)

        mean    = guided[:, i].mean()
        pass_lo = mean >= lo_bound[i]
        pass_hi = mean <= hi_bound[i]
        print(f"  {name:<18} {lo_bound[i]:>10.4f} {hi_bound[i]:>10.4f} {mean:>12.4f}")

        if not pass_lo:
            failures.append(f"{name} [{plo},{phi}]: mean={mean:.4f} not > lo={lo_bound[i]:.4f}")
        if not pass_hi:
            failures.append(f"{name} [{plo},{phi}]: mean={mean:.4f} not < hi={hi_bound[i]:.4f}")

    assert not failures, "\nFailed conditions:\n" + "\n".join(failures)
