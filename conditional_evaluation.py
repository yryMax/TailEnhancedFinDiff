"""
Conditional generation evaluation: volatility < 1% quantile
Compares rejection sampling (exact) vs guidance (approximate) on 64 samples.
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from factor_diffusion_train import FactorDenoiser
from factor_diffusion_levy import levy_noise_schedule
from factor_diffusion_sample import (
    CHECKPOINT, DEVICE, FACTOR_NAMES, FACTOR_DIM,
    LEVY_ALPHA, NUM_TIMESTEPS, generate, generate_rejection,
)

N_REF      = 512    # uncon samples to estimate q1
N_COND     = 512     # conditional samples for each method
VOL_IDX    = FACTOR_NAMES.index("volatility")
GUIDANCE_S = 0.5
SAVE_PATH  = "conditional_evaluation.png"

# ── load model ────────────────────────────────────────────────────────────────
ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
model = FactorDenoiser(**ckpt["model_kwargs"]).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
gammas, bargammas, sigmas, barsigmas = levy_noise_schedule(LEVY_ALPHA, NUM_TIMESTEPS)
scaler = ckpt["scaler"]
args   = (model, gammas, bargammas, sigmas, barsigmas, LEVY_ALPHA, scaler)

# ── unconditional reference → q1 ─────────────────────────────────────────────
print(f"Generating {N_REF} unconditional reference samples...")
uncon = generate(*args, num_samples=N_REF)
q1    = np.percentile(uncon[:, VOL_IDX], 1)
print(f"  volatility q1 = {q1:.6f}")

# normalized threshold for cond_fn
dummy = np.zeros((1, FACTOR_DIM))
dummy[0, VOL_IDX] = q1
q1_norm = float(scaler.transform(dummy)[0, VOL_IDX])

def cond_fn(x0_hat):
    """Penalise vol > q1_norm  →  loss=0 iff vol ≤ q1  (vol < 1% quantile)."""
    return torch.relu(x0_hat[:, VOL_IDX] - q1_norm).pow(2).mean()

# ── rejection sampling ────────────────────────────────────────────────────────
print(f"\nRejection sampling {N_COND} samples (hard, expected rate ~1%)...")
rej = generate_rejection(*args, cond_fn=cond_fn,
                         num_samples=N_COND, hard=False, guidance_scale=GUIDANCE_S, max_batches=1000)

# ── guidance ──────────────────────────────────────────────────────────────────
print(f"\nGuidance sampling {N_COND} samples (scale={GUIDANCE_S})...")
from factor_diffusion_sample import generate as gen_guided
gui = gen_guided(*args, cond_fn=cond_fn,
                 guidance_scale=GUIDANCE_S, num_samples=N_COND)

# ── summary stats ─────────────────────────────────────────────────────────────
for label, s in [("uncon", uncon[:N_COND]), ("rejection", rej), ("guidance", gui)]:
    v = s[:, VOL_IDX]
    pct_ok = (v <= q1).mean() * 100
    print(f"  {label:<12}  vol mean={v.mean():+.5f}  std={v.std():.5f}  "
          f"pct satisfying (≤q1)={pct_ok:.1f}%")

# ── plot ──────────────────────────────────────────────────────────────────────
from scipy.stats import gaussian_kde

fig = plt.figure(figsize=(14, 11))
gs  = gridspec.GridSpec(2, 2, height_ratios=[2, 1.2], hspace=0.5, wspace=0.35)

COLORS = {"uncon": "#888888", "rejection": "#2196F3", "guidance": "#FF5722"}
ALPHA  = 0.55
BINS   = 40

datasets = {
    "uncon (ref)":  uncon[:, VOL_IDX],
    "rejection":    rej[:, VOL_IDX],
    "guidance":     gui[:, VOL_IDX],
}
color_map = {
    "uncon (ref)":  COLORS["uncon"],
    "rejection":    COLORS["rejection"],
    "guidance":     COLORS["guidance"],
}

def _plot_vol_panel(ax, xlim, title_suffix):
    for label, data in datasets.items():
        clipped = data[(data >= xlim[0]) & (data <= xlim[1])]
        if len(clipped) < 2:
            continue
        ax.hist(clipped, bins=BINS, density=True, alpha=ALPHA,
                color=color_map[label], label=label, edgecolor="white", linewidth=0.4)
        kde = gaussian_kde(data, bw_method=0.3)
        x_grid = np.linspace(xlim[0], xlim[1], 400)
        ax.plot(x_grid, kde(x_grid), color=color_map[label], linewidth=2)

    ax.axvline(q1, color="black", linestyle="--", linewidth=1.5,
               label=f"q1 = {q1:.4f}")
    ax.set_xlabel("volatility (original space)")
    ax.set_ylabel("density")
    ax.set_xlim(xlim)
    ax.set_title(title_suffix)
    ax.legend(fontsize=8, framealpha=0.9)
    for i, (label, data) in enumerate(datasets.items()):
        pct     = (data <= q1).mean() * 100
        visible = ((data >= xlim[0]) & (data <= xlim[1])).sum()
        ax.text(0.02, 0.97 - i * 0.10,
                f"{label}: {pct:.0f}% ≤ q1  ({visible}/{len(data)} visible)",
                transform=ax.transAxes, fontsize=8,
                color=color_map[label], va="top")

# left: zoomed to uncon/rejection range
sigma_ref = uncon[:, VOL_IDX].std()
zoom_lim  = (q1 - 4 * sigma_ref, q1 + 4 * sigma_ref)
ax_zoom   = fig.add_subplot(gs[0, 0])
_plot_vol_panel(ax_zoom, zoom_lim,
                f"Zoomed (uncon/rejection range)\ncond = vol < q1, n={N_COND}")

# right: full range, all three distributions unclipped
all_data  = np.concatenate(list(datasets.values()))
full_lim  = (all_data.min() - abs(all_data.min()) * 0.05,
             all_data.max() + abs(all_data.max()) * 0.05)
ax_full   = fig.add_subplot(gs[0, 1])
_plot_vol_panel(ax_full, full_lim,
                f"Full range (all samples visible)\nguidance scale={GUIDANCE_S}")

fig.suptitle(
    f"Conditional generation: volatility < q1 = {q1:.5f}\n"
    f"rejection (hard=False, exact Boltzmann) vs guidance (approx) | {N_COND} samples",
    fontsize=11, y=1.01
)

# --- bottom: all 7 factors, mean comparison ---
ax_all = fig.add_subplot(gs[1, :])

x     = np.arange(FACTOR_DIM)
width = 0.26

for offset, (label, s, color) in enumerate([
    ("uncon",     uncon[:N_COND], COLORS["uncon"]),
    ("rejection", rej,            COLORS["rejection"]),
    ("guidance",  gui,            COLORS["guidance"]),
]):
    means = s.mean(axis=0)
    stds  = s.std(axis=0)
    bars  = ax_all.bar(x + (offset - 1) * width, means, width,
                       yerr=stds, label=label, color=color,
                       alpha=0.75, capsize=3, error_kw={"linewidth": 1})

ax_all.axhline(0, color="black", linewidth=0.7, linestyle=":")
ax_all.set_xticks(x)
ax_all.set_xticklabels(FACTOR_NAMES, fontsize=9)
ax_all.set_ylabel("mean ± std")
ax_all.set_title("All factor means (conditioning on volatility only — spillover check)")
ax_all.legend(fontsize=9, framealpha=0.9)

plt.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
print(f"\nSaved → {SAVE_PATH}")
