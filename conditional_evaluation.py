import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from factor_diffusion_train import FactorDenoiser, FACTOR_NAMES
from factor_diffusion_levy import levy_noise_schedule
from factor_diffusion_sample import generate, generate_rejection
import yaml
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



with open("cfg.yaml") as f:
    _all_cfg = yaml.safe_load(f)
    _tcfg    = _all_cfg["train"]
    _scfg    = _all_cfg["sample"]

NUM_TIMESTEPS = _tcfg["num_timesteps"]
LEVY_ALPHA    = _tcfg["levy_alpha"]
PREFIX        = _tcfg["prefix"]
BATCH_SIZE    = _tcfg["batch_size"]
FACTOR_NAMES  = _tcfg["factor_names"]
NUM_GENERATE  = _scfg["num_generate"]
FACTOR_DIM    = len(_tcfg["factor_names"])
CHECKPOINT    = f"{PREFIX}/checkpoints/factor_DLPM_ep0200.pt"
OUT_PATH      = f"{PREFIX}/samples/factor_{NUM_GENERATE}.npy"

N_COND     = 128
VOL_IDX    = FACTOR_NAMES.index("volatility")
GUIDANCE_S = 5.0

ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
model = FactorDenoiser(**ckpt["model_kwargs"]).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
gammas, bargammas, sigmas, barsigmas = levy_noise_schedule(LEVY_ALPHA, NUM_TIMESTEPS)
scaler = ckpt["scaler"]
args   = (model, gammas, bargammas, sigmas, barsigmas, LEVY_ALPHA, scaler)

uncon, _, _  = generate(*args, num_samples=64)
q1    = np.percentile(uncon[:, VOL_IDX], 1)
lower_val = uncon[:, VOL_IDX].min()
print(f"  volatility q1 = {q1:.6f}")

# normalized threshold for cond_fn
dummy = np.zeros((1, FACTOR_DIM))
dummy[0, VOL_IDX] = q1
q1_norm = float(scaler.transform(dummy)[0, VOL_IDX])

def cond_fn(x0_hat):
    """Push vol below q1_norm. Returns shape (n,)."""
    return torch.relu(x0_hat[:, VOL_IDX] - q1_norm)


print(f"\nRejection sampling {N_COND} samples (hard, expected rate ~1%)...")
rej = generate_rejection(*args, cond_fn=cond_fn,
                         num_samples=N_COND, hard=False, guidance_scale=GUIDANCE_S, max_batches=1000)

print(f"\nGuidance sampling {N_COND} samples (scale={GUIDANCE_S})...")

gui, _, _ = generate(*args, cond_fn=cond_fn,
                 guidance_scale=GUIDANCE_S, num_samples=N_COND)

for label, s in [("uncon", uncon[:N_COND]), ("rejection", rej), ("guidance", gui)]:
    v = s[:, VOL_IDX]
    pct_ok = (v <= q1).mean() * 100
    print(f"  {label:<12}  vol mean={v.mean():+.5f}  std={v.std():.5f}  "
          f"pct satisfying (≤q1)={pct_ok:.1f}%")

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

def _plot_vol_panel(ax, xlim, title_suffix, datasets, color_map, fidx=None):
    for label, data, color in datasets:
        col = data[:, fidx] if (data.ndim == 2 and fidx is not None) else data
        clipped = col[(col >= xlim[0]) & (col <= xlim[1])]
        if len(clipped) < 2:
            continue
        ax.hist(clipped, bins=BINS, density=True, alpha=ALPHA,
                color=color, label=label, edgecolor="white", linewidth=0.4)
        kde = gaussian_kde(col, bw_method=0.3)
        x_grid = np.linspace(xlim[0], xlim[1], 400)
        ax.plot(x_grid, kde(x_grid), color=color, linewidth=2)

    ax.axvline(q1, color="black", linestyle="--", linewidth=1.5,
               label=f"q1 = {q1:.4f}")
    ax.set_xlabel("volatility (original space)")
    ax.set_ylabel("density")
    ax.set_xlim(xlim)
    ax.set_title(title_suffix)
    ax.legend(fontsize=8, framealpha=0.9)
    for i, (label, data, color) in enumerate(datasets):
        col     = data[:, fidx] if (data.ndim == 2 and fidx is not None) else data
        pct     = (col <= q1).mean() * 100
        visible = ((col >= xlim[0]) & (col <= xlim[1])).sum()
        ax.text(0.02, 0.97 - i * 0.10,
                f"{label}: {pct:.0f}% ≤ q1  ({visible}/{len(col)} visible)",
                transform=ax.transAxes, fontsize=8,
                color=color, va="top")

SUPTITLE = (
    f"Conditional generation: volatility < q1 = {q1:.5f}\n"
    f"rejection (hard=False, exact Boltzmann) vs guidance (approx) | {N_COND} samples"
)

# --- Figure 1: Zoomed vol distribution ---
sigma_ref = uncon[:, VOL_IDX].std()
zoom_lim  = (q1 - 4 * sigma_ref, q1 + 4 * sigma_ref)
fig1, ax_zoom = plt.subplots(figsize=(7, 5))
_plot_vol_panel(ax_zoom, zoom_lim,
                f"Zoomed (uncon/rejection range)\ncond = vol < q1, n={N_COND}",
                datasets, color_map)
fig1.suptitle(SUPTITLE, fontsize=10)
fig1.tight_layout()
plt.show()

# --- Figure 2: Full range vol distribution ---
all_data = np.concatenate(list(datasets.values()))
full_lim = (all_data.min() - abs(all_data.min()) * 0.05,
            all_data.max() + abs(all_data.max()) * 0.05)
fig2, ax_full = plt.subplots(figsize=(7, 5))
_plot_vol_panel(ax_full, full_lim,
                f"Full range (all samples visible)\nguidance scale={GUIDANCE_S}",
                datasets, color_map)
fig2.suptitle(SUPTITLE, fontsize=10)
fig2.tight_layout()
plt.show()

# --- Figure 3: All factor means ---
fig3, ax_all = plt.subplots(figsize=(9, 4))

x     = np.arange(FACTOR_DIM)
width = 0.26

for offset, (label, s, color) in enumerate([
    ("uncon",     uncon[:N_COND], COLORS["uncon"]),
    ("rejection", rej,            COLORS["rejection"]),
    ("guidance",  gui,            COLORS["guidance"]),
]):
    means = s.mean(axis=0)
    stds  = s.std(axis=0)
    ax_all.bar(x + (offset - 1) * width, means, width,
               yerr=stds, label=label, color=color,
               alpha=0.75, capsize=3, error_kw={"linewidth": 1})

ax_all.axhline(0, color="black", linewidth=0.7, linestyle=":")
ax_all.set_xticks(x)
ax_all.set_xticklabels(FACTOR_NAMES, fontsize=9)
ax_all.set_ylabel("mean ± std")
ax_all.set_title("All factor means (conditioning on volatility only — spillover check)")
ax_all.legend(fontsize=9, framealpha=0.9)
fig3.tight_layout()
plt.show()
