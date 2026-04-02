import numpy as np
import matplotlib.pyplot as plt

# ── Data ─────────────────────────────────────────────────────────────────────
raw      = np.load("data/stocknet_81.npy")                # [1257, 81]
sb       = np.load("samples/SB2048_T22.npy")              # [2048, 81, 22]
macrovae = np.load("samples/sampled_stocks_2048_v4.npy")  # [2048, 81, 22]

N = raw.shape[1]
D = sb.shape[2]

# ── Pick 3 stocks by volatility from raw training series ─────────────────────
vols      = raw.std(axis=0)
sorted_idx = np.argsort(vols)
stocks = [
    (sorted_idx[-1],   "Highest Vol",  f"σ={vols[sorted_idx[-1]]:.4f}"),
    (sorted_idx[N//2], "Median Vol",   f"σ={vols[sorted_idx[N//2]]:.4f}"),
    (sorted_idx[0],    "Lowest Vol",   f"σ={vols[sorted_idx[0]]:.4f}"),
]

C_SB  = "#2e75b6"
C_VAE = "#c0392b"

def cum(X, stock):
    paths = np.cumsum(X[:, stock, :], axis=1)
    return np.hstack([np.zeros((paths.shape[0], 1)), paths])

days = np.arange(D + 1)
rng  = np.random.default_rng(42)

# ── Figure: 3 rows × 2 cols ───────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharey=True)

for row, (stock_idx, vol_label, sig_str) in enumerate(stocks):
    cum_sb  = cum(sb,       stock_idx)
    cum_vae = cum(macrovae, stock_idx)

    idx_sb  = rng.choice(len(cum_sb),  size=300, replace=False)
    idx_vae = rng.choice(len(cum_vae), size=300, replace=False)

    panels = [
        (axes[row, 0], cum_sb[idx_sb],   C_SB,  "Stationary Bootstrap"),
        (axes[row, 1], cum_vae[idx_vae], C_VAE, "MacroVAE"),
    ]

    for ax, paths, color, name in panels:
        for path in paths:
            ax.plot(days, path, color=color, alpha=0.15, linewidth=0.6, zorder=2)

        ax.axhline(0, color="black", linewidth=0.6, ls="--", zorder=1)
        ax.set_xlim(0, D)
        ax.set_ylim(-0.1, 0.1)

        if row == 0:
            ax.set_title(name, fontsize=12, fontweight="bold")
        if row == 2:
            ax.set_xlabel("Day", fontsize=11)

    axes[row, 0].set_ylabel(f"{vol_label}\n({sig_str})\nCumul. Return", fontsize=10)

fig.tight_layout()
fig.savefig("fig3_temporal_paths.png", dpi=200, bbox_inches="tight")
plt.close("all")
print("Saved fig3_temporal_paths.png")
