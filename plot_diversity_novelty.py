import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, cdist

real = np.load("data/stocknet_81.npy")
fdm = np.load("samples/factorDM2048.npy")
sb = np.load("samples/SB2048.npy")

C_FDM = "#E8792B"
C_SB = "#3274A1"

fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5), width_ratios=[1, 1, 1.15])

order_fdm = np.argsort(fdm.mean(axis=1))[::-1]
order_sb = np.argsort(sb.mean(axis=1))[::-1]
vlim = np.percentile(np.concatenate([fdm.ravel(), sb.ravel()]), [2, 98])
vmin, vmax = -max(abs(vlim[0]), abs(vlim[1])), max(abs(vlim[0]), abs(vlim[1]))

im0 = axes1[0].imshow(fdm[order_fdm], aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
axes1[0].set_xlabel("Stock Index", fontsize=12)
axes1[0].set_ylabel("Sample (sorted by mean return)", fontsize=12)
axes1[0].set_title("FactorDM", fontsize=13)

im1 = axes1[1].imshow(sb[order_sb], aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
axes1[1].set_xlabel("Stock Index", fontsize=12)
axes1[1].set_title("Stationary Bootstrap", fontsize=13)

fig1.colorbar(im1, ax=axes1[1], fraction=0.046, pad=0.04, label="Return")

pca = PCA(n_components=2)
pca.fit(real)
fdm_pc = pca.transform(fdm)
sb_pc = pca.transform(sb)

axes1[2].scatter(sb_pc[:, 0], sb_pc[:, 1], c=C_SB, marker="o", s=14, alpha=0.40, label="Stationary Bootstrap", zorder=2)
axes1[2].scatter(fdm_pc[:, 0], fdm_pc[:, 1], c=C_FDM, marker="o", s=14, alpha=0.40, label="FactorDM", zorder=3)
axes1[2].set_xlabel(f"PC 1  ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
axes1[2].set_ylabel(f"PC 2  ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
axes1[2].set_title("PCA Projection", fontsize=13)
axes1[2].legend(fontsize=10, markerscale=2)
fig1.tight_layout()
fig1.savefig("fig1_data_and_pca.png", dpi=200, bbox_inches="tight")

pw_fdm = pdist(fdm)
pw_sb = pdist(sb)

dist_fdm_to_real = cdist(fdm, real).min(axis=1)
dist_sb_to_real = cdist(sb, real).min(axis=1)
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 4.5))

bins_pw = np.linspace(0, max(pw_fdm.max(), pw_sb.max()) * 0.85, 80)
ax2a.hist(pw_fdm, bins=bins_pw, density=True, alpha=0.55, color=C_FDM, label="FactorDM")
ax2a.hist(pw_sb, bins=bins_pw, density=True, alpha=0.55, color=C_SB, label="Stationary Bootstrap")
ax2a.axvline(pw_fdm.mean(), color=C_FDM, ls="--", lw=1.8)
ax2a.axvline(pw_sb.mean(), color=C_SB, ls="--", lw=1.8)
ax2a.set_xlabel("Pairwise Euclidean Distance", fontsize=12)
ax2a.set_ylabel("Density", fontsize=12)
ax2a.set_title("Pairwise Distance Histogram", fontsize=13)
ax2a.legend(fontsize=10)

dist_fdm_to_real = cdist(fdm, real).min(axis=1)
dist_sb_to_real = cdist(sb, real).min(axis=1)

bins_nov = np.linspace(0, dist_fdm_to_real.max() * 1.05, 60)
ax2b.hist(dist_fdm_to_real, bins=bins_nov, density=True, alpha=0.6, color=C_FDM, label=f"FactorDM  (mean={dist_fdm_to_real.mean():.4f})")
ax2b.hist(dist_sb_to_real, bins=bins_nov, density=True, alpha=0.6, color=C_SB, label=f"SB  (mean={dist_sb_to_real.mean():.4f})")
ax2b.axvline(dist_fdm_to_real.mean(), color=C_FDM, ls="--", lw=1.5)
ax2b.set_xlabel("Nearest-Neighbour Distance to Real Data", fontsize=12)
ax2b.set_title("Nearest-Neighbour Distance Histogram", fontsize=13)
ax2b.legend(fontsize=10)

fig2.tight_layout()
fig2.savefig("fig2_diversity_novelty.png", dpi=200, bbox_inches="tight")

plt.close("all")
