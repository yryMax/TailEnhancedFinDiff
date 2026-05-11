import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

FACTOR_NAMES = ['market', 'growth', 'momentum', 'quality', 'size', 'value', 'volatility']
SHORT = [n[:3] for n in FACTOR_NAMES]

EXPERIMENTS = [
    ('synth',       '../model/ablation/factors_synth.csv',
                    '../model/ablation/samples/factor_DLPM_synth_4096.npy'),
    ('amplified',   '../model/ablation/factors_amplified_synth.csv',
                    '../model/ablation/samples/factor_DLPM_amplified_4096.npy'),
    ('anisotropic', '../model/ablation/factors_anisotropic_synth.csv',
                    '../model/ablation/samples/factor_DLPM_anisotropic_4096.npy'),
]


def from_csv(path):
    df = pd.read_csv(path, index_col=0)[FACTOR_NAMES].dropna()
    return StandardScaler().fit_transform(df.values.astype(np.float64))


def from_npy(path):
    X = np.load(path).astype(np.float64)
    return StandardScaler().fit_transform(X)


datasets = []
for name, csv_path, npy_path in EXPERIMENTS:
    datasets.append((f'{name} (CSV)',    from_csv(csv_path)))
    datasets.append((f'{name} (Sample)', from_npy(npy_path)))

diag_mask = np.eye(len(FACTOR_NAMES), dtype=bool)
triu_mask = np.triu(np.ones(len(FACTOR_NAMES), dtype=bool), k=1)


def corr_similarity(C_real, C_fake, label_real, label_fake):
    diff = C_real - C_fake
    frob = np.linalg.norm(diff)
    mae  = np.abs(diff[triu_mask]).mean()
    r    = np.corrcoef(C_real[triu_mask], C_fake[triu_mask])[0, 1]
    print(f"\n{label_real}  vs  {label_fake}")
    print(f"  Frobenius norm of diff : {frob:.4f} ")
    print(f"  MAE of off-diag entries: {mae:.4f} ")
    print(f"  Pearson r of upper tri : {r:.4f} ")


corr_matrices = {title: np.corrcoef(X.T) for title, X in datasets}

print("=" * 55)
print("  Correlation Matrix Similarity (CSV vs Sample per ablation)")
print("=" * 55)
for name, _, _ in EXPERIMENTS:
    corr_similarity(corr_matrices[f'{name} (CSV)'],
                    corr_matrices[f'{name} (Sample)'],
                    f'{name} (CSV)', f'{name} (Sample)')
print("=" * 55)

n_exp = len(EXPERIMENTS)
fig, axes = plt.subplots(n_exp, 2, figsize=(13, 5 * n_exp))

for ax, (title, X) in zip(axes.flat, datasets):
    C = np.corrcoef(X.T)
    C[diag_mask] = np.nan
    sns.heatmap(C, ax=ax, annot=True, fmt='.2f', cmap='RdBu_r',
                vmin=-1, vmax=1, center=0,
                xticklabels=SHORT, yticklabels=SHORT,
                square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.8})
    ax.set_title(title, fontsize=11)

plt.suptitle('Correlation Matrices — Ablation: CSV (truth) vs DLPM Samples',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

fig.savefig("corr_matrix_ablation.png")

# ── Spectral analysis ──
x = np.arange(1, len(FACTOR_NAMES) + 1)
fig2, axes2 = plt.subplots(1, n_exp, figsize=(6 * n_exp, 4), sharey=True)
if n_exp == 1:
    axes2 = [axes2]

for ax, (name, _, _) in zip(axes2, EXPERIMENTS):
    C_real = corr_matrices[f'{name} (CSV)']
    C_fake = corr_matrices[f'{name} (Sample)']
    eig_real = np.sort(np.linalg.eigvalsh(C_real))[::-1]
    eig_fake = np.sort(np.linalg.eigvalsh(C_fake))[::-1]
    ax.bar(x - 0.2, eig_real, width=0.4, label='CSV',    color='steelblue')
    ax.bar(x + 0.2, eig_fake, width=0.4, label='Sample', color='tomato')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xlabel('Eigenvalue index')
    ax.set_title(f'{name}', fontweight='bold')
    ax.legend()

axes2[0].set_ylabel('Eigenvalue')
plt.suptitle('Spectral Analysis of Correlation Matrices — Ablation',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

fig2.savefig("Spectral_Analysis_ablation.png")
