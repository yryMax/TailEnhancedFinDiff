import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

FACTOR_NAMES = ['market', 'growth', 'momentum', 'quality', 'size', 'value', 'volatility']
REAL_CSV     = 'model/regression/factors.csv'
FAKE_CSV     = 'model/regression/fake_factors.csv'
REAL_SAMPLE  = 'model/regression/samples/factor_4096.npy'
FAKE_SAMPLE  = 'model/regression/samples/factor_fake_4096.npy'

SHORT = [n[:3] for n in FACTOR_NAMES]


def from_csv(path):
    df = pd.read_csv(path, index_col=0)[FACTOR_NAMES].dropna()
    return StandardScaler().fit_transform(df.values.astype(np.float64))


def from_npy(path):
    X = np.load(path).astype(np.float64)
    return StandardScaler().fit_transform(X)


datasets = [
    ('Real (CSV)',    from_csv(REAL_CSV)),
    ('Fake (CSV)',    from_csv(FAKE_CSV)),
    ('Real (Sample)', from_npy(REAL_SAMPLE)),
    ('Fake (Sample)', from_npy(FAKE_SAMPLE)),
]

diag_mask = np.eye(len(FACTOR_NAMES), dtype=bool)
triu_mask = np.triu(np.ones(len(FACTOR_NAMES), dtype=bool), k=1)  # upper triangle, no diag


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
print("  Correlation Matrix Similarity")
print("=" * 55)
corr_similarity(corr_matrices['Real (CSV)'],  corr_matrices['Real (Sample)'],
                'Real (CSV)',  'Real (Sample)')
corr_similarity(corr_matrices['Fake (CSV)'],  corr_matrices['Fake (Sample)'],
                'Fake (CSV)',  'Fake (Sample)')
print("=" * 55)

fig, axes = plt.subplots(2, 2, figsize=(13, 11))

for ax, (title, X) in zip(axes.flat, datasets):
    C = np.corrcoef(X.T)
    C[diag_mask] = np.nan   # hide diagonal (always 1, no info)
    sns.heatmap(C, ax=ax, annot=True, fmt='.2f', cmap='RdBu_r',
                vmin=-1, vmax=1, center=0,
                xticklabels=SHORT, yticklabels=SHORT,
                square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.8})
    ax.set_title(title, fontsize=11)

plt.suptitle('Correlation Matrices — Real vs Generated Factor Returns',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# save the fig
fig.savefig("corr_matrix.png")

# ── Spectral analysis: eigenvalues of Real vs Fake correlation matrices ──
C_real = corr_matrices['Real (CSV)']
C_fake = corr_matrices['Fake (CSV)']

eigvals_real = np.sort(np.linalg.eigvalsh(C_real))[::-1]
eigvals_fake = np.sort(np.linalg.eigvalsh(C_fake))[::-1]

x = np.arange(1, len(FACTOR_NAMES) + 1)

fig2, ax = plt.subplots(figsize=(7, 4))
ax.bar(x - 0.2, eigvals_real, width=0.4, label='Real (CSV)', color='steelblue')
ax.bar(x + 0.2, eigvals_fake, width=0.4, label='Fake (CSV)', color='tomato')
ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
ax.set_xticks(x)
ax.set_xlabel('Eigenvalue index')
ax.set_ylabel('Eigenvalue')
ax.set_title('Spectral Analysis of Correlation Matrices', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.show()

fig2.savefig("Spectral_Analysis.png")