import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from diffusers import DDPMScheduler
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.embeddings import Timesteps, TimestepEmbedding

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QuantileNormalizer:
    """Drop-in replacement for sklearn's QuantileTransformer(output_distribution='normal').
    Only uses numpy + torch — no sklearn required.
    """

    def __init__(self, n_quantiles: int = 1000):
        self.n_quantiles = n_quantiles

    def fit(self, X: np.ndarray):
        n = min(len(X), self.n_quantiles)
        self.refs_ = np.linspace(0.0, 1.0, n, dtype=np.float32)
        # quantiles_: shape (n_quantiles, n_features)
        self.quantiles_ = np.quantile(X, self.refs_, axis=0).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        U = np.empty(X.shape, dtype=np.float32)
        for j in range(X.shape[1]):
            U[:, j] = np.interp(X[:, j], self.quantiles_[:, j], self.refs_)
        U = np.clip(U, 1e-7, 1 - 1e-7)
        # Φ⁻¹(u) = √2 · erfinv(2u − 1)
        return (torch.sqrt(torch.tensor(2.0)) *
                torch.erfinv(2.0 * torch.from_numpy(U) - 1.0)).numpy()

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        Z_t = torch.from_numpy(Z.astype(np.float32))
        # Φ(z) = 0.5 · (1 + erf(z / √2))
        U = (0.5 * (1.0 + torch.erf(Z_t / torch.sqrt(torch.tensor(2.0))))).numpy()
        U = np.clip(U, 0.0, 1.0)
        out = np.empty(Z.shape, dtype=np.float32)
        for j in range(Z.shape[1]):
            out[:, j] = np.interp(U[:, j], self.refs_, self.quantiles_[:, j])
        return out

FACTOR_NAMES  = ["growth", "momentum", "quality", "size", "value", "volatility"]
FACTOR_DIM    = 7
EPOCHS        = 120
BATCH_SIZE    = 128
LR            = 2e-4
NUM_TIMESTEPS = 200

def load_data(csv_path):
    X = pd.read_csv(csv_path, index_col=0)[FACTOR_NAMES].dropna().values.astype(np.float32)
    scaler = QuantileNormalizer(n_quantiles=min(len(X), 1000)).fit(X)
    return scaler.transform(X), scaler


# ── Model ──────────────────────────────────────────────────────────────────────
class FactorDenoiser(nn.Module):
    def __init__(self, dim=64, n_heads=4, cond_dim=128, num_blocks=1):
        super().__init__()
        self.kwargs = dict(dim=dim, n_heads=n_heads, cond_dim=cond_dim, num_blocks=num_blocks)
        self.t_sin   = Timesteps(cond_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.t_embed = TimestepEmbedding(in_channels=cond_dim, time_embed_dim=cond_dim)
        self.in_proj = nn.Linear(1, dim)
        self.blocks  = nn.ModuleList([
            BasicTransformerBlock(
                dim=dim, num_attention_heads=n_heads, attention_head_dim=dim // n_heads,
                norm_type="ada_norm_continuous",
                ada_norm_continous_conditioning_embedding_dim=cond_dim,
            )
            for _ in range(num_blocks)
        ])
        self.out_proj = nn.Linear(dim, 1)

    def forward(self, x, t):
        cond = self.t_embed(self.t_sin(t))
        h = self.in_proj(x.unsqueeze(-1))
        for blk in self.blocks:
            h = blk(h, added_cond_kwargs={"pooled_text_emb": cond})
        return self.out_proj(h).squeeze(-1)


# ── Train ──────────────────────────────────────────────────────────────────────
def train(model, loader, scheduler, optimizer, scaler):
    import matplotlib.pyplot as plt
    os.makedirs("checkpoints", exist_ok=True)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    losses = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for (x,) in loader:
            x     = x.to(DEVICE)
            t     = torch.randint(0, NUM_TIMESTEPS, (x.size(0),), device=DEVICE)
            noise = torch.randn_like(x)
            loss  = nn.functional.mse_loss(model(scheduler.add_noise(x, noise, t), t), noise)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
        lr_sched.step()
        losses.append(epoch_loss / len(loader.dataset))
        print(f"Epoch [{epoch:4d}/{EPOCHS}]  loss={losses[-1]:.6f}")

    torch.save({
        "model_state":   model.state_dict(),
        "model_kwargs":  model.kwargs,
        "scaler":        scaler,
        "epoch":         epoch,
        "num_timesteps": NUM_TIMESTEPS,
        "losses":        losses,
    }, f"checkpoints/factor_ddpm_quantileTransformer_ep{epoch:04d}.pt")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, EPOCHS + 1), losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("assets/factor_ddpm_loss.png", dpi=150)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X, scaler = load_data("data/factor_returns_regression.csv")
    print(f"Dataset: {X.shape}")

    loader    = DataLoader(TensorDataset(torch.tensor(X)), batch_size=BATCH_SIZE, shuffle=True)
    model     = FactorDenoiser().to(DEVICE)
    scheduler = DDPMScheduler(num_train_timesteps=NUM_TIMESTEPS,
                              beta_schedule="squaredcos_cap_v2", prediction_type="epsilon")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    train(model, loader, scheduler, optimizer, scaler)
