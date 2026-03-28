import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from sklearn.preprocessing import StandardScaler
from factor_diffusion_levy import levy_noise_schedule, sample_skewed_levy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IdentityScaler:
    def fit(self, X):
        return self
    def transform(self, X):
        return X
    def inverse_transform(self, X):
        return X


FACTOR_NAMES  = [ "market", "growth", "momentum", "quality", "size", "value", "volatility"]
EPOCHS        = 200
BATCH_SIZE    = 64
LR            = 1e-4
NUM_TIMESTEPS = 100
LEVY_ALPHA    = 1.9
PREFIX        = "model/regression"


def load_data(csv_path):
    X = pd.read_csv(csv_path, index_col=0)[FACTOR_NAMES].dropna().values.astype(np.float32)
    scaler = StandardScaler().fit(X)
    X_norm = scaler.transform(X)
    return X_norm, scaler


# ── Model ──────────────────────────────────────────────────────────────────────
class FactorDenoiser(nn.Module):
    def __init__(self, dim=64, n_heads=4, cond_dim=128, num_blocks=1):
        super().__init__()
        self.kwargs = dict(dim=dim, n_heads=n_heads, cond_dim=cond_dim, num_blocks=num_blocks)
        self.t_sin   = Timesteps(cond_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.t_embed = TimestepEmbedding(in_channels=cond_dim, time_embed_dim=cond_dim)
        self.in_proj = nn.Linear(1, dim)
        self.feature_embed = nn.Parameter(torch.randn(1, len(FACTOR_NAMES), dim) * 0.02)
        self.blocks = nn.ModuleList([
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
        h = h + self.feature_embed
        for blk in self.blocks:
            h = blk(h, added_cond_kwargs={"pooled_text_emb": cond})
        return self.out_proj(h).squeeze(-1)


# ── Train ──────────────────────────────────────────────────────────────────────
def train(model, loader, gammas, bargammas, sigmas, barsigmas, optimizer, scaler):
    """
    DLPM training with Tail-Weighted Huber Loss.
    """
    import matplotlib.pyplot as plt
    os.makedirs("checkpoints", exist_ok=True)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # move schedule to device once
    bargammas_d = bargammas.to(DEVICE)
    barsigmas_d = barsigmas.to(DEVICE)
    losses = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for (x,) in loader:
            x   = x.to(DEVICE)
            # t ∈ [1, T-1]
            t   = torch.randint(1, NUM_TIMESTEPS, (x.size(0),), device=DEVICE)

            bg  = bargammas_d[t].unsqueeze(-1)   # (B, 1)
            bs  = barsigmas_d[t].unsqueeze(-1)   # (B, 1)

            # Sample variance multiplier from skewed Lévy
            a_t     = sample_skewed_levy(LEVY_ALPHA, x.shape, DEVICE)  # (B, D)
            sigma_t = a_t * bs ** 2                                     # (B, D)

            # Forward noising
            z_t   = torch.randn_like(x)
            x_t   = bg * x + sigma_t.sqrt() * z_t

            # Target: normalized Lévy noise  (= sqrt(a_t) * z_t)
            eps_t = (x_t - bg * x) / bs

            # Predict noise
            pred_noise = model(x_t, t)
            
            loss = nn.functional.mse_loss(pred_noise, eps_t)

            # use huber_loss can panelty the extreme value
            #loss = nn.functional.huber_loss(pred_noise, eps_t, reduction='mean', delta=3.0)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)

        lr_sched.step()
        losses.append(epoch_loss / len(loader.dataset))
        print(f"Epoch [{epoch:4d}/{EPOCHS}]  loss={losses[-1]:.6f}")

    os.makedirs(f"{PREFIX}/checkpoints", exist_ok=True)
    torch.save({
        "model_state":   model.state_dict(),
        "model_kwargs":  model.kwargs,
        "scaler":        scaler,
        "epoch":         epoch,
        "num_timesteps": NUM_TIMESTEPS,
        "levy_alpha":    LEVY_ALPHA,
        "gammas":        gammas.cpu(),
        "bargammas":     bargammas.cpu(),
        "sigmas":        sigmas.cpu(),
        "barsigmas":     barsigmas.cpu(),
        "losses":        losses,
    }, f"{PREFIX}/checkpoints/factor_ddpm_ep{epoch:04d}.pt")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, EPOCHS + 1), losses)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title(f"Training Loss (DLPM α={LEVY_ALPHA})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("assets/factor_ddpm_loss.png", dpi=150)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X, scaler = load_data(f"{PREFIX}/factors.csv")
    print(f"Dataset: {X.shape}  |  LEVY_ALPHA={LEVY_ALPHA}  |  T={NUM_TIMESTEPS}")

    gammas, bargammas, sigmas, barsigmas = levy_noise_schedule(LEVY_ALPHA, NUM_TIMESTEPS)

    loader    = DataLoader(TensorDataset(torch.tensor(X)), batch_size=BATCH_SIZE, shuffle=True)
    model     = FactorDenoiser(dim=128, n_heads=4, cond_dim=256, num_blocks=3).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    train(model, loader, gammas, bargammas, sigmas, barsigmas, optimizer, scaler)
