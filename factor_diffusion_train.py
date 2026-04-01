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
MODE          = "DLPM"   # "DDPM" | "DLPM"
MC_OUTER      = 3        # outer MC samples: median taken over these (robust to Lévy spikes)
MC_INNER      = 1        # inner MC samples: mean taken over these (z_t is Gaussian, 1 is enough)
PREFIX        = "model/regression"


def load_data(csv_path):
    X = pd.read_csv(csv_path, index_col=0)[FACTOR_NAMES].dropna().values.astype(np.float32)
    scaler = StandardScaler().fit(X)
    X_norm = scaler.transform(X)
    return X_norm, scaler


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

"""
class FactorDenoiserMLP(nn.Module):
    def __init__(self, dim=7, hidden_dim=64, cond_dim=128):
        super().__init__()
        self.kwargs = dict(dim=dim, hidden_dim=hidden_dim, cond_dim=cond_dim)

        self.t_sin = Timesteps(cond_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.t_embed = TimestepEmbedding(in_channels=cond_dim, time_embed_dim=cond_dim)

        self.net1 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU()
        )

        self.time_proj = nn.Linear(cond_dim, hidden_dim)

        self.net2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x, t):
        t_emb = self.t_embed(self.t_sin(t))  # (B, cond_dim)
        t_hidden = self.time_proj(t_emb)  # (B, hidden_dim)
        h = self.net1(x)  # (B, hidden_dim)
        h = h + t_hidden
        out = self.net2(h)  # (B, dim=7)
        return out
"""

def train(model, loader, gammas, bargammas, sigmas, barsigmas, optimizer, scaler):
    import matplotlib.pyplot as plt
    os.makedirs("checkpoints", exist_ok=True)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

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

            if MODE == "DDPM":
                z_t = torch.randn_like(x)
                x_t = bg * x + bs * z_t
                pred_noise = model(x_t, t)
                loss       = nn.functional.mse_loss(pred_noise, z_t)
            else:
                # DLPM: Median-of-Means over Lévy variance samples
                # Outer a_t samples: different for each outer draw
                # Inner z_t samples: independent Gaussians for same a_t
                # → mean over inner, then median over outer
                B, D = x.shape
                N    = MC_OUTER * MC_INNER          # total MC samples

                # Expand x and t to (N*B, D)
                x_mc  = x.unsqueeze(0).expand(N, -1, -1).reshape(N * B, D)
                t_mc  = t.repeat(N)
                bg_mc = bargammas_d[t_mc].unsqueeze(-1)
                bs_mc = barsigmas_d[t_mc].unsqueeze(-1)

                # Outer a_t: (MC_OUTER, B, D) → replicate MC_INNER times → (N*B, D)
                a_outer = sample_skewed_levy(LEVY_ALPHA, (MC_OUTER * B, D), DEVICE)
                a_mc    = (a_outer.view(MC_OUTER, 1, B, D)
                                  .expand(-1, MC_INNER, -1, -1)
                                  .reshape(N * B, D))

                sigma_mc  = a_mc * bs_mc ** 2
                z_mc      = torch.randn(N * B, D, device=DEVICE)
                x_t_mc    = bg_mc * x_mc + sigma_mc.sqrt() * z_mc
                eps_t_mc  = (x_t_mc - bg_mc * x_mc) / bs_mc   # = sqrt(a_t) * z_t

                pred_mc   = model(x_t_mc, t_mc)

                # Per-sample squared error, mean over D → (N*B,)
                losses_mc = (pred_mc - eps_t_mc).pow(2).mean(dim=-1)

                # Reshape → (MC_OUTER, MC_INNER, B): mean over inner, median over outer
                losses_mc = losses_mc.view(MC_OUTER, MC_INNER, B).mean(dim=1)
                loss_med, _ = losses_mc.median(dim=0)          # (B,)
                loss      = loss_med.mean()

            # use huber_loss can panelty the extreme value
            # loss = nn.functional.huber_loss(pred_noise, eps_t, reduction='mean', delta=3.0)
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
        "mode":          MODE,
        "levy_alpha":    LEVY_ALPHA,
        "gammas":        gammas.cpu(),
        "bargammas":     bargammas.cpu(),
        "sigmas":        sigmas.cpu(),
        "barsigmas":     barsigmas.cpu(),
        "losses":        losses,
    }, f"{PREFIX}/checkpoints/factor_{MODE}_ep{epoch:04d}.pt")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, EPOCHS + 1), losses)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title(f"Training Loss ({MODE}" + (f", α={LEVY_ALPHA}" if MODE == "DLPM" else "") + ")")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("assets/factor_ddpm_loss.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    X, scaler = load_data(f"{PREFIX}/factors.csv")
    print(f"Dataset: {X.shape}  |  MODE={MODE}" + (f"  |  LEVY_ALPHA={LEVY_ALPHA}" if MODE == "DLPM" else "") + f"  |  T={NUM_TIMESTEPS}")

    gammas, bargammas, sigmas, barsigmas = levy_noise_schedule(LEVY_ALPHA, NUM_TIMESTEPS)

    loader    = DataLoader(TensorDataset(torch.tensor(X)), batch_size=BATCH_SIZE, shuffle=True)
    model     = FactorDenoiser().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    train(model, loader, gammas, bargammas, sigmas, barsigmas, optimizer, scaler)
