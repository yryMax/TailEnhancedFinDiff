import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from sklearn.preprocessing import StandardScaler
from factor_diffusion_levy import levy_noise_schedule, sample_skewed_levy, sample_sas

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FACTOR_NAMES  = [ "market", "growth", "momentum", "quality", "size", "value", "volatility"]
EPOCHS        = 200
BATCH_SIZE    = 64
LR            = 1e-4
NUM_TIMESTEPS = 100
LEVY_ALPHA    = 1.9
MODE          = "DLPM"   # "DDPM" | "DLPM"
MC_OUTER      = 1        # outer MC samples: median taken over these (robust to Lévy spikes)
MC_INNER      = 1        # inner MC samples: mean taken over these
PREFIX        = "model/regression"


def load_data(csv_path):
    X = pd.read_csv(csv_path, index_col=0)[FACTOR_NAMES].dropna().values.astype(np.float32)
    scaler = StandardScaler().fit(X)
    X_norm = scaler.transform(X)
    return X_norm, scaler


class FactorDenoiser(nn.Module):
    """
    Transformer-based denoiser for factor return vectors.
    Predicts the noise eps_t given noisy input x_t and timestep t.

    Each factor is treated as a token; timestep is injected via AdaLN conditioning.
    :param dim: token embedding dimension
    :param n_heads: number of attention heads
    :param cond_dim: timestep embedding dimension
    :param num_blocks: number of transformer blocks
    """
    def __init__(self, dim=64, n_heads=4, cond_dim=128, num_blocks=1):
        super().__init__()
        self.kwargs = dict(dim=dim, n_heads=n_heads, cond_dim=cond_dim, num_blocks=num_blocks)
        self.t_sin   = Timesteps(cond_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.t_embed = TimestepEmbedding(in_channels=cond_dim, time_embed_dim=cond_dim)
        self.in_proj = nn.Linear(1, dim)                                        # scalar → token
        self.feature_embed = nn.Parameter(torch.randn(1, len(FACTOR_NAMES), dim) * 0.02)  # learnable factor id
        self.blocks = nn.ModuleList([
            BasicTransformerBlock(
                dim=dim, num_attention_heads=n_heads, attention_head_dim=dim // n_heads,
                norm_type="ada_norm_continuous",
                ada_norm_continous_conditioning_embedding_dim=cond_dim,
            )
            for _ in range(num_blocks)
        ])
        self.out_proj = nn.Linear(dim, 1)                                       # token → scalar

    def forward(self, x, t):
        """
        :param x: noisy factor returns, shape (B, F)
        :param t: timestep indices, shape (B,)
        :return: predicted noise, shape (B, F)
        """
        cond = self.t_embed(self.t_sin(t))          # (B, cond_dim)
        h = self.in_proj(x.unsqueeze(-1))            # (B, F, dim)
        h = h + self.feature_embed                   # add factor identity
        for blk in self.blocks:
            h = blk(h, added_cond_kwargs={"pooled_text_emb": cond})
        return self.out_proj(h).squeeze(-1)          # (B, F)

def dlpm_loss(model, x, t, bg, bs, alpha, mc_outer, mc_inner, device):
    """
    Compute DLPM epsilon-prediction loss via median-of-means MC estimator.
    When mc_outer=1 and mc_inner=1, degenerates to a single-sample MSE (no MC overhead).
    Forward process: x_t = bg * x_0 + bs * eps,  eps = sample_sas(a),  a ~ S(alpha/2, 1)
    Target:          eps_t = sample_sas(a)
    Loss:            E[(model(x_t, t) - eps_t)^2], mean over inner then median over outer
    """
    B, D = x.shape

    if mc_outer == 1 and mc_inner == 1:
        a     = sample_skewed_levy(alpha, (B, D), device)
        eps_t = sample_sas(a)
        x_t   = bg * x + bs * eps_t
        return (model(x_t, t) - eps_t).pow(2).mean(dim=-1).mean()

    N     = mc_outer * mc_inner
    x_mc  = x.unsqueeze(0).expand(N, -1, -1).reshape(N * B, D)
    t_mc  = t.repeat(N)
    bg_mc = bg.repeat(N, 1)
    bs_mc = bs.repeat(N, 1)

    a_outer = sample_skewed_levy(alpha, (mc_outer * B, D), device)
    a_mc    = (a_outer.view(mc_outer, 1, B, D)
                      .expand(-1, mc_inner, -1, -1)
                      .reshape(N * B, D))

    eps_t_mc = sample_sas(a_mc)
    x_t_mc   = bg_mc * x_mc + bs_mc * eps_t_mc

    losses_mc = (model(x_t_mc, t_mc) - eps_t_mc).pow(2).mean(dim=-1)
    losses_mc = losses_mc.view(mc_outer, mc_inner, B).mean(dim=1)   # mean over inner
    loss, _   = losses_mc.median(dim=0)                              # median over outer
    return loss.mean()


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
            t   = torch.randint(1, NUM_TIMESTEPS, (x.size(0),), device=DEVICE)
            bg  = bargammas_d[t].unsqueeze(-1)
            bs  = barsigmas_d[t].unsqueeze(-1)

            loss = dlpm_loss(model, x, t, bg, bs, LEVY_ALPHA, MC_OUTER, MC_INNER, DEVICE)

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

    # save the loss plot
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
