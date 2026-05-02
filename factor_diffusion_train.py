import argparse
import os
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from sklearn.preprocessing import StandardScaler
from factor_diffusion_levy import levy_noise_schedule, sample_skewed_levy, sample_sas
import matplotlib.pyplot as plt
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cfg(exp_name: str) -> dict:
    """Load model/<exp_name>/cfg.yaml. Pure function — no side effects."""
    with open(f"model/{exp_name}/cfg.yaml") as f:
        return yaml.safe_load(f)


def load_data(csv_path, factor_names):
    X = pd.read_csv(csv_path, index_col=0)[factor_names].dropna().values.astype(np.float32)
    scaler = StandardScaler().fit(X)
    X_norm = scaler.transform(X)
    return X_norm, scaler


class FactorDenoiser(nn.Module):
    """
    Transformer-based denoiser for factor return vectors.
    Predicts the noise eps_t given noisy input x_t and timestep t.

    Each factor is treated as a token; timestep is injected via AdaLN conditioning.
    :param num_factors: number of factor tokens (D)
    :param dim: token embedding dimension
    :param n_heads: number of attention heads
    :param cond_dim: timestep embedding dimension
    :param num_blocks: number of transformer blocks
    """
    def __init__(self, num_factors, dim=64, n_heads=4, cond_dim=128, num_blocks=2):
        super().__init__()
        self.kwargs = dict(num_factors=num_factors, dim=dim, n_heads=n_heads,
                           cond_dim=cond_dim, num_blocks=num_blocks)
        self.t_sin   = Timesteps(cond_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.t_embed = TimestepEmbedding(in_channels=cond_dim, time_embed_dim=cond_dim)
        self.in_proj = nn.Linear(1, dim)                                        # scalar → token
        self.feature_embed = nn.Parameter(torch.randn(1, num_factors, dim) * 0.02)  # learnable factor id
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

def dlpm_loss(model, x, t, bg, bs, alpha, mc_outer, mc_inner, device, L=None):
    """
    Compute DLPM epsilon-prediction loss via median-of-means MC estimator.
    When mc_outer=1 and mc_inner=1, degenerates to a single-sample MSE.

    Forward: x_t = bg * x_0 + bs * eps,  eps = sqrt(a) * z  with a_i per-factor i.i.d.
    When L (Cholesky of data corr) is provided, the Gaussian component is L-colored:
        eps = sqrt(a) * (z @ L.T),  so  Cov(eps | a) = diag(sqrt a) C diag(sqrt a).
    A is still sampled independently per factor (DLPM style).
    """
    B, D = x.shape

    def _eps(a):
        z = torch.randn_like(a)
        if L is not None:
            z = z @ L.T
        return a.sqrt() * z

    if mc_outer == 1 and mc_inner == 1:
        a     = sample_skewed_levy(alpha, (B, D), device)
        eps_t = _eps(a)
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

    eps_t_mc = _eps(a_mc)
    x_t_mc   = bg_mc * x_mc + bs_mc * eps_t_mc

    losses_mc = (model(x_t_mc, t_mc) - eps_t_mc).pow(2).mean(dim=-1)
    losses_mc = losses_mc.view(mc_outer, mc_inner, B).mean(dim=1)   # mean over inner
    loss, _   = losses_mc.median(dim=0)                              # median over outer
    return loss.mean()


def train(model, loader, optimizer, scaler, cfg, ckpt_path,
          L=None, loss_plot_path="assets/factor_loss.png"):
    """
    Train DLPM denoiser.

    :param model: FactorDenoiser
    :param loader: DataLoader yielding (x,) batches
    :param optimizer: torch optimizer
    :param scaler: fitted StandardScaler (saved into ckpt)
    :param cfg: dict with epochs, num_timesteps, levy_alpha, mc_outer, mc_inner
    :param ckpt_path: full path to write checkpoint
    :param L: optional Cholesky factor for L-noise
    """

    epochs        = cfg["epochs"]
    num_timesteps = cfg["num_timesteps"]
    levy_alpha    = cfg["levy_alpha"]
    mc_outer      = cfg["mc_outer"]
    mc_inner      = cfg["mc_inner"]

    _, bargammas, _, barsigmas = levy_noise_schedule(levy_alpha, num_timesteps)
    bargammas_d = bargammas.to(DEVICE)
    barsigmas_d = barsigmas.to(DEVICE)
    L_d         = L.to(DEVICE) if L is not None else None

    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    losses   = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for (x,) in loader:
            x   = x.to(DEVICE)
            t   = torch.randint(1, num_timesteps, (x.size(0),), device=DEVICE)
            bg  = bargammas_d[t].unsqueeze(-1)
            bs  = barsigmas_d[t].unsqueeze(-1)

            loss = dlpm_loss(model, x, t, bg, bs, levy_alpha, mc_outer, mc_inner, DEVICE, L=L_d)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)

        lr_sched.step()
        losses.append(epoch_loss / len(loader.dataset))
        print(f"Epoch [{epoch:4d}/{epochs}]  loss={losses[-1]:.6f}")

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({
        "model_state":   model.state_dict(),
        "model_kwargs":  model.kwargs,
        "scaler":        scaler,
        "cfg":           cfg
    }, ckpt_path)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, epochs + 1), losses)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(loss_plot_path) or ".", exist_ok=True)
    fig.savefig(loss_plot_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", help="experiment name; reads model/<exp_name>/cfg.yaml")
    args = parser.parse_args()

    cfg    = load_cfg(args.exp_name)
    prefix = f"model/{args.exp_name}"

    data_file = cfg.get("data_file")
    X, scaler = load_data(f"{prefix}/{data_file}", cfg["factors"])
    print(f"data file: {data_file}")
    print(f"experiment id: {args.exp_name}")

    L = None
    if cfg.get("use_L_noise", False):
        C = np.corrcoef(X, rowvar=False).astype(np.float32)
        L = torch.from_numpy(np.linalg.cholesky(C))
        print(f"use_L_noise=True, L shape={tuple(L.shape)}, "
              f"max|C-LL^T|={np.abs(C - (L.numpy() @ L.numpy().T)).max():.2e}")

    loader    = DataLoader(TensorDataset(torch.tensor(X)), batch_size=cfg["batch_size"], shuffle=True)
    model     = FactorDenoiser(num_factors=len(cfg["factors"])).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)

    ckpt_path = f"{prefix}/checkpoints/{cfg['ckpt_name']}.pt"
    train(model, loader, optimizer, scaler, cfg, ckpt_path, L=L)
