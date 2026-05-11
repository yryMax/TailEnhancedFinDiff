import argparse
import os
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from sklearn.preprocessing import StandardScaler
from factor_diffusion_levy import levy_noise_schedule, sample_skewed_levy, sample_sas
import matplotlib.pyplot as plt
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cfg(exp_name: str) -> dict:
    """Load model/<exp_name>/cfg.yaml. Pure function — no side effects."""
    with open(f"model/{exp_name}/cfg.yaml") as f:
        return yaml.safe_load(f)


def load_data(csv_path, factor_names, seq_len):
    """
    Read factor CSV → standardize per factor → sliding-window stride=1 into (N_w, F, T).
    :param csv_path:     CSV with date index and factor columns
    :param factor_names: list of factor column names
    :param seq_len:      output sequence length T
    :return: windows of shape (N_w, F, seq_len), fitted StandardScaler over (T_dates, F)
    """
    X_flat = pd.read_csv(csv_path, index_col=0)[factor_names].dropna().values.astype(np.float32)
    scaler = StandardScaler().fit(X_flat)
    X_norm = scaler.transform(X_flat).astype(np.float32)         # (T_dates, F)

    if X_norm.shape[0] < seq_len:
        raise ValueError(f"Need at least seq_len={seq_len} rows, got {X_norm.shape[0]}")

    # sliding_window_view(axis=0) → (T_dates - seq_len + 1, F, seq_len)
    windows = np.lib.stride_tricks.sliding_window_view(X_norm, seq_len, axis=0)
    return np.ascontiguousarray(windows), scaler


class AdaLayerNorm(nn.Module):
    """LayerNorm with affine params predicted from a conditioning vector (zero-init = identity)."""
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, dim * 2))
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x, cond):
        gb = self.proj(cond)
        while gb.dim() < x.dim():
            gb = gb.unsqueeze(1)
        g, b = gb.chunk(2, dim=-1)
        return self.norm(x) * (1 + g) + b


class DilatedTemporalConv(nn.Module):
    """Dilated 1D conv along time, per (batch, factor). Multi-scale RF + residual."""
    def __init__(self, dim, dilations=(1, 2, 4)):
        super().__init__()
        layers = []
        for d in dilations:
            layers += [nn.Conv1d(dim, dim, kernel_size=3, padding=d, dilation=d), nn.GELU()]
        self.net = nn.Sequential(*layers[:-1])
        self.res = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, x):                                # x: (B*F, dim, T)
        return self.net(x) + self.res(x)


class FactorBlock(nn.Module):
    """
    One factorized block:
      Pathway A — temporal conv along T (per factor)
      Pathway B — cross-factor self-attention along F (per timestep)
    Both pathways operate on the same input in parallel, are summed onto the residual,
    then passed through AdaLN → FFN → AdaLN with timestep conditioning.
    """
    def __init__(self, dim, cond_dim, n_heads, dilations=(1, 2, 4)):
        super().__init__()
        self.t_conv = DilatedTemporalConv(dim, dilations)
        self.f_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)
        self.adanorm1 = AdaLayerNorm(dim, cond_dim)
        self.adanorm2 = AdaLayerNorm(dim, cond_dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )

    def forward(self, x, cond):
        """
        :param x:    (B, F, T, dim)
        :param cond: (B, cond_dim)
        """
        B, F, T, C = x.shape

        # Pathway A: temporal conv (per factor)
        x_t   = x.reshape(B * F, T, C).transpose(1, 2)                       # (B*F, dim, T)
        out_t = self.t_conv(x_t).transpose(1, 2).reshape(B, F, T, C)

        # Pathway B: cross-factor attention (per timestep)
        x_f   = x.transpose(1, 2).reshape(B * T, F, C)                       # (B*T, F, dim)
        out_f, _ = self.f_attn(x_f, x_f, x_f, need_weights=False)
        out_f = out_f.reshape(B, T, F, C).transpose(1, 2)                    # (B, F, T, dim)

        mixed = self.adanorm1(x + out_t + out_f, cond)
        return self.adanorm2(mixed + self.ffn(mixed), cond)


class FactorDenoiser(nn.Module):
    """
    Factorized denoiser for factor-return time series.
    Predicts noise eps_t given noisy input x_t (B, F, T) and timestep t.

    Stacks `num_blocks` FactorBlocks; each block runs temporal conv and cross-factor
    attention in parallel and fuses them via AdaLN-conditioned residual + FFN.

    :param num_factors: number of factor channels F
    :param seq_len:     temporal length T
    :param dim:         token embedding dimension
    :param n_heads:     attention heads
    :param cond_dim:    timestep embedding dimension
    :param num_blocks:  number of FactorBlocks
    :param dilations:   dilation rates for the temporal conv pathway
    """
    def __init__(self, num_factors, seq_len, dim=128, n_heads=8, cond_dim=128,
                 num_blocks=6, dilations=(1, 2, 4)):
        super().__init__()
        self.kwargs = dict(num_factors=num_factors, seq_len=seq_len, dim=dim,
                           n_heads=n_heads, cond_dim=cond_dim, num_blocks=num_blocks,
                           dilations=list(dilations))
        self.t_sin   = Timesteps(cond_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.t_embed = TimestepEmbedding(in_channels=cond_dim, time_embed_dim=cond_dim)
        self.in_proj        = nn.Linear(1, dim)
        self.feature_embed  = nn.Parameter(torch.randn(1, num_factors, 1, dim) * 0.02)
        self.temporal_embed = nn.Parameter(torch.randn(1, 1, seq_len, dim) * 0.02)
        self.blocks = nn.ModuleList([
            FactorBlock(dim, cond_dim, n_heads, dilations) for _ in range(num_blocks)
        ])
        self.final_norm = AdaLayerNorm(dim, cond_dim)
        self.out_proj   = nn.Linear(dim, 1)

    def forward(self, x, t):
        """
        :param x: noisy factor returns, shape (B, F, T)
        :param t: timestep indices, shape (B,)
        :return: predicted noise, shape (B, F, T)
        """
        cond = self.t_embed(self.t_sin(t))                                   # (B, cond_dim)

        h = self.in_proj(x.unsqueeze(-1))                                    # (B, F, T, dim)
        h = h + self.feature_embed + self.temporal_embed

        for block in self.blocks:
            h = block(h, cond)

        h = self.final_norm(h, cond)
        return self.out_proj(h).squeeze(-1)                                  # (B, F, T)

def dlpm_loss(model, x, t, bg, bs, alpha, mc_outer, mc_inner, device, L=None):
    """
    DLPM epsilon-prediction loss via median-of-means MC estimator.
    When mc_outer=1 and mc_inner=1, degenerates to single-sample MSE.

    Forward (per (b, f, τ)):
        x_t = bg * x_0 + bs * eps,   eps = sqrt(a) * z,   a ~ S(alpha/2, 1)
    When L is provided (Cholesky of factor-corr C), z is L-colored along factor F only;
    temporal correlations are left for the conv to learn. A is sampled independently
    per (B, F, T).

    :param x:  shape (B, F, T)
    :param t:  shape (B,)
    :param bg: shape (B, 1, 1)  bargammas[t]
    :param bs: shape (B, 1, 1)  barsigmas[t]
    :param L:  optional (F, F) Cholesky factor; applied via einsum on F axis
    """
    B, F, T = x.shape

    def _eps(a):
        z = torch.randn_like(a)
        if L is not None:
            # Apply L along factor axis only: z' = L @ z (per (b, τ))
            z = torch.einsum('fg,bgt->bft', L, z)
        return a.sqrt() * z

    if mc_outer == 1 and mc_inner == 1:
        a     = sample_skewed_levy(alpha, (B, F, T), device)
        eps_t = _eps(a)
        x_t   = bg * x + bs * eps_t
        return (model(x_t, t) - eps_t).pow(2).mean(dim=(-1, -2)).mean()

    N     = mc_outer * mc_inner
    x_mc  = x.unsqueeze(0).expand(N, -1, -1, -1).reshape(N * B, F, T)
    t_mc  = t.repeat(N)
    bg_mc = bg.repeat(N, 1, 1)
    bs_mc = bs.repeat(N, 1, 1)

    a_outer = sample_skewed_levy(alpha, (mc_outer * B, F, T), device)
    a_mc    = (a_outer.view(mc_outer, 1, B, F, T)
                      .expand(-1, mc_inner, -1, -1, -1)
                      .reshape(N * B, F, T))

    eps_t_mc = _eps(a_mc)
    x_t_mc   = bg_mc * x_mc + bs_mc * eps_t_mc

    losses_mc = (model(x_t_mc, t_mc) - eps_t_mc).pow(2).mean(dim=(-1, -2))
    losses_mc = losses_mc.view(mc_outer, mc_inner, B).mean(dim=1)
    loss, _   = losses_mc.median(dim=0)
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
            x   = x.to(DEVICE)                                   # (B, F, T)
            t   = torch.randint(1, num_timesteps, (x.size(0),), device=DEVICE)
            bg  = bargammas_d[t].view(-1, 1, 1)                  # broadcast over (F, T)
            bs  = barsigmas_d[t].view(-1, 1, 1)

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
        "cfg":           cfg,
        "L_noise":       L.detach().cpu() if L is not None else None,
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
    seq_len   = cfg["seq_len"]
    X, scaler = load_data(f"{prefix}/{data_file}", cfg["factors"], seq_len)   # (N_w, F, T)
    print(f"data file: {data_file}")
    print(f"experiment id: {args.exp_name}")
    print(f"windows: {X.shape}  (N_w, F, seq_len={seq_len})")

    L = None
    if cfg.get("use_L_noise", False):
        # factor-axis correlation only (temporal corr is learned by conv)
        flat = X.transpose(0, 2, 1).reshape(-1, X.shape[1])                   # (N_w*T, F)
        C    = np.corrcoef(flat, rowvar=False).astype(np.float32)
        L    = torch.from_numpy(np.linalg.cholesky(C))
        print(f"use_L_noise=True, L shape={tuple(L.shape)}, "
              f"max|C-LL^T|={np.abs(C - (L.numpy() @ L.numpy().T)).max():.2e}")

    loader    = DataLoader(TensorDataset(torch.tensor(X)), batch_size=cfg["batch_size"], shuffle=True)
    model     = FactorDenoiser(num_factors=len(cfg["factors"]), seq_len=seq_len).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)

    ckpt_path = f"{prefix}/checkpoints/{cfg['ckpt_name']}.pt"
    train(model, loader, optimizer, scaler, cfg, ckpt_path, L=L)
