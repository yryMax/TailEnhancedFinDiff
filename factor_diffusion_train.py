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


class DropPath(nn.Module):
    """Per-sample stochastic depth (residual drop)."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep)
        return x * mask / keep


def _modulate(x, shift, scale):
    # x: (B, L, dim), shift/scale: (B, dim)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    DiT-style block with AdaLN-Zero gating (Peebles & Xie 2023) and SwiGLU FFN.
    Three residual branches: joint (F*T) self-attention, factor-axis self-attention
    (per-timestep cross-factor coupling — restores the factorized inductive bias),
    and SwiGLU FFN. One linear from cond → nine modulations (shift/scale/gate × 3).
    """
    def __init__(self, dim, n_heads, cond_dim, num_factors, seq_len,
                 mlp_ratio=4, drop_path=0.0):
        super().__init__()
        self.num_factors = num_factors
        self.seq_len     = seq_len
        # Joint (F*T) self-attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn  = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        # Factor-axis self-attention (per timestep)
        self.norm_f = nn.LayerNorm(dim, elementwise_affine=False)
        self.fattn  = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        # SwiGLU FFN
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        hidden = int(dim * mlp_ratio)
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(dim, hidden)
        self.w3 = nn.Linear(hidden, dim)
        # AdaLN-Zero: 9 modulations per block
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 9 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)
        self.drop_path = DropPath(drop_path)

    def forward(self, x, cond):
        # x: (B, F*T, dim), cond: (B, cond_dim)
        B, L, C = x.shape
        F_, T = self.num_factors, self.seq_len
        s1, sc1, g1, sf, scf, gf, s2, sc2, g2 = self.ada(cond).chunk(9, dim=-1)

        # Branch 1: joint (F*T) attention
        h = _modulate(self.norm1(x), s1, sc1)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop_path(g1.unsqueeze(1) * h)

        # Branch 2: factor-axis attention (B, F*T, C) → (B*T, F, C) → attend → reshape back
        h = _modulate(self.norm_f(x), sf, scf)
        h = h.reshape(B, F_, T, C).transpose(1, 2).reshape(B * T, F_, C)
        h, _ = self.fattn(h, h, h, need_weights=False)
        h = h.reshape(B, T, F_, C).transpose(1, 2).reshape(B, F_ * T, C)
        x = x + self.drop_path(gf.unsqueeze(1) * h)

        # Branch 3: SwiGLU FFN
        h = _modulate(self.norm2(x), s2, sc2)
        h = self.w3(nn.functional.silu(self.w1(h)) * self.w2(h))
        x = x + self.drop_path(g2.unsqueeze(1) * h)
        return x


class FactorDenoiser(nn.Module):
    """
    DiT-style denoiser for factor-return time series.

    Flattens the (F, T) grid into F*T tokens and runs full self-attention with
    AdaLN-Zero timestep conditioning over `num_blocks` transformer blocks.
    Per-factor output projection lets each factor learn its own readout scale.

    :param num_factors:   number of factor channels F
    :param seq_len:       temporal length T
    :param dim:           token embedding dimension
    :param n_heads:       attention heads
    :param cond_dim:      timestep embedding dimension
    :param num_blocks:    transformer depth
    :param mlp_ratio:     SwiGLU hidden = dim * mlp_ratio
    :param drop_path_max: stochastic depth schedule peak (0 → drop_path_max over depth)
    """
    def __init__(self, num_factors, seq_len, dim=384, n_heads=8, cond_dim=384,
                 num_blocks=12, mlp_ratio=4, drop_path_max=0.1):
        super().__init__()
        self.kwargs = dict(
            num_factors=num_factors, seq_len=seq_len, dim=dim, n_heads=n_heads,
            cond_dim=cond_dim, num_blocks=num_blocks, mlp_ratio=mlp_ratio,
            drop_path_max=drop_path_max,
        )
        self.num_factors = num_factors
        self.seq_len     = seq_len

        # Timestep embedding
        self.t_sin   = Timesteps(cond_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.t_embed = TimestepEmbedding(in_channels=cond_dim, time_embed_dim=cond_dim)

        # Token embedding + 2D positional encoding for (factor, time)
        self.in_proj        = nn.Linear(1, dim)
        self.feature_embed  = nn.Parameter(torch.randn(1, num_factors, 1, dim) * 0.02)
        self.temporal_embed = nn.Parameter(torch.randn(1, 1, seq_len, dim) * 0.02)

        # Transformer trunk with linear stochastic-depth schedule
        dpr = torch.linspace(0, drop_path_max, num_blocks).tolist()
        self.blocks = nn.ModuleList([
            DiTBlock(dim, n_heads, cond_dim, num_factors, seq_len, mlp_ratio, dpr[i])
            for i in range(num_blocks)
        ])

        # Final AdaLN + per-factor output projection (zero-init → eps_pred starts at 0)
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.final_ada  = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * dim))
        nn.init.zeros_(self.final_ada[-1].weight)
        nn.init.zeros_(self.final_ada[-1].bias)
        self.out_proj = nn.Parameter(torch.zeros(num_factors, dim, 1))

    def forward(self, x, t):
        """
        :param x: noisy factor returns, shape (B, F, T)
        :param t: timestep indices, shape (B,)
        :return: predicted noise, shape (B, F, T)
        """
        B, Fc, T = x.shape
        cond = self.t_embed(self.t_sin(t))                                   # (B, cond_dim)

        h = self.in_proj(x.unsqueeze(-1))                                    # (B, F, T, dim)
        h = h + self.feature_embed + self.temporal_embed
        h = h.reshape(B, Fc * T, -1)                                         # (B, F*T, dim)

        for blk in self.blocks:
            h = blk(h, cond)

        shift, scale = self.final_ada(cond).chunk(2, dim=-1)
        h = _modulate(self.final_norm(h), shift, scale)                      # (B, F*T, dim)
        h = h.reshape(B, Fc, T, -1)
        eps = torch.einsum('bftd,fdo->bfto', h, self.out_proj).squeeze(-1)
        return eps                                                           # (B, F, T)

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

def plot_loss(epochs, losses):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, epochs + 1), losses)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname("assets/factor_loss.png") or ".", exist_ok=True)
    fig.savefig("assets/factor_loss.png", dpi=150)
    plt.close(fig)


def train(model, loader, optimizer, scaler, cfg, ckpt_path,
          L=None, loss_plot_path=""):
    """
    Train DLPM denoiser.

    :param model: FactorDenoiser
    :param loader: DataLoader yielding (x,) batches
    :param optimizer: torch optimizer
    :param scaler: fitted StandardScaler (saved into ckpt)
    :param cfg: dict with epochs, num_timesteps, levy_alpha, mc_outer, mc_inner
    :param ckpt_path: full path to write checkpoint
    :param L: optional Cholesky factor for L-noise
    :param resume: if True and ckpt_path exists, restore model/optim/sched/epoch and continue
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
    start_epoch = 0
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

        if ck.get("model_kwargs") and ck["model_kwargs"] != model.kwargs:
            raise ValueError(
                f"model_kwargs mismatch — ckpt {ck['model_kwargs']} vs current {model.kwargs}"
            )
        # cfg fields that change loss/noise semantics; must match to safely resume
        for k in ("levy_alpha", "num_timesteps", "seq_len", "mc_outer", "mc_inner"):
            if ck.get("cfg", {}).get(k) != cfg.get(k):
                raise ValueError(
                    f"cfg mismatch on '{k}': ckpt={ck['cfg'].get(k)} vs cfg={cfg.get(k)}"
                )

        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        lr_sched.load_state_dict(ck["lr_sched_state"])
        start_epoch = ck.get("epoch", 0)
        losses      = list(ck.get("losses", []))
        if start_epoch >= epochs:
            print(f"ckpt already at epoch {start_epoch} >= target {epochs}; nothing to do")
            return
        print(f"resuming from epoch {start_epoch + 1}/{epochs} (ckpt: {ckpt_path})")

    def _save_ckpt(epoch):
        torch.save({
            "model_state":     model.state_dict(),
            "model_kwargs":    model.kwargs,
            "optimizer_state": optimizer.state_dict(),
            "lr_sched_state":  lr_sched.state_dict(),
            "epoch":           epoch,
            "losses":          losses,
            "scaler":          scaler,
            "cfg":             cfg,
            "L_noise":         L.detach().cpu() if L is not None else None,
        }, ckpt_path)

    for epoch in range(start_epoch + 1, epochs + 1):
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

        if epoch % 20 == 0:
            _save_ckpt(epoch)

    _save_ckpt(epochs)

    plot_loss(epochs, losses)


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
