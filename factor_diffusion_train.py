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
from factor_diffusion_levy import levy_noise_schedule, sample_skewed_levy
import matplotlib.pyplot as plt
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cfg(exp_name: str) -> dict:
    with open(f"model/{exp_name}/cfg.yaml") as f:
        return yaml.safe_load(f)


def load_data(csv_path, factor_names):
    X = pd.read_csv(csv_path, index_col=0)[factor_names].dropna().values.astype(np.float32)
    scaler = StandardScaler().fit(X)
    X_norm = scaler.transform(X)
    return X_norm, scaler


class DiTBlock(nn.Module):
    """
    DiT block with per-token AdaLN-Zero conditioning.
        h <- h + a1 * MHSA( (1+g1) * LN(h) + b1 )
        h <- h + a2 * FFN ( (1+g2) * LN(h) + b2 )
    Zero-init makes every block an identity at start (gate a=0), so training begins
    from a stable pass-through and learns the modulation gradually.
    """
    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn  = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.ada = nn.Linear(cond_dim, 6 * dim)        # c_i -> (g1,b1,a1,g2,b2,a2)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, h, c):
        """
        :param h: (B, F, dim) tokens
        :param c: (B, F, cond_dim) per-token condition.
        """
        # ada: nn.Linear cond_dim -> 6 * dim, ada(c): (B,F,6*dim), each chunk: (B,F,dim)
        g1, b1, a1, g2, b2, a2 = self.ada(c).chunk(6, dim=-1)
        x = self.norm1(h) * (1 + g1) + b1
        attn_out, _ = self.attn(x, x, x, need_weights=False) # self attention
        h = h + a1 * attn_out
        x = self.norm2(h) * (1 + g2) + b2
        h = h + a2 * self.ffn(x)
        return h


class FactorDenoiser(nn.Module):
    """
    Per-token condition: c_i = cond_mlp(c_prev) + feature_cond_embed[i] + e_t, where
    cond_mlp maps the full previous cross-section so each factor's dynamics can depend
    on the whole previous regime (cross-factor spillover)
    :param num_factors: number of factor tokens (D)
    :param dim: token embedding dimension
    :param n_heads: number of attention heads
    :param cond_dim: condition / timestep embedding dimension
    :param num_blocks: number of DiT blocks
    """
    def __init__(self, num_factors, dim=128, n_heads=8, cond_dim=128, num_blocks=6):
        super().__init__()
        self.kwargs = dict(num_factors=num_factors, dim=dim, n_heads=n_heads,
                           cond_dim=cond_dim, num_blocks=num_blocks)
        self.t_sin   = Timesteps(cond_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.t_embed = TimestepEmbedding(in_channels=cond_dim, time_embed_dim=cond_dim)
        self.in_proj = nn.Linear(1, dim)                                            # factor -> factor embedding
        self.feature_embed = nn.Parameter(torch.randn(1, num_factors, dim) * 0.02)  # input token identity
        self.cond_mlp = nn.Sequential(                                             # prev cross-section → context
            nn.Linear(num_factors, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.null_cond = nn.Parameter(torch.zeros(1, cond_dim))                     # learned "no condition" token (BOS)
        self.feature_cond_embed = nn.Parameter(torch.randn(1, num_factors, cond_dim) * 0.02)  # per-factor cond id
        self.blocks = nn.ModuleList([DiTBlock(dim, n_heads, cond_dim) for _ in range(num_blocks)])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada_out  = nn.Linear(cond_dim, 2 * dim)                                # final AdaLN-Zero
        nn.init.zeros_(self.ada_out.weight)
        nn.init.zeros_(self.ada_out.bias)
        self.out_proj = nn.Linear(dim, 1)                                          # token → scalar

    def forward(self, x, t, c=None, cond_drop_mask=None):
        """
        :param x: noisy target factor returns (day t+1), shape (B, F)
        :param t: timestep indices, shape (B,)
        :param c: clean condition factor returns (day t), shape (B, F); None → fully
                  unconditional (every sample uses the learned null token).
        :param cond_drop_mask: optional bool (B,); True rows use the null token instead
                  of cond_mlp(c). Used for condition dropout during training
        :return: predicted noise, shape (B, F)
        """
        e_t  = self.t_embed(self.t_sin(t))                            # (B, cond_dim)
        if c is None:
            ctx = self.null_cond.expand(x.shape[0], -1)              # (B, cond_dim) all-null
        else:
            ctx = self.cond_mlp(c)                                    # (B, cond_dim) full prev cross-section
            if cond_drop_mask is not None:
                ctx = torch.where(cond_drop_mask[:, None], self.null_cond, ctx)
        cond = ctx[:, None, :] + self.feature_cond_embed + e_t[:, None, :]   # (B, F, cond_dim) per-token
        h = self.in_proj(x.unsqueeze(-1)) + self.feature_embed       # (B, F, dim)
        for blk in self.blocks:
            h = blk(h, cond)
        g, b = self.ada_out(cond).chunk(2, dim=-1)
        h = self.norm_out(h) * (1 + g) + b
        return self.out_proj(h).squeeze(-1)                          # (B, F)

def dlpm_loss(model, x, c, t, bg, bs, alpha, device, cond_drop_prob=0.0):
    """
    DLPM epsilon-prediction loss (single-sample MSE).

    Forward: x_t = bg * x_0 + bs * eps,  eps = sqrt(a) * z  with a_i per-factor i.i.d.

    Condition dropout (CFG-style): each sample's condition is replaced by the learned
    null token with probability `cond_drop_prob`. This teaches the model the marginal
    p(F_0) (no previous day) alongside the transition p(F_{t+1}|F_t), so sampling can
    self-start a path without an external seed.
    """
    B, D  = x.shape
    drop  = torch.rand(B, device=device) < cond_drop_prob   # (B,) True → use null token
    a     = sample_skewed_levy(alpha, (B, D), device)
    eps_t = a.sqrt() * torch.randn_like(a)
    x_t   = bg * x + bs * eps_t
    return (model(x_t, t, c, cond_drop_mask=drop) - eps_t).pow(2).mean()


def plot_loss(losses, loss_plot_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(losses) + 1), losses)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(loss_plot_path) or ".", exist_ok=True)
    fig.savefig(loss_plot_path, dpi=150)
    plt.close(fig)


def train(model, loader, optimizer, scaler, cfg, ckpt_path,
          loss_plot_path="assets/factor_loss.png"):
    """
    Train DLPM denoiser.

    :param model: FactorDenoiser
    :param loader: DataLoader yielding (c, x) batches — c=day-t condition, x=day-(t+1) target
    :param optimizer: torch optimizer
    :param scaler: fitted StandardScaler (saved into ckpt)
    :param cfg: dict with epochs, num_timesteps, levy_alpha
    :param ckpt_path: full path to write checkpoint
    """

    epochs         = cfg["epochs"]
    num_timesteps  = cfg["num_timesteps"]
    levy_alpha     = cfg["levy_alpha"]
    cond_drop_prob = cfg.get("cond_drop_prob", 0.1)
    ema_decay      = cfg.get("ema_decay", 0.999)

    _, bargammas, _, barsigmas = levy_noise_schedule(levy_alpha, num_timesteps)
    bargammas_d = bargammas.to(DEVICE)
    barsigmas_d = barsigmas.to(DEVICE)

    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    losses   = []

    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for (c, x) in loader:
            c   = c.to(DEVICE)
            x   = x.to(DEVICE)
            t   = torch.randint(1, num_timesteps, (x.size(0),), device=DEVICE)
            bg  = bargammas_d[t].unsqueeze(-1)
            bs  = barsigmas_d[t].unsqueeze(-1)

            loss = dlpm_loss(model, x, c, t, bg, bs, levy_alpha, DEVICE,
                             cond_drop_prob=cond_drop_prob)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)

            with torch.no_grad():
                msd = model.state_dict()
                for k, v in ema.items():
                    if v.dtype.is_floating_point:
                        v.mul_(ema_decay).add_(msd[k].detach(), alpha=1.0 - ema_decay)
                    else:
                        v.copy_(msd[k])

        lr_sched.step()
        losses.append(epoch_loss / len(loader.dataset))
        print(f"Epoch [{epoch:4d}/{epochs}]  loss={losses[-1]:.6f}")

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({
        "model_state":   model.state_dict(),
        "ema_state":     ema,
        "model_kwargs":  model.kwargs,
        "scaler":        scaler,
        "cfg":           cfg,
    }, ckpt_path)

    plot_loss(losses, loss_plot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", help="experiment name; reads model/<exp_name>/cfg.yaml")
    args = parser.parse_args()
    cfg    = load_cfg(args.exp_name)
    prefix = f"model/{args.exp_name}"
    data_file = cfg.get("data_file")
    X, scaler = load_data(f"{prefix}/{data_file}", cfg["factors"])
    cond_np   = torch.tensor(X[:-1])
    target_np = torch.tensor(X[1:])
    loader    = DataLoader(TensorDataset(cond_np, target_np), batch_size=cfg["batch_size"], shuffle=True)
    model     = FactorDenoiser(num_factors=len(cfg["factors"])).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)

    ckpt_path = f"{prefix}/checkpoints/{cfg['ckpt_name']}.pt"
    train(model, loader, optimizer, scaler, cfg, ckpt_path)
