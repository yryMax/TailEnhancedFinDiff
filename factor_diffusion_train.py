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
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline
from factor_diffusion_levy import levy_noise_schedule, sample_skewed_levy
import matplotlib.pyplot as plt
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cfg(exp_name: str) -> dict:
    with open(f"model/{exp_name}/cfg.yaml") as f:
        return yaml.safe_load(f)


def load_data(csv_path, factor_names, vix_path=None):
    """
    Load factor returns (rows = trading days, cols = factors), standardized.

    If `vix_path` is given, also load VIXCLS, align it to the factor trading days (forward-
    then-backward fill the ~302 FRED gaps), take log, and standardize. VIX is conditioned at
    the SAME day as each row, so the caller pairs vix[i+1] with target X[i+1].

    :return: (X_norm, scaler, vix_norm, vix_scaler). vix_norm/vix_scaler are None when
             vix_path is None. X_norm is (T, F); vix_norm is (T, 1).
    """
    df = pd.read_csv(csv_path, index_col=0)[factor_names].dropna()
    X  = df.values.astype(np.float32)
    scaler = StandardScaler().fit(X)
    X_norm = scaler.transform(X)

    if vix_path is None:
        return X_norm, scaler, None, None

    dates = pd.to_datetime(df.index)
    v = pd.read_csv(vix_path, parse_dates=["observation_date"]).set_index("observation_date")["VIXCLS"]
    v = v.reindex(dates).ffill().bfill()                       # align to factor days; fill FRED gaps
    v_raw      = v.to_numpy(dtype=np.float32)[:, None]         # (T, 1) raw VIX level
    vix_scaler = make_pipeline(
        FunctionTransformer(np.log, inverse_func=np.exp, check_inverse=False), StandardScaler())
    vix_norm   = vix_scaler.fit_transform(v_raw).astype(np.float32)
    return X_norm, scaler, vix_norm, vix_scaler


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
        self.vix_embed = nn.Sequential(                                            # scalar VIX → context
            nn.Linear(1, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.null_vix = nn.Parameter(torch.zeros(1, cond_dim))                      # learned "no VIX" token
        self.blocks = nn.ModuleList([DiTBlock(dim, n_heads, cond_dim) for _ in range(num_blocks)])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada_out  = nn.Linear(cond_dim, 2 * dim)                                # final AdaLN-Zero
        nn.init.zeros_(self.ada_out.weight)
        nn.init.zeros_(self.ada_out.bias)
        self.out_proj = nn.Linear(dim, 1)                                          # token → scalar

    def forward(self, x, t, c=None, vix=None, cond_drop_mask=None, vix_drop_mask=None):
        """
        :param x: noisy target factor returns (day t+1), shape (B, F)
        :param t: timestep indices, shape (B,)
        :param c: clean condition factor returns (day t), shape (B, F); None → fully
                  unconditional (every sample uses the learned null token).
        :param vix: scalar VIX condition, shape (B, 1); None → use the learned null-VIX
                  token (sample not conditioned on VIX). Added to the context so a single
                  model covers both VIX-conditioned and VIX-free generation.
        :param cond_drop_mask: optional bool (B,); True rows use the null token instead
                  of cond_mlp(c). Used for condition dropout during training
        :param vix_drop_mask: optional bool (B,); True rows use the null-VIX token instead
                  of vix_embed(vix). Independent VIX dropout (CFG) during training
        :return: predicted noise, shape (B, F)
        """
        e_t  = self.t_embed(self.t_sin(t))                            # (B, cond_dim)
        if c is None:
            ctx = self.null_cond.expand(x.shape[0], -1)              # (B, cond_dim) all-null
        else:
            ctx = self.cond_mlp(c)                                    # (B, cond_dim) full prev cross-section
            if cond_drop_mask is not None:
                ctx = torch.where(cond_drop_mask[:, None], self.null_cond, ctx)
        if vix is None:
            ctx = ctx + self.null_vix                                # (B, cond_dim) "no VIX"
        else:
            vctx = self.vix_embed(vix)                               # (B, cond_dim) from scalar VIX
            if vix_drop_mask is not None:
                vctx = torch.where(vix_drop_mask[:, None], self.null_vix, vctx)
            ctx = ctx + vctx
        cond = ctx[:, None, :] + self.feature_cond_embed + e_t[:, None, :]   # (B, F, cond_dim) per-token
        h = self.in_proj(x.unsqueeze(-1)) + self.feature_embed       # (B, F, dim)
        for blk in self.blocks:
            h = blk(h, cond)
        g, b = self.ada_out(cond).chunk(2, dim=-1)
        h = self.norm_out(h) * (1 + g) + b
        return self.out_proj(h).squeeze(-1)                          # (B, F)

def dlpm_loss(model, x, c, t, bg, bs, alpha, device, cond_drop_prob=0.0,
              vix=None, vix_drop_prob=0.0):
    """
    DLPM epsilon-prediction loss (single-sample MSE).

    Forward: x_t = bg * x_0 + bs * eps,  eps = sqrt(a) * z  with a_i per-factor i.i.d.

    Condition dropout (CFG-style): each sample's condition is replaced by the learned
    null token with probability `cond_drop_prob`. This teaches the model the marginal
    p(F_0) (no previous day) alongside the transition p(F_{t+1}|F_t), so sampling can
    self-start a path without an external seed.

    VIX dropout is INDEPENDENT of the factor-condition dropout (separate Bernoulli), so the
    model jointly learns p(.|F_t, VIX), p(.|F_t), p(.|VIX) and p(.) — i.e. it can generate
    with or without the VIX condition from a single set of weights.
    """
    B, D     = x.shape
    drop     = torch.rand(B, device=device) < cond_drop_prob   # (B,) True → use null cond token
    a        = sample_skewed_levy(alpha, (B, D), device)
    eps_t    = a.sqrt() * torch.randn_like(a)
    x_t      = bg * x + bs * eps_t
    vix_drop = None if vix is None else (torch.rand(B, device=device) < vix_drop_prob)
    pred = model(x_t, t, c, vix=vix, cond_drop_mask=drop, vix_drop_mask=vix_drop)
    return (pred - eps_t).pow(2).mean()


class FactorDenoiserXS(nn.Module):
    """
    Cross-sectional (unconditional) Transformer denoiser — the pre-temporal architecture.
    Kept alongside the conditional FactorDenoiser so that checkpoints from the
    cross-sectional experiments (stable / new_factors / DDPM / ablation / dow30 /
    data_scaling / temperal_new ...) still load. Predicts eps_t from (x_t, t) only;
    timestep is injected via diffusers' AdaLN-continuous conditioning.

    :param num_factors: number of factor tokens (D)
    :param dim: token embedding dimension
    :param n_heads: number of attention heads
    :param cond_dim: timestep embedding dimension
    :param num_blocks: number of transformer blocks
    """
    def __init__(self, num_factors, dim=128, n_heads=8, cond_dim=128, num_blocks=6):
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

    def forward(self, x, t, c=None, vix=None, **_ignored):
        """
        :param x: noisy factor returns, shape (B, F)
        :param t: timestep indices, shape (B,)
        (c / vix accepted for signature compatibility with FactorDenoiser; must be None)
        :return: predicted noise, shape (B, F)
        """
        assert c is None and vix is None, "FactorDenoiserXS is unconditional"
        cond = self.t_embed(self.t_sin(t))          # (B, cond_dim)
        h = self.in_proj(x.unsqueeze(-1))            # (B, F, dim)
        h = h + self.feature_embed                   # add factor identity
        for blk in self.blocks:
            h = blk(h, added_cond_kwargs={"pooled_text_emb": cond})
        return self.out_proj(h).squeeze(-1)          # (B, F)


def is_conditional_ckpt(ckpt: dict) -> bool:
    """A checkpoint is conditional (Markov-1 temporal DiT) iff its weights carry the
    condition machinery. Robust to cfg differences across experiment generations."""
    sd = ckpt.get("model_state") or ckpt.get("ema_state") or {}
    return "cond_mlp.0.weight" in sd


def build_denoiser(ckpt: dict, device=None):
    """
    Construct the right denoiser class for a checkpoint and load its weights.
    Conditional checkpoints prefer EMA weights when present.
    :return: model in eval mode on `device`.
    """
    device = device or DEVICE
    cls    = FactorDenoiser if is_conditional_ckpt(ckpt) else FactorDenoiserXS
    model  = cls(**ckpt["model_kwargs"]).to(device)
    state  = ckpt.get("ema_state") or ckpt["model_state"]
    model.load_state_dict(state)
    model.eval()
    return model


def dlpm_loss_xs(model, x, t, bg, bs, alpha, mc_outer, mc_inner, device, L=None):
    """
    Cross-sectional DLPM epsilon-prediction loss via median-of-means MC estimator.
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


def train_xs(model, loader, optimizer, scaler, cfg, ckpt_path,
             L=None, loss_plot_path="assets/factor_loss.png"):
    """
    Train the cross-sectional (unconditional) DLPM denoiser.

    :param model: FactorDenoiserXS
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

            loss = dlpm_loss_xs(model, x, t, bg, bs, levy_alpha, mc_outer, mc_inner, DEVICE, L=L_d)

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

    plot_loss(losses, loss_plot_path)


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
          loss_plot_path="assets/factor_loss.png", vix_scaler=None):
    """
    Train DLPM denoiser.

    :param model: FactorDenoiser
    :param loader: DataLoader yielding (c, x) or (c, vix, x) batches — c=day-t condition,
                   x=day-(t+1) target, vix=day-(t+1) VIX condition.
    :param optimizer: torch optimizer
    :param scaler: fitted StandardScaler (saved into ckpt)
    :param cfg: dict with epochs, num_timesteps, levy_alpha
    :param ckpt_path: full path to write checkpoint
    :param vix_scaler: fitted VIX scaler (saved into ckpt for inference); None if no VIX
    """

    epochs         = cfg["epochs"]
    num_timesteps  = cfg["num_timesteps"]
    levy_alpha     = cfg["levy_alpha"]
    cond_drop_prob = cfg.get("cond_drop_prob", 0.1)
    vix_drop_prob  = cfg.get("cond_drop_prob_vix", 0.15)
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
        for batch in loader:
            if len(batch) == 3:
                c, vix, x = batch
                vix = vix.to(DEVICE)
            else:
                (c, x), vix = batch, None
            c   = c.to(DEVICE)
            x   = x.to(DEVICE)
            t   = torch.randint(1, num_timesteps, (x.size(0),), device=DEVICE)
            bg  = bargammas_d[t].unsqueeze(-1)
            bs  = barsigmas_d[t].unsqueeze(-1)

            loss = dlpm_loss(model, x, c, t, bg, bs, levy_alpha, DEVICE,
                             cond_drop_prob=cond_drop_prob,
                             vix=vix, vix_drop_prob=vix_drop_prob)

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
        "vix_scaler":    vix_scaler,
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
    vix_file  = cfg.get("vix_file")

    # Dispatch: cfgs with `cond_drop_prob` train the Markov-1 conditional temporal model
    # (FactorDenoiser); all others train the cross-sectional model (FactorDenoiserXS).
    temporal = "cond_drop_prob" in cfg

    X, scaler, vix_norm, vix_scaler = load_data(f"{prefix}/{data_file}", cfg["factors"], vix_file)
    print(f"experiment id: {args.exp_name}  (mode={'temporal-cond' if temporal else 'cross-sectional'})")
    print(f"data file: {data_file}")

    ckpt_path = f"{prefix}/checkpoints/{cfg['ckpt_name']}.pt"

    if temporal:
        cond_np   = torch.tensor(X[:-1])                  # day t
        target_np = torch.tensor(X[1:])                   # day t+1
        if vix_norm is not None:
            vix_np  = torch.tensor(vix_norm[1:])          # VIX at day t+1 (same day as target)
            dataset = TensorDataset(cond_np, vix_np, target_np)
        else:
            dataset = TensorDataset(cond_np, target_np)
        loader    = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)
        model     = FactorDenoiser(num_factors=len(cfg["factors"])).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
        train(model, loader, optimizer, scaler, cfg, ckpt_path, vix_scaler=vix_scaler)
    else:
        L = None
        if cfg.get("use_L_noise", False):
            C = np.corrcoef(X, rowvar=False).astype(np.float32)
            L = torch.from_numpy(np.linalg.cholesky(C))
            print(f"use_L_noise=True, L shape={tuple(L.shape)}, "
                  f"max|C-LL^T|={np.abs(C - (L.numpy() @ L.numpy().T)).max():.2e}")

        loader    = DataLoader(TensorDataset(torch.tensor(X)), batch_size=cfg["batch_size"], shuffle=True)
        model     = FactorDenoiserXS(num_factors=len(cfg["factors"])).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
        train_xs(model, loader, optimizer, scaler, cfg, ckpt_path, L=L)
