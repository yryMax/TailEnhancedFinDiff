# TailEnhancedFinDiff

Two-stage pipeline: FactorModel (stage 1, extracts factor returns) → DLPM diffusion
denoiser (stage 2, `FactorDenoiser` in `factor_diffusion_train.py`). Current denoiser
is a DiT-style transformer (joint F×T attention + factor-axis attention + SwiGLU,
AdaLN-Zero), ~52M params, ckpt `DDPM_big_corr`.

---

## Diagnostic Report — Overfitting vs Mode-Concentration vs Regime-Shift (2026-05-17)

**Question:** the DiT denoiser's marginals (KDE/QQ) look worse than the bootstrap
baseline. Is it overfitting, mode-concentrating, or is OOS just a different regime?

**Method:** `diagnostic_overfit.py` — load `DDPM_big_corr.pt` (epoch 200), put model
in `.eval()` (DropPath off), and measure ε-MSE on **2000 subsampled train windows**
vs **all 239 OOS windows**, per diffusion timestep `t`, fixed noise seeds. Plus
per-factor daily dispersion of generated samples vs train vs OOS (original space).

### Result 1 — ε-MSE train vs OOS, per timestep (THE smoking gun)

| t  | train | OOS  | gap%  |
|----|-------|------|-------|
| 1  | 0.44  | 4.29 | 880%  |
| 3  | 0.16  | 3.88 | 2399% |
| 5  | 0.13  | 2.85 | 2041% |
| 10 | 0.19  | 1.90 | 902%  |
| 20 | 0.33  | 1.25 | 283%  |
| 30 | 0.38  | 0.87 | 132%  |
| 50 | 0.31  | 0.44 | 42%   |
| 70 | 0.16  | 0.18 | 12%   |
| 90 | 0.02  | 0.02 | -3%   |
| 99 | 0.0003| 0.0003| -4%  |
| **mean** | **0.21** | **1.57** | **644%** |

**Verdict: SEVERE overfitting, concentrated at low timesteps.**

---

## Denoiser Architecture Comparison — 4 models + 1 ablation (2026-05-17)

All 5 ckpts in `model/temperal_new/checkpoints/`. **Only the denoiser differs**;
everything else is held fixed: `num_factors=7`, `levy_alpha=2.0` (→ Gaussian DLPM
≡ DDPM), `num_timesteps=100`, `epochs=200`, `lr=1e-4`, `batch_size=64`,
`seq_len=22`, `L_noise=None`, same StandardScaler, ε-prediction MSE loss, cosine
Lévy schedule. Architectures below are **reconstructed from each ckpt's
`model_state` key structure** (the old denoiser source is no longer in
`factor_diffusion_train.py` — only `DDPM_big*`'s DiT class remains).

| ckpt | dim | blk | params | core denoiser design |
|------|-----|-----|--------|----------------------|
| `DDPM_vanilla`     | 128 | 6  | 3.00M  | TCN: stacked dilated temporal Conv1d (d=1,2,4) **then separate** factor-attention blocks |
| `DDPM_newDenoiser` | 128 | 6  | 2.24M  | per-block **fused** [dilated temporal conv + factor-axis attn + FFN], AdaLN (shift/scale) cond |
| `DDPM_big`         | 384 | 12 | 39.6M  | DiT transformer: **joint F×T self-attn** + SwiGLU + AdaLN-Zero (6 mods), per-factor zero-init out head |
| `DDPM_big_corr`    | 384 | 12 | 52.0M  | `DDPM_big` **+ extra factor-axis self-attn branch** per block (AdaLN-Zero 9 mods); current best on corr |
| `DDPM_1` *(ablation)* | 128 | 6 | 2.24M | **= `newDenoiser` with `seq_len=1`** → temporal axis removed (1 day at a time) |

### Architecture deltas (what each step changed)

1. **vanilla → newDenoiser**: same dim/blocks/param-class, but restructured from
   "conv-stack → attention" (two separate stages) into a **per-block fusion**
   (conv+factor-attn+FFN inside every block) with **AdaLN** time conditioning.
   Tighter cross-factor/temporal coupling at equal capacity (params even drop
   3.00M→2.24M, the separate attn_blocks were heavier than the fused FFN).
2. **newDenoiser → big**: architectural family change. Drops dilated temporal
   conv entirely; goes **pure attention DiT** — flatten (F,T)→F·T tokens, one
   **joint F×T self-attention**, SwiGLU FFN, **AdaLN-Zero** (gated, zero-init so
   block starts as identity), per-factor zero-init output projection. Scale-up:
   dim 128→384, blocks 6→12 (2.24M→39.6M, ~18×).
3. **big → big_corr**: adds a **second attention branch per block on the factor
   axis** (reshape so the 7 factors attend to each other directly), on top of the
   joint F×T attn. AdaLN-Zero modulation count 6→9 (third gated branch). This is
   the explicit cross-factor inductive bias added to **recover the corr metric**
   (52.0M).
4. **DDPM_1 ablation**: identical to `newDenoiser` but `seq_len=1`
   (`temporal_embed` is (1,1,**1**,128), temporal conv degenerate over length-1).
   Isolates the **value of temporal modeling**: compare vs `newDenoiser` to read
   off how much the T-axis context contributes.

### How the families differ conceptually

- **Temporal modeling:** vanilla/newDenoiser/DDPM_1 use **dilated 1-D conv**
  along T (local, multi-scale receptive field). `big`/`big_corr` use
  **attention over the flattened F·T grid** (global, content-based, no conv).
- **Cross-factor coupling:** vanilla = separate post-hoc attn blocks;
  newDenoiser = per-block factor attn; `big` = only implicit via joint F·T attn;
  `big_corr` = explicit dedicated factor-axis attn branch.
- **Time conditioning:** vanilla has no AdaLN (plain embed add); newDenoiser =
  AdaLN (shift+scale); `big`/`big_corr` = **AdaLN-Zero** (adds zero-init gate →
  residual-safe, the DiT recipe).
- **Capacity:** ~2–3M (small family) vs ~40–52M (big family), ~18–23× gap —
  relevant to the overfitting finding above (the big family is the one that
  memorizes low-t; small family is capacity-limited instead).

**Caveat:** `DDPM_vanilla`/`DDPM_newDenoiser`/`DDPM_1` ckpts have `epoch=None`
saved (older save format) but `cfg.epochs=200`; treat as 200-epoch runs.
