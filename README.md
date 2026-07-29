# TailEnhancedFinDiff

Heavy-tail-aware diffusion models for financial scenario generation.

A two-stage pipeline: (1) a **factor model** compresses the stock cross-section into a small
set of factor returns (OLS betas + Student-t idiosyncratic residuals); (2) a **denoising
diffusion model** learns the factor return distribution and generates synthetic scenarios,
which are mapped back to full stock cross-sections. The diffusion stage implements **DLPM**
(Denoising Lévy Probabilistic Models): the Gaussian noise of DDPM is replaced by SαS Lévy
noise (`levy_alpha < 2`), which fattens the tails of the learned distribution;
`levy_alpha = 2` degenerates exactly to DDPM.

Two diffusion model families live in this codebase and are **auto-detected from the
checkpoint** — no flags needed at inference:

| family | class | trains on | generates | experiments |
|---|---|---|---|---|
| cross-sectional | `FactorDenoiserXS` | i.i.d. daily factor cross-sections | one day at a time | `stable`, `new_factors`, `DDPM`, `ablation`, `dow30`, `data_scaling`, ... |
| conditional temporal | `FactorDenoiser` (DiT + AdaLN-Zero) | Markov-1 pairs `(F_t → F_{t+1})` with CFG-style null-token dropout, optional VIX conditioning | self-starting autoregressive paths | `temporal_cond`, `vix` |

## Setup

```bash
conda env create -f env.yaml
conda activate diffusion_factor_model
```

## Project structure

```
.
├── factor_model.py               # stock returns ⇄ factor returns; OLS betas + Student-t residuals
├── factor_diffusion_levy.py      # Lévy noise schedule and SαS sampling primitives
├── factor_diffusion_train.py     # both denoiser classes + both training loops (dispatch on cfg)
├── factor_diffusion_sample.py    # unified generate() dispatcher, SDEdit style transfer,
│                                 #   autoregressive rollout (generate_path), rejection sampling
├── scenario_generator.py         # sampler abstraction: Diffusion / Gaussian(-EWMA) / Resample
│                                 #   (bootstrap) + ScenarioGenerator (factor → stock mapping)
├── init_experiment.py            # scaffold a new experiment under model/<exp_name>/
│
├── factor_evaluation.ipynb           # cross-sectional factor eval (moments, KDE, Q-Q, corr)
├── stock_evaluation.ipynb            # cross-sectional stock eval + portfolio-value bands
├── conditional_evaluation.ipynb      # stress conditioning: guidance vs rejection vs truncated Gaussian
├── style_transfer.ipynb              # per-day OOS counterfactuals (SDEdit, e.g. "Value −30%")
├── temporal_factor_evaluation.ipynb  # temporal model: factor path eval  (exp: temporal_cond)
├── temporal_stock_evaluation.ipynb   # temporal model: stock path eval   (exp: temporal_cond)
├── vix_conditional_evaluation.ipynb  # VIX-conditioned generation eval   (exp: vix)
├── dow_evaluation.ipynb              # Dow30 1-min cross-section, no factor stage (exp: dow30)
├── downstream_replica.ipynb          # downstream stress-test replication
├── gics.ipynb                        # GICS sector-level analysis
│
├── metrics/                      # reusable evaluation metrics (statistic / temporal / risk)
├── ablations/                    # correlation-noise & data-scaling ablation scripts + results
├── figures_ppt/                  # scripts regenerating the presentation figures
├── analyze_ill_paths.py          # diagnose ill-behaved autoregressive rollouts
├── massiveDS/pull_minute.py      # Dow30 1-minute data downloader
├── CLAUDE.md                     # design notes: DLPM math, bug post-mortems, guidance theory
└── legacy/                       # earlier experiments (not part of the main pipeline)
```

## Data

Raw data are `.parquet` train/test splits (not in git). Each row needs: `returns`,
`stock_id` (`csecid`), and the characteristics of interest (size, value, momentum, ...).

```
data/
├── train24y.parquet
├── test1y.parquet
└── VIXCLS.csv          # only for VIX-conditioned experiments (FRED download)
```

## Pipeline

```bash
python init_experiment.py <exp_name>          # 1. scaffold model/<exp_name>/ + cfg.yaml
python factor_model.py <exp_name>             # 2. fit factor model → factors.csv, model.npz
python factor_diffusion_train.py <exp_name>   # 3. train the diffusion model → checkpoints/
python factor_diffusion_sample.py <exp_name>  # 4. sample → model/<exp_name>/samples/*.npy
```

Everything is driven by `model/<exp_name>/cfg.yaml`. Key switches:

- `levy_alpha` — tail heaviness; `2.0` = Gaussian/DDPM, `< 2` = heavy-tailed DLPM.
- `cond_drop_prob` — **presence of this key selects the conditional temporal model**;
  its value is the null-token dropout rate.
- `vix_file` — enables VIX conditioning (temporal family only; independent `cond_drop_prob_vix`).
- `use_L_noise` — L-coloured (correlated) noise from the data correlation Cholesky
  (cross-sectional family only).
- `mc_outer` / `mc_inner` — median-of-means MC loss estimator; important for stability
  when `levy_alpha < 2` (see CLAUDE.md).

Sampling extras (temporal checkpoints): `--seed-oos` / `--seed-date YYYY-MM-DD` seed the
rollout from a real out-of-sample day; `--vix 40` conditions the whole path on a VIX level.

The checkpoint stores its own `cfg`, scaler(s), and (if used) the L-noise Cholesky, so
inference is always consistent with training — `DiffusionSampler(ckpt_path)` needs
nothing else.

## Evaluation & regenerating artifacts

All result artifacts (pdf/png/parquet) are kept **out of git**; final versions live on the
company drive (`drive_export/`, see its `MANIFEST.md`). Every artifact can be regenerated:

| artifact | how to regenerate |
|---|---|
| factor-level eval report | run `factor_evaluation.ipynb` (set `_exp`) |
| stock-level eval + PV bands | run `stock_evaluation.ipynb` |
| stress-conditioning comparison | run `conditional_evaluation.ipynb` |
| per-day counterfactual panels | run `style_transfer.ipynb` |
| temporal path evals | run `temporal_factor_evaluation.ipynb` / `temporal_stock_evaluation.ipynb` |
| VIX regime scenarios | run `vix_conditional_evaluation.ipynb` |
| presentation figures | `python figures_ppt/make_figs.py`, `python figures_ppt/make_violins.py` |
| correlation ablation | `python ablations/corr.py` (see `ablations/corr.md`) |
| data-scaling ablation | `python ablations/data_scaling.py` (results csv committed) |
| generated sample paths (`achievement/`) | `python factor_diffusion_sample.py temporal_cond --seed-oos` |

Notebooks set their target experiment via the `_exp` variable in the first cell.

## Credits

DLPM implementation adapted from https://github.com/darioShar/DLPM
(`bndm`: https://github.com/xchhuang/bndm was consulted for baselines; neither is vendored —
both former submodules were removed, the code here is self-contained).
