# TailEnhancedFinDiff

Heavy-tail-aware diffusion models for financial scenario generation.


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
├── stock_evaluation.ipynb            # stock eval(cross-sectional + temporal)
├── conditional_evaluation.ipynb      # stress conditioning: guidance vs rejection vs truncated Gaussian
├── temporal_factor_evaluation.ipynb  # temporal model: factor path eval  (exp: temporal_cond)
├── temporal_stock_evaluation.ipynb   # temporal model: stock path eval   (exp: temporal_cond)
├── vix_conditional_evaluation.ipynb  # VIX-conditioned generation eval   (exp: vix)
├── gics.ipynb                        # Downstream tasks
│
├── metrics/                      # reusable evaluation metrics (statistic / temporal / risk)
```

## Data

Raw data are `.parquet` train/test splits (not in git). Each row needs: `returns`,
`stock_id` (`csecid`), and the characteristics of interest (size, value, momentum, ...).

## Checkpoints Structure
Two diffusion model families live in this codebase and are auto-detected from the
checkpoint

| family | class | trains on | generates | experiments |
|---|---|---|---|---|
| cross-sectional | `FactorDenoiserXS` | i.i.d. daily factor cross-sections | one day at a time | `stable`, `new_factors`, `DDPM`, `ablation`, `dow30`, `data_scaling`, ... |
| conditional temporal | `FactorDenoiser` (DiT + AdaLN-Zero) | Markov-1 pairs `(F_t → F_{t+1})` with CFG-style null-token dropout, optional VIX conditioning | self-starting autoregressive paths | `temporal_cond`, `vix` |

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

**For every time you evaluate always refer to the checkpoint's stored configuration**

## Evaluation 

All result artifacts (pdf/png/parquet) are kept **out of git**; final versions live on the
company drive (`drive_export/`, see its `MANIFEST.md`). Every artifact can be regenerated:

| artifact                      | how to regenerate                                                          |
|-------------------------------|----------------------------------------------------------------------------|
| factor-level eval report      | run `factor_evaluation.ipynb` (set `_exp`)                                 |
| stock-level eval    | run `stock_evaluation.ipynb`                                               |
| stress-conditioning comparison | run `conditional_evaluation.ipynb`                                         |
| per-day counterfactual panels | run `style_transfer.ipynb`                                                 |
| temporal path evals           | run `temporal_factor_evaluation.ipynb` / `temporal_stock_evaluation.ipynb` |
| downstream tasks              | run `gics.ipynb`                                                           |



Notebooks set their target experiment via the `_exp` variable in the first cell. To
reproduce your own results, set `_exp` to the experiment name of interest and run the notebook.
## Credits

DLPM: https://github.com/darioShar/DLPM
BNDM: https://github.com/xchhuang/bndm