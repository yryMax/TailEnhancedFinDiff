### PROJECT SETUP

Create and activate the conda environment:

```bash
conda env create -f env.yaml
conda activate diffusion_factor_model
```

### PROJECT STRUCTURE
```
.
├── cfg.yaml                      # all (hyper)parameters and config settings
├── factor_model.py               # cross-sectional OLS factor model (fit + save/load)
├── factor_diffusion_levy.py      # Lévy noise schedule and SαS sampling primitives
├── factor_diffusion_train.py     # DLPM training (FactorDenoiser + dlpm_loss)
├── factor_diffusion_sample.py    # reverse diffusion sampling (unconditional + conditional)
│
├── factor_evaluation.ipynb       # factor-level evaluation (moments, KDE, Q-Q, correlation)
├── stock_evaluation.ipynb        # stock-level evaluation (return reconstruction, dispersion)
│
├── metrics/                      # reusable evaluation metrics
│   ├── statistic.py              # distributional statistics (covariance, Frobenius, ...)
│   ├── statistic_temporal.py     # temporal / autocorrelation metrics
│   └── risk.py                   # portfolio risk metrics
│
├── data/                         # input panel data (not tracked)
│   ├── train24y.parquet
│   └── test1y.parquet
├── model/                        # fitted model outputs (not tracked)
│   └── regression/
│       ├── factors.csv
│       ├── model.npz
│       ├── checkpoints/
│       └── samples/
├── assets/                       # figures and result exports
└── legacy/                       # earlier experiments (not part of main pipeline)
```

### DATA PREPROCESSING

Place panel data parquets under `data/`:

```
data/
  train24y.parquet   # columns: date, csecid, returns, <factor_names>
  test1y.parquet
```

### CONFIGURATION

Edit `cfg.yaml` to configure hyperparameters before running:

```yaml
train:
  factor_names: [market, growth, momentum, quality, size, value, volatility]
  prefix: model/regression   # output directory for checkpoints and factors.csv
  levy_alpha: 1.9            # stability index; set to 2.0 to recover standard DDPM
  epochs: 100
  num_timesteps: 100
  mc_outer: 1   # outer draws of the Lévy subordinator a ~ S(alpha/2, 1); median taken across these
  mc_inner: 1   # inner draws per outer a; mean taken across these (same a, different Gaussian z)

sample:
  num_generate: 4096
```

**Monte Carlo loss estimator** (`mc_outer` / `mc_inner`):
The DLPM training loss is an expectation over the Lévy subordinator `a ~ S(alpha/2, 1)`, which has heavy tails and can produce extreme values. The *median-of-means* estimator reduces variance:

- `mc_inner` i.i.d. Gaussian draws `z` are taken for the same `a`; squared errors are **averaged** — reduces variance from the Gaussian component.
- `mc_outer` independent draws of `a` each yield one inner mean; the **median** is taken across these — robust against rare extreme Lévy draws.

Setting both to `1` degenerates to a standard single-sample MSE (equivalent to DDPM training). Increasing `mc_outer` (e.g. 5–10) improves stability when `levy_alpha` is small, at a proportional compute cost.

### RUNNING

```bash
# 1. Fit factor model and save factor return series
python factor_model.py

# 2. Train diffusion model on factor returns
python factor_diffusion_train.py

# 3. Generate synthetic factor samples
python factor_diffusion_sample.py
```

Outputs are written to `{prefix}/` (`model/regression/` by default):

| File | Description |
|------|-------------|
| `factors.csv` | Fitted factor return time series |
| `model.npz` | OLS betas and residual parameters |
| `checkpoints/factor_ep{N:04d}.pt` | Diffusion model checkpoint |
| `samples/factor_{N}.npy` | Generated factor samples |
