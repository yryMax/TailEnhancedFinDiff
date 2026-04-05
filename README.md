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
├── factor_model.py               # convert between stock returns and factor returns; fit the systematic part and idiosyncratic residuals
├── factor_diffusion_levy.py      # Lévy noise schedule and SαS sampling primitives
├── factor_diffusion_train.py     # diffusion training
├── factor_diffusion_sample.py    # diffusion sampling
│
├── factor_evaluation.ipynb       # factor-level evaluation (moments, KDE, Q-Q, correlation)
├── stock_evaluation.ipynb        # stock-level evaluation (return reconstruction, dispersion)
│
├── metrics/                      # reusable evaluation metrics
│   ├── statistic.py              # distributional statistics (covariance, Frobenius, ...)
│   ├── statistic_temporal.py     # temporal / autocorrelation metrics
│   └── risk.py                   # portfolio risk metrics
├── assets/                       # figures and result exports
└── legacy/                       # earlier experiments (not part of main pipeline)
```

### DATA PREPROCESSING
The raw data must be in .parquet format with train/data splits.

Each .parquet must contain rows with features: 
- returns
- stock_id (cse_cid)
- the characteristics that you are interested in (e.g. size, value, momentum, ...)

After running `factor_model.py`, the derived folder should be structured as follows
```
├── data/
│   ├── train24y.parquet # training dataset that the user prepared
│   └── test1y.parquet # test dataset that the user prepared
├── model/
│   └── regression/   # user defined prefix
│       ├── factors.csv # factor returns as training data for the diffusion model
│       ├── model.npz # contains the requied data to convert between stock returns and factor returns (OLS betas and residual parameters)
│       ├── checkpoints/ # saved diffusion model that generate factor returns
│       └── samples/ # generated factor return samples
```

The stock returns can be constructed in memory via sampled factor and factor model.


### TRAINING

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
