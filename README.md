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

### TRAINING

1. Edit `cfg.yaml` to configure hyperparameters before running
2. run `python factor_model.py` to fit the betas and residuals according to the provided parquet,
   do this for training data and test data separately.
3. run `python factor_diffusion_train.py` to train the diffusion model

The default config is trained on RTX 4080 mobile(16GB) with 200 epochs and 100 timestamps.
training and sampling take within minutes, but you definitely need a GPU to train.

After training the achievements in saved under PREFIX/checkpoints.

**Some tips on hyperparameters**

- `levy_alpha` controls how much the model focus on the tail behavior, the less it is the more it focuses on the tail,
  if set to 2 the algorithm will be equivalent to Gaussian diffusion
- The loss converge in a few epochs, 200 is already too much so dont set it larger, same goes for timestamps.
- mc_outer/mc_inner: controls the number of outlier, set them >1 can smooth out some outlier,
  but the effect seems not significant, and use mc sampling on each step could make the sampling process very long.
- Denoiser: from initial experiments attention is better than MLP, and 1 layer is already enough, (probably) we
  don't need a more complex architecture for the denoiser.

### SAMPLING

1. Edit `cfg.yaml` to configure number of samples
2. run `python factor_diffusion_sample.py` , the output is a .npy under /samples of your training directory

### EVALUATION

1. run `factor_evaluation.ipynb` to evaluate the generated factor returns
2. run `stock_evaluation.ipynb` to evaluate the reconstructed stock returns

Refer to `evaluation.pdf` if you only want to see my results.

### CREDITS

Code adapted from
https://github.com/darioShar/DLPM
