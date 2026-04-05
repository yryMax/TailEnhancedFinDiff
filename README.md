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

Some Tips on hyperparameters
- levy_alpha



### SAMPLING
The stock returns can be constructed in memory via sampled factor and factor model.