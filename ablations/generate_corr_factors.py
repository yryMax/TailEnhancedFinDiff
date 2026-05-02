import numpy as np
import pandas as pd
import yaml

with open("../cfg.yaml") as f:
    _exp = yaml.safe_load(f)["experiment_name"]
PREFIX = f"model/{_exp}"

REFERENCE_CSV = f"{PREFIX}/factors.csv"
SEED = 42

FACTOR_COLS = ["market", "growth", "momentum", "quality", "size", "value", "volatility"]


Historical = np.array([
    [1.000000,0.159385,-0.081098,0.031570,-0.305856,0.211322,-0.669525],  # market
    [0.159385,1.000000,0.119482,-0.012941,-0.279095,0.025260,-0.156905],  # growth
    [-0.081098,0.119482,1.000000,-0.039689,0.059821,0.031362,0.058105],  # momentum
    [ 0.031570,-0.012941,-0.039689,1.000000,-0.062059,-0.028770,-0.170962],  # quality
    [-0.305856,-0.279095,0.059821,-0.062059,1.000000,0.011590,0.425909],  # size
    [0.211322,0.025260,0.031362,-0.028770,0.011590,1.000000,-0.246636],  # value
    [-0.669525,-0.156905,0.058105,-0.170962,0.425909,-0.246636,1.000000],  # volatility
])


def anisotropic_corr(C_hist, p=5.0):
    vals, vecs = np.linalg.eigh(C_hist)
    vals = vals[::-1];
    vecs = vecs[:, ::-1]

    vals_new = vals ** p
    vals_new = vals_new / vals_new.mean() * vals.mean()

    C_new = vecs @ np.diag(vals_new) @ vecs.T

    d = np.sqrt(np.diag(C_new))
    return C_new / np.outer(d, d)


def amplify_corr(C_hist, beta=4):
  n = C_hist.shape[0]
  lam_min = np.linalg.eigvalsh(C_hist).min()
  beta_max = 1.0 / (1.0 - lam_min) * 0.99   # 留一点 margin
  if beta > beta_max:
      beta = beta_max
  return beta * C_hist + (1 - beta) * np.eye(n)


with open("../cfg.yaml") as f:
    _cfg = yaml.safe_load(f)

_exp = _cfg["experiment_name"]
PREFIX = f"model/{_exp}"

def main() -> None:
    OUTPUT_CSV = f"{PREFIX}/factors_amplified_synth.csv"
    CORR = amplify_corr(Historical)
    ref = pd.read_csv(REFERENCE_CSV, index_col=0)
    n_rows = len(ref)

    eigvals = np.linalg.eigvalsh(CORR)
    assert eigvals.min() > 0, f"correlation matrix not PSD: min eig = {eigvals.min()}"

    rng = np.random.default_rng(SEED)
    samples = rng.multivariate_normal(mean=np.zeros(7), cov=CORR, size=n_rows)

    df = pd.DataFrame(samples, columns=FACTOR_COLS, index=ref.index)
    df.insert(0, "alpha", 1.0)

    df.to_csv(OUTPUT_CSV)


if __name__ == "__main__":
    main()
