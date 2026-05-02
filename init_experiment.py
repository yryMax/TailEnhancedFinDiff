"""
One-shot experiment scaffolding.

Usage:
    python init_experiment.py <exp_name> [--overwrite]
or:
    from init_experiment import init
    init("my_new_exp")
"""
import argparse
import os
import yaml

DEFAULT_CFG = {
    "characteristics": ["growth", "momentum", "quality", "size", "value", "volatility"],
    "factors":         ["market", "growth", "momentum", "quality", "size", "value", "volatility"],
    "train_path":    "data/train24y.parquet",
    "test_path":     "data/test1y.parquet",
    "data_file":     "factors.csv",
    "epochs":        200,
    "batch_size":    64,
    "lr":            1.0e-4,
    "num_timesteps": 100,
    "levy_alpha":    2.0,
    "mc_outer":      1,
    "mc_inner":      1,
    "ckpt_name":     "DDPM_vanilla",
    "num_generate":  4096,
    "use_L_noise":   False,
}

SUBDIRS = ["checkpoints", "samples", "test"]


def init(exp_name: str, root: str = "model") -> str:
    prefix = os.path.join(root, exp_name)
    for sub in SUBDIRS:
        os.makedirs(os.path.join(prefix, sub), exist_ok=True)

    cfg_path = os.path.join(prefix, "cfg.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(DEFAULT_CFG, f, sort_keys=False)

    print(f"Created experiment at {prefix}/")
    print(f"  subfolders: {', '.join(SUBDIRS)}")
    print(f"  cfg:        {cfg_path}")
    return prefix


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("exp_name")
    args = p.parse_args()
    init(args.exp_name)
