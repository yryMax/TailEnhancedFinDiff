import argparse
import os
import shutil
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
    "ckpt_name":     "DDPM_vanilla",
    "num_generate":  4096,
    "cond_drop_prob": 0.1,
}

SUBDIRS = ["checkpoints", "samples", "test"]


def init(exp_name: str, root: str = "model", from_exp: str | None = None) -> str:
    prefix = os.path.join(root, exp_name)
    for sub in SUBDIRS:
        os.makedirs(os.path.join(prefix, sub), exist_ok=True)

    cfg_path = os.path.join(prefix, "cfg.yaml")
    if from_exp is not None:
        src_cfg = os.path.join(root, from_exp, "cfg.yaml")
        if not os.path.isfile(src_cfg):
            raise FileNotFoundError(f"source cfg not found: {src_cfg}")
        shutil.copyfile(src_cfg, cfg_path)
    else:
        with open(cfg_path, "w") as f:
            yaml.safe_dump(DEFAULT_CFG, f, sort_keys=False)

    return prefix


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("exp_name")
    p.add_argument("--from", dest="from_exp", default=None,
                   help="existing exp_id under model/ to copy cfg.yaml from")
    args = p.parse_args()
    init(args.exp_name, from_exp=args.from_exp)
