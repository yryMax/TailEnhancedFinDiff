# Factor-space violin figures for the PPT (slimmed remake of cond1/cond2).
# The daily factor samples behind conditional_evaluation.ipynb's violins are not
# on disk, so they are regenerated here with the same code path (guidance_scale=5,
# N=5000) and cached under figures_ppt/factor_samples/. Additive only.
import os
import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from factor_model import *              # noqa: E402,F403
from scenario_generator import *        # noqa: E402,F403

OUT = ROOT / "figures_ppt"
CACHE = OUT / "factor_samples"
CACHE.mkdir(exist_ok=True)

PREFIX = "model/new_factors"
with open(f"{PREFIX}/cfg.yaml") as f:
    _cfg = yaml.safe_load(f)
FACTOR_NAMES = _cfg["factors"]
FACTOR_DIM = len(FACTOR_NAMES)
CKPT_PATH = f"{PREFIX}/checkpoints/{_cfg['ckpt_name']}.pt"

GUIDANCE_S = 5
N = 5000

# ------------------------------------------------------------- condition setup
def per_day_rate(cum_return, days=22):
    return (1 + cum_return) ** (1 / days) - 1

def _to_norm(scaler, value, idx):
    dummy = np.zeros((1, FACTOR_DIM))
    dummy[0, idx] = value
    return float(scaler.transform(dummy)[0, idx])

def make_cond_fn(interval, scaler, feature_idx=0, cond_name=None):
    thl, thr = interval
    if thl is not None:
        qmax_norm = _to_norm(scaler, thl, feature_idx)
        def cond_fn(x0_hat):
            return torch.relu(x0_hat[:, feature_idx] - qmax_norm) ** 2
        cond_fn.bounds = (None, thl)
    else:
        qmin_norm = _to_norm(scaler, thr, feature_idx)
        def cond_fn(x0_hat):
            return torch.relu(qmin_norm - x0_hat[:, feature_idx]) ** 2
        cond_fn.bounds = (thr, None)
    cond_fn.feature_idx = feature_idx
    cond_fn.label = cond_name
    return cond_fn

# ------------------------------------------------------------- sample / cache
print("loading model ...")
train_model = load_model(PREFIX)
sampler = DiffusionSampler(CKPT_PATH, guidance_scale=GUIDANCE_S)
scenario_gen = ScenarioGenerator(train_model, sampler)
scaler = sampler.scaler

scenarios = {}
for fac in ("value", "momentum"):
    idx = FACTOR_NAMES.index(fac)
    scenarios[f"{fac}_up_30"] = make_cond_fn((None, per_day_rate(0.3)), scaler, idx, f"{fac}_up_30")
    scenarios[f"{fac}_down_30"] = make_cond_fn((per_day_rate(-0.3), None), scaler, idx, f"{fac}_down_30")

uncon = np.load(f"{PREFIX}/samples/uncon_factors.npy")

diff_f, gaus_f = {}, {}
gaussian_sampler = GaussianSampler(train_model.F, scaler, guidance_scale=GUIDANCE_S)
for label, cond in scenarios.items():
    p = CACHE / f"{label}_diffusion.npy"
    if p.exists():
        diff_f[label] = np.load(p)
    else:
        print(f"sampling diffusion {label} ...")
        diff_f[label] = scenario_gen.cond_factor_generate(N, cond)
        np.save(p, diff_f[label])
    g = CACHE / f"{label}_gaussian.npy"
    if g.exists():
        gaus_f[label] = np.load(g)
    else:
        print(f"sampling gaussian {label} ...")
        gaus_f[label] = gaussian_sampler.cond_generate_truncated(N, cond.feature_idx, *cond.bounds)
        np.save(g, gaus_f[label])
    print(f"{label:18s} diffusion {diff_f[label].shape}  gaussian {gaus_f[label].shape}")

# ------------------------------------------------------------- plotting
INK = "#0b0b0b"; INK2 = "#52514e"; BASE = "#c3c2b7"; GRID = "#e1e0d9"; GRAY = "#9c9a94"
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "semibold",
    "axes.titlecolor": INK, "axes.labelsize": 12, "axes.labelcolor": INK2,
    "axes.edgecolor": BASE, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 12, "ytick.labelsize": 11, "legend.fontsize": 11,
    "legend.frameon": False, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white",
})

STYLE = {
    "value_up_30":      dict(label="Value +30%",    dark="#008300", light="#a6d9a6"),
    "value_down_30":    dict(label="Value −30%",    dark="#b50063", light="#eba8cb"),
    "momentum_up_30":   dict(label="Momentum +30%", dark="#d95926", light="#f5bda4"),
    "momentum_down_30": dict(label="Momentum −30%", dark="#4a3aa7", light="#beb7ec"),
}
SHOW = ["market", "momentum", "value", "volatility"]   # slimmed factor set
PAGES = {"value": ["value_up_30", "value_down_30"],
         "momentum": ["momentum_up_30", "momentum_down_30"]}
PAGES_TITLE = {"value": "Value ±30%", "momentum": "Momentum ±30%"}

def fig_violin(page, scens):
    cols = [FACTOR_NAMES.index(f) for f in SHOW]
    datasets = [("Unconditional", uncon, GRAY, 0.55)]
    for s in scens:
        st = STYLE[s]
        datasets.append((f"{st['label']}  Ours", diff_f[s], st["dark"], 0.85))
        datasets.append((f"{st['label']}  Gaussian", gaus_f[s], st["light"], 0.85))

    x_center = np.arange(len(SHOW))
    k = len(datasets)
    span = 0.84
    offsets = np.linspace(-span / 2, span / 2, k)
    width = span / k * 0.9

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    for offset, (label, data, color, alpha) in zip(offsets, datasets):
        parts = ax.violinplot([data[:, c] for c in cols], positions=x_center + offset,
                              widths=width, showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_edgecolor("white")
            pc.set_alpha(alpha)
        parts["cmedians"].set_color("white")
        parts["cmedians"].set_linewidth(1.4)

    # robust y-limits: heavy tails would otherwise squash the bodies
    allv = np.concatenate([d[:, cols].ravel() for _, d, _, _ in datasets])
    lo, hi = np.quantile(allv, [0.001, 0.999])
    pad = 0.12 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)

    ax.axhline(0, color=BASE, lw=1.0)
    ax.grid(axis="y", color=GRID, lw=1.0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xticks(x_center, SHOW)
    cond_fac = page  # the conditioned factor name
    for tick, name in zip(ax.get_xticklabels(), SHOW):
        if name == cond_fac:
            tick.set_fontweight("bold")
            tick.set_color(INK)
    ci = SHOW.index(cond_fac)
    ax.axvspan(ci - span / 2 - 0.09, ci + span / 2 + 0.09, color="#f3f2ee", zorder=0)
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(ci, 0.975, "conditioned", transform=trans, ha="center", va="top",
            fontsize=10.5, color=INK2, style="italic")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.set_ylabel("daily factor return")
    ax.set_title(f"Factor-space response to {PAGES_TITLE[page]} conditioning", loc="left")

    handles = [mpatches.Patch(color=c, alpha=a, label=l) for l, _, c, a in datasets]
    ax.legend(handles=handles, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              handlelength=1.4)
    fig.tight_layout()
    fig.savefig(OUT / f"p_{page}_violin.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

for page, scens in PAGES.items():
    fig_violin(page, scens)
print("done ->", OUT)
