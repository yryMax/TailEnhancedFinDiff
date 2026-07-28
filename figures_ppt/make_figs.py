# PPT figures for the downstream stress-test story (gics.ipynb results, restyled).
# Additive only: reads the stored sample parquets, writes PNGs into figures_ppt/.
# Unlike gics.ipynb (one scenario draw per date, seed=0), all statistics here use
# the FULL scenario distribution per date: mean / 5-95% band instead of a single draw.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures_ppt"
SAMPLES = ROOT / "model/new_factors/samples"

# ---------------------------------------------------------------- chrome tokens
INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASE = "#c3c2b7"; REAL = "#0b0b0b"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 12,
    "axes.titlesize": 14, "axes.titleweight": "semibold", "axes.titlecolor": INK,
    "axes.labelsize": 12, "axes.labelcolor": INK2,
    "axes.edgecolor": BASE, "axes.linewidth": 1.0,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 11, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})

# per-scenario hue, (dark = diffusion, light = gaussian)
STYLE = {
    "value_up_30":      dict(label="Value +30%",    short="Val +30%", dark="#008300", light="#a6d9a6"),
    "value_down_30":    dict(label="Value −30%",    short="Val −30%", dark="#b50063", light="#eba8cb"),
    "momentum_up_30":   dict(label="Momentum +30%", short="Mom +30%", dark="#d95926", light="#f5bda4"),
    "momentum_down_30": dict(label="Momentum −30%", short="Mom −30%", dark="#4a3aa7", light="#beb7ec"),
}
PAGES = {"value": ["value_up_30", "value_down_30"],
         "momentum": ["momentum_up_30", "momentum_down_30"]}

GICS = {10: 'Energy', 15: 'Materials', 20: 'Industrials', 25: 'Consumer Discretionary',
        30: 'Consumer Staples', 35: 'Health Care', 40: 'Financials',
        45: 'Information Technology', 50: 'Communication Services',
        55: 'Utilities', 60: 'Real Estate'}

def style_axis(ax, xgrid=False, ygrid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y" if ygrid and not xgrid else "x", color=GRID, lw=1.0, alpha=1.0)
    ax.set_axisbelow(True)

# ---------------------------------------------------------------- data
print("loading data ...")
df = pd.read_parquet(ROOT / "data/downstream2025.parquet").dropna(subset=["mcap"])
df["gics_name"] = df["gics"].astype(int).map(GICS)

def scenario_matrix(pq):
    d = pd.read_parquet(pq)
    scn = d.groupby("csecid").cumcount()
    R = d.assign(scn=scn).pivot(index="scn", columns="csecid", values="returns")
    assert not R.isna().any().any()
    return R

def day_weights(day, w=None):
    if w is None:
        wd = (day["mcap"] / day["mcap"].sum()).to_numpy()
    else:
        wd = day["csecid"].map(w.set_index("csecid")["w"]).fillna(0).to_numpy()
        assert wd.sum() > 0
        wd = wd / wd.sum()
    return wd

days = {date: day for date, day in df.groupby("date")}

# realized forward monthly (22d compounded) portfolio return, mcap weights
_R = df.pivot_table(index="date", columns="csecid", values="returns")
_fwd = ((1 + _R.fillna(0)).rolling(22).apply(np.prod, raw=True) - 1).shift(-21)
_fwd = _fwd.iloc[:-21]
real_monthly = pd.Series({
    date: day_weights(day) @ _fwd.loc[date, day["csecid"]].fillna(0).to_numpy()
    for date, day in days.items() if date in _fwd.index}).sort_index()

print("loading scenario matrices ...")
labels = [s for page in PAGES.values() for s in page]
R_diff = {s: scenario_matrix(SAMPLES / f"{s}.parquet") for s in labels}
R_gaus = {s: scenario_matrix(SAMPLES / f"{s}_gaussian.parquet") for s in labels}

def pv_distribution(R, n_draw=50, seed=0):
    """per date: portfolio monthly return over a random subsample of scenarios.
    n_draw=200 keeps visible day-to-day sampling variation (a real desk would
    price a finite scenario set per day) instead of the dead-flat full mean."""
    rng = np.random.default_rng(seed)
    out = {}
    for date, day in days.items():
        wv = day_weights(day)
        pv = R[day["csecid"]].to_numpy() @ wv
        out[date] = rng.choice(pv, size=min(n_draw, len(pv)), replace=False)
    return pd.Series(out).sort_index()

def sector_contrib(R):
    """per sector: mean contribution to portfolio monthly return (over scenarios & dates)"""
    mean_ret = R.mean(axis=0)          # E[return] per stock over scenarios
    rows = []
    for date, day in days.items():
        wv = day_weights(day)
        c = pd.Series(wv * mean_ret[day["csecid"]].to_numpy(),
                      index=day["gics_name"].to_numpy()).groupby(level=0).sum()
        rows.append(c)
    return pd.concat(rows, axis=1).mean(axis=1)

print("computing pv distributions ...")
pv_diff = {s: pv_distribution(R_diff[s]) for s in labels}
pv_gaus = {s: pv_distribution(R_gaus[s]) for s in labels}
sec_diff = {s: sector_contrib(R_diff[s]) for s in labels}
sec_gaus = {s: sector_contrib(R_gaus[s]) for s in labels}

for s in labels:
    d = np.mean([v.mean() for v in pv_diff[s]]) * 100
    g = np.mean([v.mean() for v in pv_gaus[s]]) * 100
    r = real_monthly.mean() * 100
    print(f"{s:18s} diffusion {d:+6.2f}%  gaussian {g:+6.2f}%  (real {r:+.2f}%)")

# ---------------------------------------------------------------- fig 1: PV bands
def fig_pv_band(page, scens):
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    dates = real_monthly.index
    for s in scens:
        st = STYLE[s]
        for pv, color, ls, tag in ((pv_diff[s], st["dark"], "-", "Ours"),
                                   (pv_gaus[s], st["light"], "--", "Gaussian")):
            sub = pv.loc[dates]
            mean = np.array([v.mean() for v in sub])
            lw, alpha = (2.2, 1.0) if tag == "Ours" else (1.3, 0.8)
            ax.plot(dates, mean * 100, color=color, lw=lw, ls=ls, alpha=alpha,
                    label=f"{st['label']}  {tag}  ({mean.mean()*100:+.1f}%)")
    ax.plot(dates, real_monthly * 100, color=REAL, lw=1.6, alpha=0.9,
            label=f"Realized 2025  ({real_monthly.mean()*100:+.1f}%)")
    ax.axhline(0, color=BASE, lw=1.0)
    style_axis(ax)
    ax.set_ylabel("portfolio monthly return (%)")
    ax.set_title(f"Market portfolio under {PAGES_TITLE[page]} stress — mean of 50 scenarios per day")
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), handlelength=1.6)
    fig.tight_layout()
    fig.savefig(OUT / f"p_{page}_pvband.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

PAGES_TITLE = {"value": "Value ±30%", "momentum": "Momentum ±30%"}

# ---------------------------------------------------------------- fig 2: sector bars
def fig_sector(page, scens):
    order = (sum(sec_diff[s].abs() for s in scens)).sort_values().index  # big movers on top
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    y = np.arange(len(order))
    bh = 0.36
    for ax, s in zip(axes, scens):
        st = STYLE[s]
        vd = sec_diff[s].reindex(order).fillna(0) * 100
        vg = sec_gaus[s].reindex(order).fillna(0) * 100
        ax.barh(y + bh / 2 + 0.02, vd, height=bh, color=st["dark"], label="Ours")
        ax.barh(y - bh / 2 - 0.02, vg, height=bh, color=st["light"], label="Gaussian")
        ax.axvline(0, color=BASE, lw=1.0)
        style_axis(ax, xgrid=True)
        ax.set_title(st["short"], loc="left")
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2,
                  borderaxespad=0, handlelength=1.2, columnspacing=0.9)
        ax.set_xlabel("contribution to monthly return (pp)")
        # direct-label the biggest mover, outside the bar tip (with axis headroom)
        lo = min(vd.min(), vg.min(), 0.0)
        hi = max(vd.max(), vg.max(), 0.0)
        span = hi - lo
        ax.set_xlim(lo - 0.16 * span, hi + 0.16 * span)
        i = int(np.argmax(np.abs(vd.to_numpy())))
        v = vd.iloc[i]
        pad = 0.03 * span
        ax.text(v + (pad if v >= 0 else -pad), y[i] + bh / 2 + 0.02, f"{v:+.1f}",
                va="center", ha="left" if v >= 0 else "right",
                fontsize=11, color=INK, fontweight="semibold")
    axes[0].set_yticks(y, order)
    fig.suptitle(f"Where the P&L comes from — sector contribution under {PAGES_TITLE[page]}",
                 fontsize=14, fontweight="semibold", color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / f"p_{page}_sector.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------- fig 3: heatmap
def fig_heatmap():
    xlsx = pd.ExcelFile(ROOT / "data/portfolio_weights.xlsx")
    ws = {}
    for strat in xlsx.sheet_names:
        sheet = xlsx.parse(strat)
        for col in ("wpnew", "wb"):
            w_ = sheet[["csecid", col]].rename(columns={col: "w"}).dropna(subset=["w"])
            ws[f"{strat} {col}"] = w_[w_["w"] != 0]

    def active_stress(R, strat):
        mean_ret = R.mean(axis=0)
        vals = []
        for date, day in days.items():
            r = mean_ret[day["csecid"]].to_numpy()
            vals.append(day_weights(day, ws[f"{strat} wpnew"]) @ r
                        - day_weights(day, ws[f"{strat} wb"]) @ r)
        return np.mean(vals)

    def active_real(strat):
        vals = []
        for date, day in days.items():
            if date not in _fwd.index:
                continue
            r = _fwd.loc[date, day["csecid"]].fillna(0).to_numpy()
            vals.append(day_weights(day, ws[f"{strat} wpnew"]) @ r
                        - day_weights(day, ws[f"{strat} wb"]) @ r)
        return np.mean(vals)

    strategies = xlsx.sheet_names
    col_scens = ["value_up_30", "value_down_30", "momentum_up_30", "momentum_down_30"]
    print("computing heatmap table ...")
    cols = [[active_real(st) * 100 for st in strategies]]
    cols += [[active_stress(R_diff[s], st) * 100 for st in strategies] for s in col_scens]
    cols += [[active_stress(R_gaus[s], st) * 100 for st in strategies] for s in col_scens]
    vals = np.column_stack(cols)
    col_labels = ["Realized\n2025"] + [STYLE[s]["short"] for s in col_scens] * 2
    print(pd.DataFrame(vals.round(1), index=strategies, columns=range(9)).to_string())

    vmax = np.abs(vals).max()
    cmap = LinearSegmentedColormap.from_list("div", ["#b03a37", "#f2f1ee", "#1c5cab"])

    fig, ax = plt.subplots(figsize=(12.6, 3.6))
    ax.imshow(vals, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(vals.shape[1]), col_labels, fontsize=11)
    ax.set_yticks(range(len(strategies)), [s.title() for s in strategies], fontsize=12)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for x in (0.5, 4.5):
        ax.axvline(x, color="white", lw=5)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=12,
                    color="white" if abs(v) > 0.55 * vmax else INK,
                    fontweight="semibold" if abs(v) > 0.55 * vmax else "normal")
    # block headers
    ax.text(2.5, -0.72, "Ours", ha="center", fontsize=12,
            fontweight="semibold", color=INK)
    ax.text(6.5, -0.72, "Gaussian", ha="center", fontsize=12, color=INK2)
    ax.set_title("Stress test P&L — mean active monthly return, wpnew − wb (%)",
                 pad=36)
    fig.tight_layout()
    fig.savefig(OUT / "p_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

for page, scens in PAGES.items():
    print(f"figures for {page} ...")
    fig_pv_band(page, scens)
    fig_sector(page, scens)
fig_heatmap()
print("done ->", OUT)
