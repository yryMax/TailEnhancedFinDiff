import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# fixed color map: column name -> color, built on first use
_COLOR_CACHE: dict[str, str] = {}

def _get_color(col: str) -> str:
    if col == 'intercept':
        return '#222222'
    if col not in _COLOR_CACHE:
        idx = len(_COLOR_CACHE)
        _COLOR_CACHE[col] = cm.tab20(idx % 20)
    return _COLOR_CACHE[col]


def plot_cumulative(F: pd.DataFrame, title: str = 'Cumulative Returns') -> None:
    cols = sorted(F.columns)
    colors = [_get_color(c) for c in cols]
    fig, ax = plt.subplots(figsize=(12, 4))
    (1 + F[cols]).cumprod().plot(ax=ax, color=colors)
    ax.axhline(1, color='black', linewidth=0.5, linestyle='--')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def summarize_factors(F: pd.DataFrame) -> pd.DataFrame:
    T = len(F)
    t_stats = F.mean() / (F.std() / np.sqrt(T))

    summary = pd.DataFrame({
        'mean_annual(%)': (F.mean() * 252 * 100).round(2),
        'vol_annual(%)':  (F.std() * np.sqrt(252) * 100).round(2),
        't_stat':         t_stats.round(2),
        'p_value':        t_stats.apply(lambda t: 2 * stats.t.sf(abs(t), df=T - 1)).round(4),
        'significant':    t_stats.abs() > 2,
    })

    plot_cumulative(F, title='Cumulative Factor Returns')

    return summary
