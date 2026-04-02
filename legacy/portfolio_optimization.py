import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from risk_management import minimum_variance_weights


def efficient_frontier(mu, cov, n_points=50):
    """Compute long-only efficient frontier.
    Returns (vols, rets, weights_list, max_sharpe_idx).
    """
    d = len(mu)
    scale = 1.0 / np.median(np.diag(cov))
    cov_s = cov * scale
    w0 = np.ones(d) / d

    w_mv = minimum_variance_weights(cov)
    min_ret = mu @ w_mv
    max_ret = mu.max()

    targets = np.linspace(min_ret, max_ret * 0.9, n_points)
    frontier = []

    for target in targets:
        res = minimize(
            lambda w: w @ cov_s @ w, w0,
            jac=lambda w: 2 * cov_s @ w,
            method='SLSQP',
            bounds=[(0, 1)] * d,
            constraints=[
                {'type': 'eq', 'fun': lambda w: w.sum() - 1},
                {'type': 'ineq', 'fun': lambda w, t=target: mu @ w - t},
            ],
        )
        if res.success:
            w_opt = res.x
            frontier.append((np.sqrt(w_opt @ cov @ w_opt), mu @ w_opt, w_opt))

    vols = np.array([f[0] for f in frontier])
    rets = np.array([f[1] for f in frontier])
    weights = [f[2] for f in frontier]

    sharpes = rets / (vols + 1e-10)
    ms_idx = int(np.argmax(sharpes))

    return vols, rets, weights, ms_idx


if __name__ == "__main__":
    training = np.load("data/stocknet_81.npy")
    factor_dm = np.load("samples/factorDM2048.npy")
    bootstrap = np.load("samples/SB2048.npy")

    datasets = {
        'Historical (GT)': training,
        'FactorDM': factor_dm,
        'Stationary Bootstrap': bootstrap,
    }

    # Estimate mu, cov from each data source
    params = {}
    for name, data in datasets.items():
        params[name] = (np.mean(data, axis=0), np.cov(data.T))

    mu_gt, cov_gt = params['Historical (GT)']

    # GT efficient frontier (for reference curve)
    gt_vols, gt_rets, _, _ = efficient_frontier(mu_gt, cov_gt)

    # For each method: optimize on its own params, evaluate on GT
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gt_vols, gt_rets, 'k-', lw=1.5, alpha=0.5, label='GT Efficient Frontier')

    colors = {'Historical (GT)': 'black', 'FactorDM': 'tab:blue', 'Stationary Bootstrap': 'tab:orange'}
    markers = {'Historical (GT)': '*', 'FactorDM': 'o', 'Stationary Bootstrap': 's'}
    sizes = {'Historical (GT)': 200, 'FactorDM': 100, 'Stationary Bootstrap': 100}

    for name in datasets:
        mu, cov = params[name]
        _, _, weights, ms_idx = efficient_frontier(mu, cov)
        w_star = weights[ms_idx]

        # Evaluate on GT parameters
        actual_ret = mu_gt @ w_star
        actual_vol = np.sqrt(w_star @ cov_gt @ w_star)
        actual_sharpe = actual_ret / (actual_vol + 1e-10)

        ax.scatter(actual_vol, actual_ret, c=colors[name], marker=markers[name],
                   s=sizes[name], zorder=5,
                   label=f'{name} (Sharpe={actual_sharpe:.3f})')

        print(f"{name}: Vol={actual_vol:.4f}, Ret={actual_ret:.6f}, Sharpe={actual_sharpe:.3f}")

    ax.set_xlabel('Portfolio Volatility')
    ax.set_ylabel('Portfolio Expected Return')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('assets/portfolio_optimization.png', dpi=150, bbox_inches='tight')
    print("Saved to assets/portfolio_optimization.png")
