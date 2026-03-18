def compute_moments(X):
    return {
        'mean': np.mean(X),
        'std': np.std(X),
        'skew': skew(X.flatten(), bias=False),
        'kurt': kurtosis(X.flatten(), bias=False)
    }

def frobenius_distance(A, B):
    return np.linalg.norm(A - B, ord='fro')

def compute_acf(data, nlags=10, is_synthetic=False):
    if is_synthetic and len(data.shape) == 3:
        n_samples, _, n_assets = data.shape
        all_acfs = []
        for asset in range(n_assets):
            period_acfs = []
            for sample in range(n_samples):
                period_data = data[sample, :, asset]
                if len(period_data) > nlags:
                    acf_vals = acf(period_data, nlags=nlags, fft=True)[1:]
                    period_acfs.append(acf_vals)
            if period_acfs:
                all_acfs.append(np.mean(period_acfs, axis=0))
        return np.mean(all_acfs, axis=0) if all_acfs else np.array([])
    else:
        return np.mean([acf(data[:, i], nlags=nlags, fft=True)[1:] for i in range(data.shape[1])], axis=0)

def leverage_calc(x, k_max):
    lev = []
    mean_r = np.mean(x)
    mean_r2 = np.mean(x ** 2)
    denom = mean_r2 ** 2 + 1e-8
    for k in range(1, min(k_max + 1, len(x))):
        term = x[:-k] * (x[k:] ** 2)
        numerator = np.mean(term) - mean_r * mean_r2
        lev.append(numerator / denom)
    return np.array(lev)

def leverage_effect(data, k_max=10, is_synthetic=False):
    if is_synthetic and len(data.shape) == 3:
        n_samples, _, n_assets = data.shape
        all_leverage = []
        for asset in range(n_assets):
            per_sample_lev = []
            for sample in range(n_samples):
                r = data[sample, :, asset]
                lev = leverage_calc(r, k_max)
                if len(lev) > 0:
                    per_sample_lev.append(lev)
            if per_sample_lev:
                all_leverage.append(np.mean(per_sample_lev, axis=0))
        return np.mean(all_leverage, axis=0) if all_leverage else np.zeros(k_max)
    elif len(data.shape) == 2:
        T, A = data.shape
        asset_leverage = [leverage_calc(data[:, a], k_max) for a in range(A)]
        return np.mean(asset_leverage, axis=0)

def coarse_fine_calc(X, tau, k_max):
    if len(X) < tau + k_max:
        return np.array([])
    coarse_vol = np.array([np.abs(np.sum(X[max(0, t - tau + 1):t + 1])) for t in range(tau - 1, len(X))])
    fine_vol = np.array([np.sum(np.abs(X[max(0, t - tau + 1):t + 1])) for t in range(tau - 1, len(X))])
    correlations = []
    for k in range(1, k_max + 1):
        if len(coarse_vol) > k and len(fine_vol) > k:
            coarse_shifted = coarse_vol[k:]
            fine_base = fine_vol[:-k]
            if len(coarse_shifted) == len(fine_base) and np.std(coarse_shifted) > 1e-8 and np.std(fine_base) > 1e-8:
                corr = np.corrcoef(coarse_shifted, fine_base)[0, 1]
                correlations.append(corr)
    return np.array(correlations)

def compute_coarse_fine_volatility(data, tau=5, k_max=10, is_synthetic=False):
    if is_synthetic and len(data.shape) == 3:
        n_samples, _, n_assets = data.shape
        all_correlations = []
        for asset in range(n_assets):
            period_correlations = []
            for sample in range(n_samples):
                X = data[sample, :, asset]
                corr = coarse_fine_calc(X, tau, k_max)
                if len(corr) > 0:
                    period_correlations.append(corr)
            if period_correlations:
                all_correlations.append(np.mean(period_correlations, axis=0))
        return np.mean(all_correlations, axis=0) if all_correlations else np.array([])
    else:
        asset_correlations = []
        for i in range(data.shape[1]):
            corr = coarse_fine_calc(data[:, i], tau, k_max)
            if len(corr) > 0:
                asset_correlations.append(corr)
        return np.mean(asset_correlations, axis=0) if asset_correlations else np.array([])

def compute_heavy_tails(data):
    flat_data = data.flatten()
    pos_data = flat_data[flat_data > 0]
    neg_data = -flat_data[flat_data < 0]
    alpha_pos = powerlaw.Fit(pos_data, verbose=False).alpha
    alpha_neg = powerlaw.Fit(neg_data, verbose=False).alpha
    return alpha_pos, alpha_neg

def compute_all_metrics(ref, test_returns=None, is_synthetic=False):
    """
    Compute all metrics
    """
    #WASSERSTEIN DISTANCE
    w_dist = wasserstein_distance(ref.flatten(), test_returns.flatten())

    #TEMPORAL METRICS (must split for synthetic)
    ref_acf = compute_acf(ref, nlags=10, is_synthetic=is_synthetic)
    ref_vol_clust = compute_acf(np.abs(ref), nlags=10, is_synthetic=is_synthetic)
    ref_leverage = leverage_effect(ref, k_max=10, is_synthetic=is_synthetic)
    ref_coarse_fine = compute_coarse_fine_volatility(ref, tau=5, k_max=10, is_synthetic=is_synthetic)

    #DISTRIBUTIONAL METRICS (can combine - always flatten)
    ref_flat = ref.reshape(-1, ref.shape[-1]) if len(ref.shape) == 3 else ref
    ref_moments = compute_moments(ref_flat)
    ref_corr = np.corrcoef(ref_flat.T)
    off_diag_mask = ~np.eye(ref_corr.shape[0], dtype=bool)
    ref_alpha_pos, ref_alpha_neg = compute_heavy_tails(ref_flat)

    result = {
        # Wasserstein distance
        'w_dist': w_dist,

        # Temporal metrics
        'acf_mean': np.mean(ref_acf),
        'vol_clust_mean': np.mean(ref_vol_clust),
        'leverage_mean': np.mean(ref_leverage),
        'coarse_fine_mean': np.mean(ref_coarse_fine),

        # Distributional metrics
        'mean': ref_moments['mean'],
        'std': ref_moments['std'],
        'skew': ref_moments['skew'],
        'kurt': ref_moments['kurt'],
        'avg_correlation': np.mean(ref_corr[off_diag_mask]),
        'alpha_pos': ref_alpha_pos,
        'alpha_neg': ref_alpha_neg,
    }
    return result