import pickle
import datetime
import numpy as np
import pandas as pd
import torch
import powerlaw
from dateutil.relativedelta import relativedelta
import matplotlib
matplotlib.use('Qt5Agg')
import os
from scipy.stats import wasserstein_distance, skew, kurtosis
from statsmodels.tsa.stattools import acf
from utility import standardize, select_empirical_dataset
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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


#Load Data
with open("data.pickle", 'rb') as f:
    dataset, macro_dataset, dates = pickle.load(f)

data_df = pd.read_excel('Futures Data.xlsx', index_col=[0])
data_df = data_df.dropna().sort_index()
data_df = data_df.loc[~data_df.index.weekday.isin([5,6])]
returns_df = data_df.pct_change(periods=1)

latent_dim = 32
batch_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

returns_df_month_starts = returns_df.loc[[True] + (returns_df.index.month[:-1] !=
                                                   returns_df.index.month[1:]).tolist()].index

all_results = {'test': {}, 'macrovae': {}, 'hist_1m': {}, 'hist_1y': {}, 'all_hist': {}}

#Main Evaluation Loop
for end_year in range(2014, 2025):
    print(end_year)
    months = [12] if end_year == 2005 else [3, 6, 9, 12]

    for month in months:
        print(month)
        train_start = datetime.date(2000, 1, 1)
        train_end = datetime.date(end_year, month + 1, 1) - relativedelta(days=1) \
            if month < 12 else datetime.date(end_year, 12, 31)

        train_data, train_macro, train_dates = select_empirical_dataset(train_start, train_end, dates,
                                                                        dataset, macro_dataset)

        macro_array = np.array(train_macro)
        standardized_macro, macro_means, macro_stds = standardize(macro_array)

        model_file = f"models//macro_vae_{end_year}_{month}.pt" if month < 12 else f"models//macro_vae_{end_year}.pt"
        model = torch.load(model_file, map_location=device, weights_only=False)
        model.eval()

        start_dates = returns_df_month_starts[returns_df_month_starts > pd.to_datetime(train_end)][:3]

        for d in start_dates:

            t_idx = [i for (i, data) in enumerate(dataset) if data.index[0] >= d]
            if len(t_idx) > 0:
                t_idx = t_idx[0]
                test_returns = dataset[t_idx].values

                test_macro = macro_dataset[t_idx]
                standardized_test_macro = (np.array(test_macro) - macro_means) / macro_stds
                macro_tensor = torch.tensor(standardized_test_macro.reshape(1, 36), dtype=torch.float32)

                macro_cond = macro_tensor.repeat(batch_size * 2, 1).to(device)
                z = torch.randn(batch_size * 2, latent_dim).to(device)

                with torch.no_grad():
                    norm_ret, vol, mean = model.decode(z, macro_cond)
                    vol = vol[:, 0, :].unsqueeze(1)
                    samples = (norm_ret + mean) * torch.exp(vol)

                gen_returns = samples.cpu().numpy()

                last_test_d = dataset[t_idx].index[-1]

                # Historical returns
                hist_1y = returns_df.loc[returns_df.index < train_dates[-1]].iloc[-260:].values
                hist_1m = returns_df.loc[returns_df.index < train_dates[-1]].iloc[-22:].values
                all_train_data = returns_df.loc[(returns_df.index >= train_data[0].index[0])
                                                & (returns_df.index < train_dates[-1])].values
                all_hist_metrics = compute_all_metrics(all_train_data, test_returns, is_synthetic=False)

                all_results['test'][last_test_d] = compute_all_metrics(test_returns, test_returns, is_synthetic=False)
                all_results['macrovae'][last_test_d] = compute_all_metrics(gen_returns, test_returns, is_synthetic=True)
                all_results['hist_1m'][last_test_d] = compute_all_metrics(hist_1m, test_returns, is_synthetic=False)
                all_results['hist_1y'][last_test_d] = compute_all_metrics(hist_1y, test_returns, is_synthetic=False)
                all_results['all_hist'][last_test_d] = compute_all_metrics(all_train_data, test_returns,
                                                                           is_synthetic=False)

for r in all_results.keys():
    pd.DataFrame().from_dict(all_results[r]).T.to_excel(r + '.xlsx')
