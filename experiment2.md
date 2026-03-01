### Data
[M, N, D] for m samples and n stocks and d days.
for stocknet M = 2048, N = 81, D = 22

Note:
1. W1(Wasserstein-1 Distance): computed on cross-sectional.
2. Mean/Std/Skew/Kurt: computed per asset and aggregated as mean(std).
3. Cov: Pull as  [M*D, N] and Frobenius norm of correlation matrix on each sample
4. ES:  Cumulative return on temporal, get ES per asset, aggregated as mean(std).




### Results


| Method | W1↓ | Cov↓ | ES↓ | Mean↓ | Std↓ | Skew↓ | Kurt↓ | LinUnpred↓ | VolClust↓ | CoarseFine↓ | 
|---|---|---|---|---|---|---|---|---|---|---|
| Stationary Bootstrap | 0.0000 | 0.0160 | -0.063±0.10 | +0.254±0.84 | +0.001±0.01 | +0.034±0.55 | -0.019±0.05 | -0.007 | +0.001 | -0.212  |
| MacroVAE | 0.0021 | 0.6190 | -0.300±0.21 | -0.231±2.42 | +0.150±0.07 | -0.743±4.44 | -0.483±0.34 | -0.067 | -0.072 | -0.719  |
---

<div style="text-align:center;">
  <img src="fig3_temporal_paths.png" alt="Temporal paths" style="max-width:60%; height:auto;">
  <p>Simulated cumulative return paths (300 of 2048 scenarios) for the highest-, median-, and lowest-volatility stocks.</p>
</div>

### Appendix: Metrics

Let $r_{m,n,t}$ denote the return of asset $n$ at day $t$ in scenario $m$. All metrics are computed per scenario per asset on the $D$-day series, then averaged over $M$ scenarios and $N$ assets. Reported as relative error $(generated - reference) / |reference|$.

- **Lin. Unpred. (Linear Unpredictability)**: serial autocorrelation of the return series $\{r_{m,n,t}\}_{t=1}^{D}$ at lags $k=1,\ldots,10$, averaged over lags, assets, and scenarios.

- **Vol Clust. (Volatility Clustering)**: serial autocorrelation of the absolute return series $\{|r_{m,n,t}|\}_{t=1}^{D}$ at lags $k=1,\ldots,10$, averaged over lags, assets, and scenarios.

- **Coarse-Fine**: cross-correlation between coarse and fine volatility measures at lag $k$, with rolling window $\tau = 5$ days.

$$\text{coarse}_t = \left|\sum_{s=t-\tau+1}^{t} r_s\right|, \quad \text{fine}_t = \sum_{s=t-\tau+1}^{t} |r_s|$$

$$\overline{\text{CF}} = \frac{1}{KN}\sum_{k=1}^{K}\sum_{n=1}^{N} \text{Corr}_m(\text{coarse}_{t+k},\, \text{fine}_t \mid n)$$

where $\text{Corr}_m$ averages the per-scenario correlation over $M$ scenarios.