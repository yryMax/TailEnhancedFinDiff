### Something to agree first
We have the temporal relations of assets returns + cross-sectional relations of assets returns.

We want to sample some cross-sectional returns s.t. 

1. the correlation structure is maintained. (unconditional sampling)

2. when given some conditions that are related to (extreme)regime, the sampled cross-sectional returns behave as under such regime. (conditional sampling)

We assume asset pool to be fixed(i.e, only to be able to generate assets that we know).

**NOTE**: (1) is somehow achieved by using diffusion model and seen the cross-sectional returns as muti-variate RV, 
it is better to have our novelty while achieving (2). And I decided go to with the conditional idea, because it is more useful in practice and easier to evaluate. If
things don't go well, the unconditional idea can be a backup plan.


### Condition Modeling / Regime Modeling / Tail Condition Modeling
Based on whether the denoiser depends on the condition, we can
divide the conditional sampling strategies into two categories: Building-in conditioning and Post-hoc conditioning.
Building-in conditioning learns p(x|c) directly; the condition is learned via a joint NN and (often) doesn't need labeling.
Post-hoc conditioning learns p(x) and then adjusts the sampling process to get p(x|c), condition is often learned via a separate NN or needs a supervised label. The training process is not involved or partially involved. 

Based on different ways to model the condition and how the condition is involved, there are various implementations for both sub-categories.

#### Building-in conditioning
- [class-conditioning](https://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf): concat based.
- [AdaGN conditioning](https://arxiv.org/abs/2105.05233): adjust the weights.
- [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598): require supervised learning(labeling the data), and are hard to generate to unseen conditions.


#### Post-hoc conditioning
Soft guidance that have continuous panalty during the sampling process
- [Classifier guidance](https://arxiv.org/abs/2105.05233?): train a separate classifier on condition and use their gradient to penalize the sampling gradient.
- [loss-based guidance](https://proceedings.mlr.press/v202/song23k.html)
- [Universal Guidance](https://arxiv.org/pdf/2302.07121)
- [Energy-based Guidance](https://arxiv.org/abs/2302.11552)

Hard guidance that substitute (part-of) the intermediate samples.
- [ILVR](https://arxiv.org/abs/2108.02938?)
- [RePaint](https://arxiv.org/abs/2201.09865)
- [SDEdit](https://arxiv.org/abs/2108.01073)

#### Others
- [Diffusion Posterior Sampling (DPS)](https://arxiv.org/abs/2209.14687)

- [Score Distillation Sampling (SDS)](https://arxiv.org/abs/2209.14988)




### Factor Asset Pricing Model
- **Definition**
  - $$\mathbb{E}[R_i] = R_f + {\beta}_i^{\top} {\lambda}$$
  - $$R_i - R_f = {\beta}_i^{\top} \mathbf{f} + \varepsilon_i$$
- **Where:**
  - $R_i$: return on asset $i$ over the period.
  - $R_f$: risk-free rate over the same period.
  - $\mathbb{E}[R_i]$: expected return on asset $i$.
  - ${\beta}_i$: vector of factor loadings (sensitivities) for asset $i$.
  - ${\lambda}$: vector of factor risk premia (expected return per unit of factor risk).
  - $\mathbf{f}$: vector of realized factor returns in the period.
  - $\varepsilon_i$: idiosyncratic return component (asset-specific, mean zero).






There have been enough evidence that latent factors exists
(means there is an interpolatable and continuous representation that expresses the correlation between assets), 
for example
[IPCA](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2983919)

### [FactorVAE](https://ojs.aaai.org/index.php/AAAI/article/view/20369)




