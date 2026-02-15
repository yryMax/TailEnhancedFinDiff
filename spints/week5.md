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
- [class-conditioning](https://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf): maps class labels through a learned embedding, adds it to the timestep embedding, and injects the combined signal into each UNet residual block via addition or scale-and-shift modulation.
- [AdaGN conditioning](https://arxiv.org/abs/2105.05233): replaces standard norm with Adaptive Group Normalization, weighted by the condition.
- [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598): require supervised learning(labeling the data), and are hard to generate to unseen conditions.

#### Post-hoc conditioning
Soft guidance that have continuous panalty during the sampling process
- [Classifier guidance](https://arxiv.org/abs/2105.05233?): train a separate classifier on condition and use their gradient to penalize the sampling gradient.
- [loss-based guidance (LGD)](https://proceedings.mlr.press/v202/song23k.html): MC sampling to approximate the non-integrable guidance.
- [Energy-based Guidance](https://arxiv.org/abs/2302.11552): uses energy-based parameterization and MCMC-inspired samplers for compositional guidance.
- [Universal Guidance](https://arxiv.org/abs/2302.07121): enables arbitrary guidance modalities (segmentation, face recognition, object detection, etc.) to control diffusion models without retraining any use-specific components.
- [Diffusion Posterior Sampling (DPS)](https://arxiv.org/abs/2209.14687): soft manifold-constrained gradient guide


Hard guidance that substitute (part-of) the intermediate samples.
- [ILVR](https://arxiv.org/abs/2108.02938?): at each reverse step, replaces the low-frequency component of intermediate samples with that of a reference image.
- [RePaint](https://arxiv.org/abs/2201.09865): at each reverse step, replaces the known (unmasked) region with forward-diffused known pixels and keeps only the unknown region from the model.
- [SDEdit](https://arxiv.org/abs/2108.01073): adds noise to a user-provided guide image then denoises through the learned SDE prior.

#### Others
- [Score Distillation Sampling (SDS)](https://arxiv.org/abs/2209.14988):  it uses a pretrained conditional diffusion model’s denoising score as a differentiable training signal to push the outputs of any parameterized generator toward the model’s conditional distribution



### Factor Asset Pricing Model


#### Definition
The first equation describes the expected excess return of an asset, while the second equation describes the realized excess return.
$$\mathbb{E}[R_{i,t} - R_{f,t}] = \alpha_i + \beta_{i,t}^\top \lambda_t$$
$$R_{i,t} - R_{f,t} = \alpha_i + \beta_{i,t}^\top f_t + \varepsilon_{i,t}$$
where:
- $R_{i,t}$: The return of asset $i$ at time $t$. 
- $R_{f,t}$: The risk-free rate at time $t$.
- $\alpha_i$: The abnormal or idiosyncratic expected return of asset $i$.
- $\beta_{i,t} \in \mathbb{R}^{K}$: The factor exposures (or loadings) of asset $i$ to $K$ risk factors at time $t$.
- $\lambda_t \in \mathbb{R}^{K}$: The factor risk premia at time $t$. 
- $f_t \in \mathbb{R}^{K}$: The factor returns at time $t$. 
- $\varepsilon_{i,t}$: The idiosyncratic return (or error term) for asset $i$ at time $t$.

#### The Prior
There have been enough evidence that latent factors exists (means there is an interpolatable and continuous representation that expresses the correlation between assets), for example [IPCA](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2983919) shows that under a structured conditional factor model where characteristics instrument time-varying loadings, the cross-section of returns admits an accurate low-rank representation.

- $R_{i,t}$ can be decomposed into a risk-free rate, systematic (factor-driven) components, and idiosyncratic noise. Empirically, returns exhibit non-normal characteristics like fat tails, skewness, and volatility clustering.

- The risk-free rate $R_{f,t}$ is typically treated as an observable. While time-varying, its dynamics are stable and predictable and reluctant to short term shocks.

- The factor exposure $\beta_{i,t}$ is asset-specific and assumed to be persistent or weekly dependent on t. But could be influenced by the regime change.
Theoretically it is independent of other assets.

- abnormal return $\alpha_i$ is supposed to be zero. A central goal of asset pricing is to test this null hypothesis.

- The factor return $f_t$ is the realized outcome of a stochastic process, representing the "event" itself. It is treated as an observable input or regressor in the time-series pass of model estimation.

- The factor risk premium $\lambda_t$ represents the unobservable "price" of risk. it is estimated from the cross-section of returns after the factor loadings are determined.
Theoratically, it is unrelated to individual stocks and only related to market conditions.
- $\varepsilon_{i,t}$ is assumed to be uncorrelated with the factors and have a mean of zero. Often parameterized by gaussian distribution.


### Literature
All the papers are about design choice of 1. assume some prior. 2. decide the condition. 3. decide the conditioning sampling strategy.

| Model                                         | Condition                         | lambda_t             | alpha                   | beta_t                   | Sampling strategy  | Prior                                              |
|-----------------------------------------------|-----------------------------------|----------------------|-------------------------|--------------------------|-------------------|----------------------------------------------------|
| [FactorDM](https://arxiv.org/abs/2504.06566)  | None                              | extracted by U-net   | 0                       | orthogonal parameterized | Unconditional     | factor and factorloader should be separated        |
| [FactorVAE](https://arxiv.org/abs/1802.05983) | temporal_f_t                      | VAE latent           | parameterized Gaussian  | extracted by GNU         | class-conditional | the factor asset pricing fomula                    |
| [FactorDiff](https://arxiv.org/pdf/2509.22088)                                | Cross-sectional factor loader     | N/A                  | N/A                     | extracted MLP/Attention  | AdaLN             | IPCA is indicator of beta                          |

Explainability by factor model: FactorVAE > FactorDM > FactorDiff
