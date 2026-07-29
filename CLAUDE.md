# DLPM Forward Process: Step-by-Step Derivation

## Step 1: Single Step $x_{t-1} \to x_t$

| | DDPM                                                                      | DLPM                                                                        |
|---|---------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| Noise type | Gaussian $z_t \sim \mathcal{N}(0,I)$                                      | SαS $\varepsilon_t \sim S(\alpha_{\mathrm{levy}},\, 0)$                     |
| Single step | $x_t = \sqrt{\alpha_t}\,x_{t-1} + \sqrt{\beta_t}\,z_t$ s.t $\alpha_t + \beta_t = 1$ | $x_t = \gamma_t\,x_{t-1} + \sigma_t\,\varepsilon_t$ s.t $\gamma_t^{\alpha} + \sigma_t^{\alpha} = 1$                    |
| Conditional distribution | $x_t \mid x_{t-1} \sim \mathcal{N}(\sqrt{\alpha_t}\,x_{t-1},\ \beta_t I)$ | $x_t \mid x_{t-1},A_t \sim \mathcal{N}(\gamma_t\,x_{t-1},\ \sigma_t^2 A_t)$ |

In DLPM, the SαS noise $\varepsilon_t$ is represented as a **Gaussian scale mixture**:

$$\varepsilon_t = \sqrt{A_t}\cdot z_t, \qquad A_t \sim S(\alpha_{\mathrm{levy}}/2,\,1),\quad z_t \sim \mathcal{N}(0, I)$$

Conditioned on the subordinator $A_t$, the SαS step becomes Gaussian — this is what makes the posterior tractable.

---

## Step 2: Preserving Constraint

|                    | DDPM | DLPM |
|--------------------|---|---|
| Preserved quantity | $\mathrm{Var}(x_t) = 1$ | SαS scale $= 1$ |
| Constraint         | $\gamma_t^2 + \sigma_t^2 = 1$ | $\gamma_t^{\alpha_{\mathrm{levy}}} + \sigma_t^{\alpha_{\mathrm{levy}}} = 1$ |
| std                | $\sqrt{\beta_t}$ | $\beta_t^{1/\alpha_{\mathrm{levy}}}$ |

**Deriving $\sigma_t$ in DLPM.** SαS scales combine via $\alpha$-norm rather than variance addition:

$$x_t = \gamma_t x_{t-1} + \sigma_t \varepsilon_t \implies \text{scale}(x_t) = (\gamma_t^{\alpha_{\mathrm{levy}}} + \sigma_t^{\alpha_{\mathrm{levy}}})^{1/\alpha_{\mathrm{levy}}}$$

Setting scale $= 1$ and substituting $\gamma_t^{\alpha_{\mathrm{levy}}} = \alpha_t$:

$$\boxed{\sigma_t = (1-\alpha_t)^{1/\alpha_{\mathrm{levy}}} = \beta_t^{1/\alpha_{\mathrm{levy}}}}$$

This is `sigmas = (1 - gammas**alpha)**(1/alpha)` in code. When $\alpha_{\mathrm{levy}}=2$ this reduces to $\sqrt{\beta_t}$, recovering DDPM.

---

## Step 3: Unrolling to $x_0$

Unrolling $t$ steps in both cases:

| | DDPM | DLPM |
|---|---|---|
| Unrolled | $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sum_{s=1}^t \frac{\sqrt{\bar\alpha_t}}{\sqrt{\bar\alpha_s}}\sqrt{\beta_s}\,z_s$ | $x_t = \bar\gamma_t\,x_0 + \sum_{s=1}^t \frac{\bar\gamma_t}{\bar\gamma_s}\,\sigma_s\sqrt{A_s}\,z_s$ |
| Cumulative signal | $\sqrt{\bar\alpha_t} = \prod\sqrt{\alpha_s}$ | $\bar\gamma_t = \prod \gamma_s$ |

**Conditioned on $A_{1:t}$**, the noise sum in DLPM is a sum of independent Gaussians:

$$\boxed{x_t \mid x_0,\, A_{1:t} \;\sim\; \mathcal{N}\!\left(\bar\gamma_t\, x_0,\;\Sigma_t\right)}$$

The variance $\Sigma_t$ satisfies the recurrence (expanding the sum of variances):

| | DDPM | DLPM |
|---|---|---|
| Variance recurrence | $1-\bar\alpha_t = \beta_t + \alpha_t(1-\bar\alpha_{t-1})$ | $\Sigma_t = \sigma_t^2 A_t + \gamma_t^2\,\Sigma_{t-1}$ |
| Nature of variance | deterministic (fixed by schedule) | **stochastic** (depends on random $A_t$) |

---

## Step 4: Marginal over $A_{1:t}$

Integrating out the subordinators using the SαS stability property (SαS scales add in $\alpha$-norm):

| | DDPM | DLPM |
|---|---|---|
| Cumulative noise scale | $\sqrt{1-\bar\alpha_t}$ | $\bar\sigma_t = (1-\bar\gamma_t^\alpha)^{1/\alpha}$ |
| Marginal | $x_t \mid x_0 \sim \mathcal{N}(\sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I)$ | $x_t \mid x_0 = \bar\gamma_t\,x_0 + \bar\sigma_t\,\varepsilon,\quad \varepsilon \sim S(\alpha_{\mathrm{levy}},\, 0)$ |
| Scale constraint | $\bar\alpha_t + (1-\bar\alpha_t) = 1$ | $\bar\gamma_t^\alpha + \bar\sigma_t^\alpha = 1$ |

DLPM's marginal is a direct generalization: $\sqrt{\cdot}$ in DDPM becomes $(\cdot)^{1/\alpha}$ in DLPM.
When $\alpha_{\mathrm{levy}}=2$: $A_t \equiv 1$, $\bar\sigma_t = \sqrt{1-\bar\alpha_t}$, and DLPM reduces exactly to DDPM.

---

# DLPM vs DDPM: Notation Correspondence

## Forward Process

**DDPM single step.** The forward process is defined as:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(\sqrt{\alpha_t}\, x_{t-1},\ \beta_t I\right), \qquad \alpha_t = 1 - \beta_t$$

The signal coefficient is $\sqrt{\alpha_t}$ and the noise variance is $\beta_t$.

**DDPM marginal.** Unrolling $t$ steps (by induction, using independence of noise):

$$q(x_t \mid x_0) = \mathcal{N}\!\left(\sqrt{\bar\alpha_t}\, x_0,\ (1 - \bar\alpha_t)\, I\right), \qquad \bar\alpha_t = \prod_{s=1}^t \alpha_s$$

**How DLPM's $\gamma_t, \sigma_t$ relate to $\alpha_t, \beta_t$.** DLPM keeps the same $\alpha_t = 1 - \beta_t$ from the noise schedule, but redefines the single-step coefficients to preserve SαS structure:

$$\gamma_t = \alpha_t^{1/\alpha_{\mathrm{levy}}}, \qquad \sigma_t = (1 - \gamma_t^{\alpha_{\mathrm{levy}}})^{1/\alpha_{\mathrm{levy}}} = \beta_t^{1/\alpha_{\mathrm{levy}}}$$

When $\alpha_{\mathrm{levy}} = 2$: $\gamma_t = \sqrt{\alpha_t}$ and $\sigma_t = \sqrt{\beta_t}$, recovering DDPM exactly.

The cumulative versions follow by taking products ($\bar\gamma_t = \prod \gamma_t$), giving:

$$\bar\gamma_t = \bar\alpha_t^{1/\alpha_{\mathrm{levy}}}, \qquad \bar\sigma_t = (1 - \bar\gamma_t^{\alpha_{\mathrm{levy}}})^{1/\alpha_{\mathrm{levy}}} = (1-\bar\alpha_t)^{1/\alpha_{\mathrm{levy}}}$$

So the DLPM marginal $x_t = \bar\gamma_t x_0 + \bar\sigma_t \varepsilon$ is a direct generalization of the DDPM marginal, with $\sqrt{\cdot}$ replaced by $(\cdot)^{1/\alpha_{\mathrm{levy}}}$ everywhere.

---

## Correspondence Table

| DLPM variable | DDPM equivalent | Meaning |
|---|---|---|
| `gammas[t]` $= \gamma_t$ | $\sqrt{\alpha_t} = \sqrt{1-\beta_t}$ | Single-step signal retention |
| `bargammas[t]` $= \bar\gamma_t$ | $\sqrt{\bar\alpha_t}$ | Cumulative signal retention |
| `sigmas[t]` $= \sigma_t$ | $\sqrt{\beta_t}$ | Single-step noise scale |
| `barsigmas[t]` $= \bar\sigma_t$ | $\sqrt{1 - \bar\alpha_t}$ | Cumulative noise scale |
| `Sigma_t` | $1 - \bar\alpha_t$ | Forward marginal variance |
| `Gamma_t` | $\beta_t\, /\, (1 - \bar\alpha_t)$ | Posterior contraction ratio |
| `var = Gamma_t * Sigma_{t-1}` | $\beta_t(1-\bar\alpha_{t-1})/(1-\bar\alpha_t)$ | Posterior variance |

---

## Deriving `Gamma_t`

`Gamma_t` is defined as:

$$\Gamma_t = 1 - \frac{\gamma_t^2\, \Sigma_{t-1}}{\Sigma_t}$$

Substituting DDPM quantities ($\gamma_t^2 = \alpha_t$, $\Sigma_t = 1 - \bar\alpha_t$):

$$\Gamma_t = 1 - \frac{\alpha_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t} = \frac{(1-\bar\alpha_t) - \alpha_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}$$

The numerator simplifies using $\alpha_t \bar\alpha_{t-1} = \bar\alpha_t$:

$$= \frac{1 - \bar\alpha_t - \alpha_t + \bar\alpha_t}{1-\bar\alpha_t} = \frac{\beta_t}{1 - \bar\alpha_t}$$

---

## Posterior Distribution

Both DDPM and DLPM share the same posterior form (conditioned on $A$ in DLPM):

$$q(x_{t-1} \mid x_t, x_0, A) = \mathcal{N}(\mu_t,\ \underbrace{\Gamma_t \cdot \Sigma_{t-1}}_{\tilde\sigma_t^2})$$

**Posterior variance** (`var` in code):

$$\tilde\sigma_t^2 = \Gamma_t \cdot \Sigma_{t-1} = \frac{\beta_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}$$

**Posterior mean** (`mean` in code):

$$\mu_t = \frac{x_t - \bar\sigma_t \cdot \Gamma_t \cdot \hat\varepsilon}{\gamma_t} = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\hat\varepsilon\right)$$

The right-hand side is the standard DDPM posterior mean formula.

---

## The Key Difference: Stochastic vs Deterministic Variance

In DDPM, $1 - \bar\alpha_t$ is **deterministic** — fixed by the noise schedule, the same for every sample.

In DLPM, $\Sigma_t$ is **stochastic** — it accumulates the Lévy subordinators $A_t \sim S(\alpha_{\mathrm{levy}}/2,\, 1)$, which are different random draws for each sample:

$$\Sigma_t = \sigma_t^2 \underbrace{A_t}_{\text{random}} + \gamma_t^2\, \Sigma_{t-1}$$

This means each sample travels a different "effective noise path" through the diffusion. The randomness in $\Sigma_t$ (and therefore $\Gamma_t$) is what gives DLPM its heavy-tailed marginal distribution — when $\alpha_{\mathrm{levy}} < 2$, the subordinators $A_t$ can be very large, producing samples far in the tails.

When $\alpha_{\mathrm{levy}} = 2$: $A_t \equiv 1$ always, so $\Sigma_t$ becomes deterministic and DLPM degenerates exactly to DDPM.

---

## Conditional Guidance vs (Soft) Rejection

We often want to bias unconditional samples toward a constraint encoded by an energy (loss) function $L(x_0)$ and a strength parameter $s>0$.

Target distribution in data space:

$$p^*(x_0) \propto p(x_0)\, \exp\big(-s\,L(x_0)\big).$$

Two practical routes implemented in code:

1) Soft rejection sampling (exact for the above target)

- Draw $x_0 \sim p(x_0)$; accept with prob. $\exp(-s L(x_0))$.
- The accepted samples are distributed exactly as $p^*(x_0)$ (up to Monte Carlo error).
- In code: `generate_rejection(..., hard=False, guidance_scale=s)`.

2) Gradient guidance inside reverse process (approximate)

- At step $t$, with Tweedie estimate $\hat x_0 = (x_t - \bar\sigma_t\,\hat\varepsilon_t)/\bar\gamma_t$, shift the posterior mean by
  $$\mu_t \leftarrow \mu_t - s \cdot \mathrm{var}_t \cdot \nabla_{x_t} L(\hat x_0), \qquad \mathrm{var}_t = \Gamma_t\,\Sigma_{t-1}.$$
- This mirrors classifier/energy guidance in DDPMs, generalized to DLPM with the Bayesian scaling $\mathrm{var}_t$.
- In code: `mean -= guidance_scale * var * grad` inside `generate()`.

Why guidance is an approximation (and differs from soft rejection)

- Missing conditional expectation: the exact drift that yields $p^*_t(x_t) \propto p_t(x_t)\, \mathbb{E}[\exp(-sL(x_0))\mid x_t]$ adds
  $$\nabla_{x_t} \log \mathbb{E}[\exp(-sL(x_0))\mid x_t].$$
  The implementation replaces the intractable conditional expectation by a plug‑in $L(\hat x_0)$ and uses $\nabla_{x_t} L(\hat x_0)$.
- Detaching the model path: $\hat x_0$ depends on $\hat\varepsilon_t=\mathrm{model}(x_t,t)$. The code computes gradients w.r.t. $x_t$ while treating $\hat\varepsilon_t$ as constant (detached), dropping the $\partial \hat\varepsilon_t/\partial x_t$ term for stability and cost.
- Discrete-time integration: guidance uses finite diffusion steps (Euler-like updates). Even with the exact drift, discretization introduces bias unless $T$ is large or higher-order solvers are used.
- Heavy-tail mixture geometry: for $\alpha<2$, the Gaussian-mixture posterior is handled via a single variance factor $\mathrm{var}_t$; this is a principled scaling but still an approximation of the exact energy-shaped reverse dynamics.

When do guidance and soft rejection get close?

- Small $s$ (weak bias), smooth $L$, well-trained denoiser (Tweedie close to Bayes), large $T$ (small step size), and at late steps where $\bar\sigma_t\!\to 0$ so $\hat x_0\!\approx x_0$. In these regimes, guidance statistics (means/quantiles) tend to align with soft rejection, but they are not guaranteed to match exactly.

Hard rejection vs soft rejection

- `hard=True` approximates sampling from the truncated distribution $p(x_0\mid L(x_0)=0)$ (subject to tolerance); it is not the same as $p^*(x_0)$ unless $s\to\infty$ and $L$ defines a hard constraint.
- `hard=False` implements the exact energy reweighting $p^*(x_0)$ for the chosen $s$.

Practical tips for matching behavior

- Use soft rejection as a reference to calibrate `guidance_scale` (match target means/quantiles on the constrained coordinates).
- Prefer smooth $L$ (e.g., Huber/softplus) to reduce discretization artifacts at kinks.
- Consider time‑dependent weights (proportional to $\mathrm{var}_t$ is already applied; optionally taper $s$ early vs late time).
- Increase $T$ or use better samplers (e.g., ODE/DPM-style solvers) if guidance bias is large.
- If compute allows, remove `detach` for diagnostics to gauge the impact of the dropped term (may be unstable; keep as an experiment).

Code waypoints

- Posterior mean/variance: `factor_diffusion_sample.generate()`.
- Guidance update: mean shift by `guidance_scale * var * grad` inside `generate()`.
- Soft/hard rejection: `factor_diffusion_sample.generate_rejection()`.

Conceptual references

- Classifier/energy guidance for diffusion models; plug‑and‑play/energy-based sampling; SDE/ODE probability-flow adjustments. The implementation here follows the same spirit with DLPM-consistent scaling.

---

# Sampling Initialization Bug (alpha < 2) — Diagnosis & Fix

## The Bug

`generate()` in `factor_diffusion_sample.py` builds a Sigma chain from a pre-sampled A array `A[0..T-1]`, but then initializes $x_T$ from an **independent** subordinator `a_init`:

```python
# OLD — wrong for alpha < 2
a_init = sample_skewed_levy(LEVY_ALPHA, shape, DEVICE)   # independent of A[]
x = barsigmas[-1] * sample_sas(a_init)
```

For alpha < 2, `a_init ~ S(alpha/2, 1)` is heavy-tailed and can differ wildly from the A values used to build the Sigma chain.  The scale of `x_T = barsigmas[-1] * sqrt(a_init) * z` and the scale predicted by the chain `sqrt(Sigma_T)` can mismatch by up to **50×** in practice (observed: 1.81 % of samples at n = 4096 exceeded a 5× mismatch; see below).

For alpha = 2.0 there is no bug: `a_init ≡ 1` and `Sigma_T = barsigmas_T^2`, so the two initializations are identical.

## Why the Mismatch Causes Huge Kurtosis

When `|x_T| >> sqrt(Sigma_T)`, the model receives an input far outside its training distribution.  Its `eps_pred` is unreliable, so the posterior mean

$$\mu_t = \frac{x_t - \bar\sigma_t \,\Gamma_t\, \hat\varepsilon}{\gamma_t}$$

amplifies $x$ (division by $\gamma_t < 1$) instead of denoising it.  This cascade produces a handful of extreme outlier samples.  Verified on the regression checkpoint (alpha = 1.9):

| factor | real kurtosis | generated kurtosis |
|--------|:---:|:---:|
| market | 10.9 | **1851** |
| momentum | 15.0 | **2589** |
| growth | 7.5 | 9.9 |

The perfect-denoiser simulation (oracle $\hat\varepsilon$) recovers $x_0$ with MSE = 0 for both inits, confirming the **math is correct**; the issue is entirely the model encountering out-of-distribution inputs.

## The Fix

Initialize $x_T$ from $\mathcal{N}(0,\,\Sigma_T)$, consistent with the pre-sampled A chain:

```python
# NEW — consistent with A chain
x = Sigmas[-1].sqrt() * torch.randn(n, FACTOR_DIM, device=DEVICE)
```

This is valid because: sample $A_{1:T}$, then $x_T \mid A_{1:T} \sim \mathcal{N}(0, \Sigma_T)$, then run the DLPM reverse process conditioned on $A_{1:T}$.  Marginalizing over $A_{1:T}$ recovers the correct $p(x_0)$.

After the fix, samples with mismatch ratio > 5× drop from 74 (1.81 %) to **0**.

## Loss Scale Is Not Comparable Across Alpha

- **alpha = 2.0**: `eps_t ~ N(0,1)`, trivial baseline (output 0) gives loss = 1.  Loss < 1 means the model works.
- **alpha = 1.9**: `eps_t ~ SaS(1.9, 0)`, infinite variance (clamped to A ≤ 2000), trivial baseline gives loss >> 1.  Loss > 1 does **not** mean the model is worse than random — the baselines are on different scales.

## DDPM Does Not Generate Gaussian Samples

Setting alpha = 2.0 correctly degenerates to DDPM.  DDPM learns the **data distribution**, which for financial factor returns has kurtosis ≈ 7–15.  Generated samples will also be heavy-tailed.  The Gaussian qualifier refers to the noise process, not the generated distribution.

## Secondary Concern: Training Instability for alpha < 2 with mc_outer = 1

With `mc_outer = 1`, a single batch step containing a sample where $A_t \approx 2000$ produces an MSE loss spike of $\sim 2000$, swamping the gradient and destabilizing training.  The median-of-means estimator (mc_outer > 1, e.g. 5–10) removes this: the median is robust to extreme outer draws.  With `mc_outer = 1`, the model may learn to output near-zero for all inputs, causing mode collapse even after the initialization fix.

---

# EXP Environment Variable Bug — levy_alpha Mismatch at Inference

## The Bug

`factor_diffusion_sample.py` loads `LEVY_ALPHA` and `NUM_TIMESTEPS` from `model/{EXP}/cfg.yaml` **at module import time**, where `EXP = os.environ.get("EXP", "regression")`.

`factor_evaluation.ipynb` sets `_exp = "DDPM"` for its own path construction but **never exports `EXP` to the environment**. When `DiffusionSampler` calls `from factor_diffusion_sample import generate`, the already-imported module uses `LEVY_ALPHA` from `model/regression/cfg.yaml` (= 1.9) — not from `model/DDPM/cfg.yaml` (= 2.0).

Result: a checkpoint trained with `levy_alpha=2.0` (pure Gaussian schedule, `A_t ≡ 1`) is sampled with `levy_alpha=1.9` (heavy-tailed Lévy subordinators, stochastic `Sigma_t` that can reach ~2000). The model sees inputs far outside its training distribution; the denoising chain amplifies rather than contracts:

| Metric | Generated (buggy) | Resample baseline |
|--------|:-----------------:|:-----------------:|
| Kurtosis MAE | **110.2** | 5.9 |
| Std MAE | **0.159** | 0.001 |
| Cov Frobenius dist | **3901** | 1.05 |

## The Fix

Three-part fix so inference always uses the schedule that matches training:

1. **`factor_diffusion_train.py`** — save `levy_alpha` and `num_timesteps` inside the checkpoint:
   ```python
   torch.save({..., "levy_alpha": LEVY_ALPHA, "num_timesteps": NUM_TIMESTEPS}, path)
   ```

2. **`factor_diffusion_sample.py`** — `generate()` accepts `levy_alpha` and `num_timesteps` as explicit keyword args (fall back to module-level constants when `None`):
   ```python
   def generate(model, scaler, ..., levy_alpha=None, num_timesteps=None):
       _levy_alpha    = levy_alpha    if levy_alpha    is not None else LEVY_ALPHA
       _num_timesteps = num_timesteps if num_timesteps is not None else NUM_TIMESTEPS
       ...
   ```

3. **`scenario_generator.py`** — `DiffusionSampler.__init__` reads `levy_alpha` and `num_timesteps` from the checkpoint and passes them to every `generate()` call:
   ```python
   self.levy_alpha    = ckpt.get("levy_alpha")
   self.num_timesteps = ckpt.get("num_timesteps")
   ```

Existing checkpoints were back-patched with the correct values so no retraining is needed.

## Verified Result

After fix, with `EXP=DDPM` (or any EXP), `DiffusionSampler` always uses `levy_alpha=2.0` from the checkpoint. Generated kurtosis per factor: [6.2, 3.2, 4.1, 3.9, 3.5, 7.7, 2.6] — consistent with real data (target ≈ 7–15).

## Status: Resolved (cfg refactor)

The `EXP` env var path no longer exists. After the global `_cfg` refactor:

- `factor_diffusion_sample.generate()` takes `cfg` as an explicit argument and reads `levy_alpha`, `num_timesteps`, `batch_size` from it directly — no module-level constants, no env-var lookup at import time.
- `DiffusionSampler.__init__` loads `self.cfg = ckpt["cfg"]` from the checkpoint and threads it through every `generate()` call.

So inference is structurally guaranteed to use the schedule baked into the checkpoint. The `levy_alpha` / `num_timesteps` keyword args described in the original three-part fix are no longer needed and have been removed; the cfg dict carries them.

---

# Conditional Generation: Missing Cross-Factor Co-movement

## The Symptom

In `conditional_evaluation.ipynb`, conditioning the diffusion model on `volatility < q3` (3% quantile, threshold ≈ −0.0076) via gradient guidance (`guidance_scale = 5`) successfully shifts the **volatility** marginal below the threshold, but the **other six factors barely move** from their unconditional marginals.

By contrast:
- **Gaussian baseline** (`GaussianSampler`, multivariate-normal closed-form conditioning): other factors visibly co-move in the direction implied by historical correlations.
- **Soft rejection sampling** (`generate_rejection`, target `p(x)·exp(−s·L)`): other factors also co-move, since cross-factor structure is inherited from the unconditional model and only reweighted on the constrained axis.
- **Historical OOS stress days** (filtered for `momentum < q3`): clear co-movement across factors.

So the failure is **specific to gradient-guided diffusion**, not to DLPM as a model — the unconditional correlation matrix from diffusion already matches the historical/Gaussian correlation structure (see correlation scatter plot in the notebook).

## Root Cause: Detached Model Path Kills Cross-Factor Gradient

In `factor_diffusion_sample.generate()`, guidance is applied as:

$$\mu_t \leftarrow \mu_t - s\cdot \mathrm{var}_t \cdot \nabla_{x_t} L(\hat x_0), \qquad \hat x_0 = (x_t - \bar\sigma_t\,\hat\varepsilon_\theta(x_t,t))/\bar\gamma_t$$

The energy used here only depends on the constrained coordinate (volatility, index `vol`):

$$L(\hat x_0) = \mathrm{relu}(\hat x_{0,\mathrm{vol}} - q)^2$$

so $\nabla_{\hat x_0} L$ is non-zero only on the `vol` coordinate. The cross-factor coupling — i.e. "if vol is pushed low, market/momentum/value should follow" — must come entirely through $\partial \hat x_0 / \partial x_t$, which factors as:

$$\frac{\partial \hat x_0}{\partial x_t} = \frac{1}{\bar\gamma_t}\!\left(I - \bar\sigma_t \frac{\partial \hat\varepsilon_\theta}{\partial x_t}\right)$$

The dense Jacobian $\partial \hat\varepsilon_\theta / \partial x_t$ from the Transformer denoiser is exactly what carries cross-factor information. **The implementation `.detach()`s `eps_pred` before computing the guidance gradient** (for stability/cost — see "Why guidance is an approximation" section above). After detaching:

$$\frac{\partial \hat x_0}{\partial x_t} = \frac{1}{\bar\gamma_t}\,I \quad\Longrightarrow\quad \nabla_{x_t} L(\hat x_0) = \frac{1}{\bar\gamma_t}\big[0,\dots,0,\,2\,\mathrm{relu}(\hat x_{0,\mathrm{vol}}-q),\,0,\dots,0\big]$$

The guidance vector is **zero on every coordinate except `vol`**. So guidance only pushes volatility down; the other factors evolve under the unconditional reverse process and end up at their unconditional marginals. Increasing `guidance_scale` cannot fix this — it only multiplies a zero.

## Why the Other Methods Don't Have This Problem

| Method | Mechanism for cross-factor coupling |
|---|---|
| Gaussian (`GaussianSampler`) | Closed-form conditional mean $\mu_{-i} + \Sigma_{-i,i}\Sigma_{ii}^{-1}(x_i - \mu_i)$ — exact by construction. |
| Soft rejection (`generate_rejection`) | Samples from full $p(x)$ then reweights by $\exp(-sL)$; the joint dependence in $p$ is preserved on all axes. |
| Diffusion + gradient guidance (detached) | Only the constrained axis receives a non-zero gradient → no coupling. |

## Fix Options (in order of recommendation)

1. **Don't detach `eps_pred` during conditional sampling.** Let autograd flow through the denoiser so $\partial \hat\varepsilon_\theta / \partial x_t$ contributes to the guidance direction. Costs one extra backward through the model per step; may need gradient clipping for stability with $\alpha < 2$.
2. **Use soft rejection as the production path** for stress conditioning. With 7 factors and a 3% quantile, acceptance rate ≈ 5% (4096 / 82688 in the notebook) is tractable.
3. **SMC / particle-filter guidance.** At each step, resample particles by weights $\exp(-sL)$ instead of taking gradient steps. No gradient needed; cross-factor structure preserved by the unconditional dynamics.
4. **Calibrate `guidance_scale` against rejection** as a reference, but understand that under detached guidance, no scale value will recover cross-factor co-movement — it is a structural limitation of the current implementation, not a tuning problem.

## Code Waypoints

- Guidance gradient (the detach is here): `factor_diffusion_sample.generate()` — search for `eps_pred` and `grad`.
- Reference behavior: `factor_diffusion_sample.generate_rejection()`, `GaussianSampler` in `scenario_generator.py`.
- Reproduction notebook: `conditional_evaluation.ipynb`, In[15] (histogram across methods) and In[19] (per-factor KDE grid showing flat non-vol marginals for Diffusion row vs. shifted marginals for Rejection / Gaussian rows).

---

# SNR-Weighted Guidance Schedule

## Problem with the Hard Cutoff

`generate()` previously gated guidance by `1 < t < (T - 30)` — a binary on/off mask. The 30-step skip at high $t$ exists because $\hat x_0 = (x_t - \bar\sigma_t\,\hat\varepsilon)/\bar\gamma_t$ is unreliable when $\bar\sigma_t$ is large (denoiser is far OOD), so the energy gradient $\nabla L(\hat x_0)$ would push $x_t$ in nonsense directions and amplify under `1/gammas[t]`. But a hard cutoff is brittle: 30 is hand-tuned, and there's no smooth transition between "guidance off" and "full strength."

## First Attempt: `bargammas[t]^p` (didn't work)

Replace the mask with $w_t = \bar\gamma_t^p$ as a multiplier on the guidance update:

$$\mu_t \leftarrow \mu_t - s\cdot w_t\cdot \mathrm{var}_t \cdot \nabla L(\hat x_0)$$

This decays guidance smoothly at high noise (since $\bar\gamma_t \to 0$ as $t \to T$). But for $p \geq 3$ the histogram of conditioned samples collapses to the unconditional distribution. Reason: $\bar\gamma_t \in (0,1)$ raised to a high power kills $w_t$ everywhere except the very last few steps, *and* those last steps also have small $\mathrm{var}_t = \Gamma_t \Sigma_{t-1}$ — so the product $w_t\cdot \mathrm{var}_t$ is squashed at both ends and the middle, integrating to ≈ 0.

## Fix: SNR-Based Weighting (Karras/EDM-style)

Use the Bayes-optimal denoiser weight from the EDM family:

$$w_t = \left(\frac{\mathrm{SNR}_t}{\mathrm{SNR}_t + 1}\right)^p, \qquad \mathrm{SNR}_t = \frac{\bar\gamma_t^2}{\bar\sigma_t^2}$$

Properties:

- **Saturates to 1** at low noise ($\mathrm{SNR}_t \to \infty$), so guidance is full-strength where $\hat x_0$ is trustworthy.
- **Goes to 0** at high noise ($\mathrm{SNR}_t \to 0$), naturally suppressing the OOD regime.
- **Sigmoid-shaped transition** in the middle — no collapse like $\bar\gamma_t^p$.
- **Theoretically motivated**: $w_t$ is exactly the weight on $x_t/\bar\gamma_t$ in the Bayes-optimal $\hat x_0(x_t)$ assuming a unit-variance Gaussian prior. So guidance is weighted by "how much the denoiser actually trusts the data term at step $t$."

For DDPM ($\alpha=2$), $\bar\gamma_t^2 + \bar\sigma_t^2 = 1$, so $w_t = \bar\gamma_t^2 = \bar\alpha_t$ — the standard SNR weight.

For DLPM ($\alpha<2$), the constraint is $\bar\gamma_t^\alpha + \bar\sigma_t^\alpha = 1$ (α-norm), so $\bar\gamma_t^2 + \bar\sigma_t^2 \neq 1$. The 2-norm SNR is still the right form because the *posterior* of $\hat x_0 \mid x_t, A$ is Gaussian (conditioned on the subordinator chain) — variance addition is in 2-norm.

## Knobs

- `guidance_decay_pow=1.0` (default): canonical SNR weight.
- `guidance_decay_pow=0.0`: disable decay (guidance applied uniformly, only `t > 1` mask).
- `guidance_decay_pow > 1`: sharpen the high-noise cutoff while preserving the sigmoid shape.
- Tune `guidance_scale` *first* for total strength; tune `guidance_decay_pow` *second* for the temporal distribution. They are orthogonal knobs.

## Code Waypoints

- Weight computation: `factor_diffusion_sample.generate()` — `snr_t = bargammas[t]**2 / barsigmas[t]**2; w_t = (snr_t / (snr_t + 1))**guidance_decay_pow`.
- Threading from sampler API: `DiffusionSampler.__init__(guidance_decay_pow=...)` in `scenario_generator.py`.
- `grad_history` now logs `(t, grad_mean, grad_max, w_t)` so you can plot the effective schedule.

---

# VIX Conditioning (optional exogenous regime signal)

## Goal

Let the temporal model optionally condition the next-day factor cross-section on a **market
volatility regime** given by the CBOE VIX (`data/VIXCLS.csv`), *without losing the ability to
generate VIX-free*. One set of weights must serve both:

- `p(F_{t+1} \mid F_t, \mathrm{VIX})` — stress-scenario generation ("given tomorrow's VIX is 40, …")
- `p(F_{t+1} \mid F_t)` — the original behavior, when no VIX is supplied

## Why this is the clean fix (vs. gradient guidance)

The existing energy/gradient guidance only moves the constrained coordinate and — because
`eps_pred` is detached — produces **no cross-factor co-movement** (see "Conditional
Generation: Missing Cross-Factor Co-movement"). Conditioning on VIX as a **trained input**
sidesteps that entirely: the cross-factor response to a volatility regime is learned
end-to-end, so a high-VIX condition moves *all* factors as they co-move in the training data.
No detach problem, no per-step gradient.

## Mechanism: a second null-token (independent CFG)

VIX is just another scalar context added into the per-token condition `ctx`. It gets its own
**learned null token** so it can be present or absent independently of the factor condition:

```python
# FactorDenoiser.__init__
self.vix_embed = nn.Sequential(nn.Linear(1, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
self.null_vix  = nn.Parameter(torch.zeros(1, cond_dim))

# FactorDenoiser.forward(x, t, c, vix, cond_drop_mask, vix_drop_mask)
if vix is None:
    ctx = ctx + self.null_vix                       # "no VIX"
else:
    vctx = self.vix_embed(vix)                      # (B, cond_dim)
    if vix_drop_mask is not None:
        vctx = torch.where(vix_drop_mask[:, None], self.null_vix, vctx)
    ctx = ctx + vctx
```

During training, `vix` is dropped with **independent** probability `cond_drop_prob_vix`
(separate Bernoulli from the factor-condition dropout). The four combinations teach the
model `p(·|F_t,VIX)`, `p(·|F_t)`, `p(·|VIX)`, `p(·)` jointly — so "optionally conditioned" is
the *default product* of this design, not extra machinery, and a single checkpoint covers it.

## Data alignment & timing

- **Timing: VIX at day t+1** — the same day as the target `x = X[1:]`, not the seed day. This
  is what makes it a forward "regime" knob: at rollout you supply the VIX of the day being
  generated. Training pairs are `(c=X[i], vix=vix_norm[i+1], x=X[i+1])`.
- **Transform: log + standardize.** VIX is right-skewed (9–80, spikes to ~83); `log` makes it
  closer to Gaussian for the linear embedding, then `StandardScaler`. The fitted `vix_scaler`
  is saved in the checkpoint so inference uses the identical transform.
- **Gaps:** FRED encodes ~302 market-holiday rows as missing; aligned to the factor calendar
  and `ffill().bfill()`-ed. The 2025 OOS window is fully covered.

## Sampling: branch-free rollout via an adaptor

`vix_cond_gen(vix_cond, length, num_paths, vix_scaler)` normalizes the user spec into a
length-`horizon` per-step sequence so the reverse loop never branches on "VIX or not":

- `vix_cond is None`  → `[None] * length`            (every step uses `null_vix`)
- `vix_cond` scalar   → constant trajectory, shared across all paths (log+standardized via `vix_scaler`)

`_reverse(..., vix=...)` takes a single step's `(n,1)` tensor (or `None`) and feeds it to the
model; `generate_path` indexes `vix_seq[d]` for the day `d` being generated. Only constant /
None are implemented now — a full per-day VIX *trajectory* (e.g. the real OOS VIX path, or a
synthetic mean-reverting one) is the intended extension point in `vix_cond_gen`.

## Optional CFG amplification (not implemented yet)

Plain conditioning (feed VIX, one forward) already biases generation. If the VIX effect comes
out too weak, classifier-free guidance can amplify it:
`eps = eps(F_t) + w·(eps(F_t,VIX) − eps(F_t))` with `w>1` (two forwards). `w=0` ≡ unconditional,
`w=1` ≡ plain conditional. This is the clean (no-detach) guidance and is orthogonal to the
existing energy guidance. Left out of the minimal version; add only if needed.

## Knobs

- `cfg["vix_file"]` (e.g. `data/VIXCLS.csv`): enables VIX in `load_data`. Absent → no VIX, old behavior.
- `cfg["cond_drop_prob_vix"]` (default 0.15): independent VIX dropout during training.
- `--vix <level>` on `factor_diffusion_sample.py`: constant VIX to condition on; omit → VIX-free.

## Status: requires retraining

`null_vix` / `vix_embed` are new parameters, so pre-VIX checkpoints will not load with
`strict=True`. This is intentional — train fresh with `vix_file` set. Backward compatibility
was explicitly *not* a goal.

## Code Waypoints

- Model: `FactorDenoiser.__init__` / `forward` (`vix`, `vix_drop_mask`, `null_vix`) in `factor_diffusion_train.py`.
- Loss: `dlpm_loss(..., vix=, vix_drop_prob=)` — independent dropout Bernoulli.
- Data: `load_data(csv, factors, vix_path)` returns `(X_norm, scaler, vix_norm, vix_scaler)`; VIX taken at `t+1`.
- Checkpoint: `train(..., vix_scaler=)` saves `"vix_scaler"`.
- Sampling: `vix_cond_gen`, `_reverse(..., vix=)`, `generate_path(..., vix_cond=, vix_scaler=)` in `factor_diffusion_sample.py`; `--vix` in its `__main__`.

---
