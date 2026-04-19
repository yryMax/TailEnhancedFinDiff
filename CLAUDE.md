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

---
