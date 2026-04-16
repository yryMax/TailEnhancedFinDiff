import os
import yaml
import numpy as np
import torch
from factor_diffusion_train import FactorDenoiser
from factor_diffusion_levy import levy_noise_schedule, sample_skewed_levy, sample_sas

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("cfg.yaml") as f:
    _all_cfg = yaml.safe_load(f)
    _tcfg    = _all_cfg["train"]
    _scfg    = _all_cfg["sample"]

NUM_TIMESTEPS = _tcfg["num_timesteps"]
LEVY_ALPHA    = _tcfg["levy_alpha"]
PREFIX        = _tcfg["prefix"]
BATCH_SIZE    = _tcfg["batch_size"]

NUM_GENERATE  = _scfg["num_generate"]
FACTOR_DIM    = len(_tcfg["factor_names"])
CHECKPOINT    = f"{PREFIX}/checkpoints/factor_DLPM_ep0200.pt"
OUT_PATH      = f"{PREFIX}/samples/factor_{NUM_GENERATE}.npy"



@torch.no_grad()
def generate(model, scaler, cond_fn=None, guidance_scale=5.0, num_samples=None):
    """
    DLPM reverse process. alpha=2 automatically degenerates to DDPM.
    For each batch:
    1. Pre-sample A_{1:T} ~ S(alpha/2, beta=1)
    2. Compute Sigma_t chain:
           Sigma_0 = sigma_0^2 * A_0
           Sigma_t = sigma_t^2 * A_t + gamma_t^2 * Sigma_{t-1}
    3. Start from x_T ~ barsigmas[-1] * SαS  (marginal forward distribution).
    4. Reverse denoising T-1 → 1:
           Gamma_t  = 1 - gamma_t^2 * Sigma_{t-1} / Sigma_t
           mean     = (x_t - barsigmas[t] * Gamma_t * eps_pred) / gammas[t]
           variance = Gamma_t * Sigma_{t-1}
           x_{t-1}  = mean + sqrt(variance) * N(0,I)

    Optional DLPM-correct gradient guidance:
        cond_fn(x0_hat: Tensor[n,D]) -> scalar loss
            Receives the Tweedie x0 estimate; returns a scalar loss (higher = farther from target).
        The posterior mean is corrected by:
            mean -= guidance_scale * var * ∂loss/∂x_t
        where var = Gamma_t * Sigma_{t-1} provides the proper Bayesian scaling.
    """
    gammas, bargammas, sigmas, barsigmas = levy_noise_schedule(LEVY_ALPHA, NUM_TIMESTEPS)
    T = len(gammas)
    gammas    = gammas.to(DEVICE)
    bargammas = bargammas.to(DEVICE)
    sigmas    = sigmas.to(DEVICE)
    barsigmas = barsigmas.to(DEVICE)

    if num_samples is None:
        num_samples = NUM_GENERATE

    model.eval()
    batches = []
    var_history = []
    grad_history = []

    for start in range(0, num_samples, BATCH_SIZE):
        n     = min(BATCH_SIZE, num_samples - start)
        shape = (n, FACTOR_DIM)

        A = [sample_skewed_levy(LEVY_ALPHA, shape, DEVICE) for _ in range(T)]

        Sigmas = [sigmas[0] ** 2 * A[0]]
        for t in range(1, T):
            Sigmas.append(sigmas[t] ** 2 * A[t] + gammas[t] ** 2 * Sigmas[-1])

        a_init = sample_skewed_levy(LEVY_ALPHA, shape, DEVICE)
        # barsigmas[-1] is nearly 1 because of scale preserving
        x = barsigmas[-1] * sample_sas(a_init)

        for t in range(T - 1, 0, -1):
            t_b      = torch.full((n,), t, dtype=torch.long, device=DEVICE)
            eps_pred = model(x, t_b)

            Sigma_t  = Sigmas[t]
            Sigma_t1 = Sigmas[t - 1]

            # posterior contraction factor
            Gamma_t = 1 - (gammas[t] ** 2 * Sigma_t1) / (Sigma_t + 1e-8)

            # posterior mean
            mean = (x - barsigmas[t] * Gamma_t * eps_pred) / gammas[t]

            # posterior variance (Gaussian, conditioned on A)
            var = (Gamma_t * Sigma_t1).clamp(min=0.0)

            if start == 0:
                var_history.append((t, float(var.mean()), float(var.max())))

            # DLPM-correct guidance: shift posterior mean by -s * var * ∂loss/∂x_t
            # var provides natural Bayesian scaling
            if cond_fn is not None and 1 < t < T // 2:
                with torch.enable_grad():
                    x_g    = x.detach().requires_grad_(True)
                    x0_hat = (x_g - barsigmas[t] * eps_pred) / bargammas[t]
                    loss   = cond_fn(x0_hat).sum()
                    grad   = torch.autograd.grad(loss, x_g)[0]
                # per-sample unit-norm clip: preserve direction, bound magnitude
                grad = grad / grad.norm(dim=1, keepdim=True).clamp(min=1e-8) # zero - aware
                if start == 0:
                    grad_history.append((t, float(grad.mean()), float(grad.max())))
                mean = mean - guidance_scale * var * grad.detach()


            if t > 1:
                x = mean + var.sqrt() * torch.randn_like(x)
            else:
                x = mean   # no noise at last step

        batches.append(x.cpu())

    return scaler.inverse_transform(torch.cat(batches).numpy()), var_history, grad_history




def generate_rejection(model, scaler, cond_fn, num_samples=None, guidance_scale=1.0, max_batches=50000):
    """
    Exact conditional sampling via rejection sampling on top of generate().

    Each iteration calls generate(..., num_samples=BATCH_SIZE, cond_fn=None) to get
    an unconditional batch, then accepts each sample based on cond_fn:
      hard=True  (default): accept if cond_fn(x0[i:i+1]) < 1e-8
                            (relu hinge loss == 0 iff constraint satisfied)
      hard=False           : accept with probability exp(-guidance_scale * loss_i)
                            exact samples from p_uncon(x0) * exp(-guidance_scale * L(x0))

    cond_fn called with n=1 so .mean() in the closure returns the per-sample value.
    Returns np.ndarray (num_samples, FACTOR_DIM) in original space.
    """
    if num_samples is None:
        num_samples = NUM_GENERATE

    accepted = []
    n_tried  = 0

    for _ in range(max_batches):
        if len(accepted) >= num_samples:
            break

        batch_np, _, _ = generate(model, scaler, num_samples=BATCH_SIZE)
        batch_t  = torch.tensor(scaler.transform(batch_np), dtype=torch.float32)

        n_tried += len(batch_t)
        for i in range(len(batch_t)):
            xi     = batch_t[i:i+1]
            loss_i = float(cond_fn(xi))

            accept = np.random.rand() < np.exp(-guidance_scale * loss_i)

            if accept:
                accepted.append(batch_np[i])
            if len(accepted) >= num_samples:
                break

    n_got = len(accepted)
    rate  = n_got / n_tried if n_tried > 0 else 0.0
    if n_got < num_samples:
        import warnings
        warnings.warn(
            f"generate_rejection: only {n_got}/{num_samples} samples after {n_tried} tries "
            f"(rate={rate:.4f}). Increase max_batches or relax the condition."
        )
    else:
        print(f"  [rejection] accepted {n_got}/{n_tried}  (rate={rate:.4f})")

    return np.stack(accepted[:num_samples])


if __name__ == "__main__":
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    print(f"Loaded checkpoint from {CHECKPOINT}, generate {NUM_GENERATE} samples...")
    os.makedirs(f"{PREFIX}/samples", exist_ok=True)
    model = FactorDenoiser(**ckpt["model_kwargs"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])

    scaler = ckpt["scaler"]

    print(f"LEVY_ALPHA={LEVY_ALPHA}, T={NUM_TIMESTEPS}")
    samples, _, _ = generate(model, scaler)
    np.save(OUT_PATH, samples)
    print(f"Saved {samples.shape} samples → {OUT_PATH}")
