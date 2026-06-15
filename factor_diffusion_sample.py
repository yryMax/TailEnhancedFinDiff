import argparse
import os
import numpy as np
import torch
from factor_diffusion_train import FactorDenoiser
from factor_diffusion_levy import levy_noise_schedule, sample_skewed_levy, sample_sas

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def generate(model, scaler, cfg, num_samples=None, cond_fn=None,
             guidance_scale=5.0, guidance_decay_pow=1.0, L=None):
    """
    DLPM reverse process. alpha=2 automatically degenerates to DDPM.

    For each batch:
    1. Pre-sample A_{1:T} ~ S(alpha/2, beta=1)
    2. Compute Sigma_t chain:
           Sigma_0 = sigma_0^2 * A_0
           Sigma_t = sigma_t^2 * A_t + gamma_t^2 * Sigma_{t-1}
    3. Start from x_T ~ N(0, Sigma_T) (consistent with the pre-sampled A chain).
    4. Reverse denoising T-1 → 1:
           Gamma_t  = 1 - gamma_t^2 * Sigma_{t-1} / Sigma_t
           mean     = (x_t - barsigmas[t] * Gamma_t * eps_pred) / gammas[t]
           variance = Gamma_t * Sigma_{t-1}
           x_{t-1}  = mean + sqrt(variance) * N(0,I)

    :param cfg: dict with keys levy_alpha, num_timesteps, batch_size; num_generate optional
                (used as default for num_samples)
    :param num_samples: total samples to generate; defaults to cfg["num_generate"]
    :param cond_fn: optional cond_fn(x0_hat) -> per-sample loss; enables gradient guidance
    :param guidance_scale: guidance strength `s`; mean -= s * w_t * var * grad
    :param guidance_decay_pow: exponent p in w_t = (SNR_t / (SNR_t + 1))**p, where
                               SNR_t = bargammas[t]^2 / barsigmas[t]^2. The base SNR weight
                               (p=1) is the Karras/EDM-style Bayes-optimal weighting:
                               smoothly fades to 0 at high noise (where x0_hat is unreliable)
                               and saturates near 1 at low noise. p=0 disables decay;
                               larger p sharpens the transition at high noise.
    :param L: optional Cholesky factor for L-coloured noise injection
    """
    levy_alpha    = cfg["levy_alpha"]
    num_timesteps = cfg["num_timesteps"]
    batch_size    = cfg["batch_size"]
    if num_samples is None:
        num_samples = cfg["num_generate"]
    factor_dim = model.feature_embed.shape[1]

    gammas, bargammas, sigmas, barsigmas = levy_noise_schedule(levy_alpha, num_timesteps)
    T = len(gammas)
    gammas    = gammas.to(DEVICE)
    bargammas = bargammas.to(DEVICE)
    sigmas    = sigmas.to(DEVICE)
    barsigmas = barsigmas.to(DEVICE)

    L_d = L.to(DEVICE) if L is not None else None

    model.eval()
    batches = []
    var_history = []
    grad_history = []

    for start in range(0, num_samples, batch_size):
        n     = min(batch_size, num_samples - start)
        shape = (n, factor_dim)

        A = [sample_skewed_levy(levy_alpha, shape, DEVICE) for _ in range(T)]

        Sigmas = [sigmas[0] ** 2 * A[0]]
        for t in range(1, T):
            Sigmas.append(sigmas[t] ** 2 * A[t] + gammas[t] ** 2 * Sigmas[-1])

        z_init = torch.randn(n, factor_dim, device=DEVICE)
        if L_d is not None:
            z_init = z_init @ L_d.T
        x = Sigmas[-1].sqrt() * z_init

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

            # DLPM-correct guidance: shift posterior mean by -s * w_t * var * ∂loss/∂x_t
            # eps recomputed under enable_grad so ∂eps/∂x_t (dense Jacobian carrying
            # cross-factor coupling) flows into the guidance gradient.
            # w_t = (SNR_t / (SNR_t + 1))**p — SNR-based weighting (Karras/EDM-style):
            # ~0 at high noise where x0_hat is unreliable, saturates to 1 at low noise.
            if cond_fn is not None and t > 1:
                snr_t = bargammas[t] ** 2 / (barsigmas[t] ** 2 + 1e-8)
                w_t   = (snr_t / (snr_t + 1.0)) ** guidance_decay_pow
                with torch.enable_grad():
                    x_g        = x.detach().requires_grad_(True)
                    eps_pred_g = model(x_g, t_b)
                    x0_hat     = (x_g - barsigmas[t] * eps_pred_g) / bargammas[t]
                    loss       = cond_fn(x0_hat).sum()
                    grad       = torch.autograd.grad(loss, x_g)[0]
                if start == 0:
                    grad_history.append((t, float(grad.mean()), float(grad.max()), float(w_t)))
                mean = mean - guidance_scale * w_t * var * grad.detach()

            if t > 1:
                z = torch.randn_like(x)
                if L_d is not None:
                    z = z @ L_d.T
                x = mean + var.sqrt() * z
            else:
                x = mean   # no noise at last step

        batches.append(x.cpu())

    return scaler.inverse_transform(torch.cat(batches).numpy()), var_history, grad_history


@torch.no_grad()
def generate_style_transfer(model, scaler, cfg, x0_obs, cond_fn, *,
                    strength=0.6, guidance_scale=5.0, guidance_decay_pow=1.0):
    """
    :param x0_obs:   (N, factor_dim) observed factor vectors in *raw* units; the scaler
                     normalizes them internally.
    :param cond_fn:  cond_fn(x0_hat) -> per-sample loss; defines the scenario the transfer
                     steers toward (required — without a condition there is nothing to transfer).
    :param strength: fraction of the diffusion horizon to re-noise, in (0, 1].
    :return: (samples_ndarray, var_history, grad_history) — same convention as generate().
    """
    levy_alpha    = cfg["levy_alpha"]
    num_timesteps = cfg["num_timesteps"]
    batch_size    = cfg["batch_size"]
    factor_dim    = model.feature_embed.shape[1]

    x0_all = np.asarray(scaler.transform(np.asarray(x0_obs, dtype=np.float32)),
                        dtype=np.float32)
    num_samples = x0_all.shape[0]

    gammas, bargammas, sigmas, barsigmas = levy_noise_schedule(levy_alpha, num_timesteps)
    T = len(gammas)
    gammas    = gammas.to(DEVICE)
    bargammas = bargammas.to(DEVICE)
    sigmas    = sigmas.to(DEVICE)
    barsigmas = barsigmas.to(DEVICE)

    t_star = int(round(strength * (T - 1)))
    t_star = max(1, min(T - 1, t_star))

    model.eval()
    batches = []
    var_history = []
    grad_history = []

    for start in range(0, num_samples, batch_size):
        n     = min(batch_size, num_samples - start)
        shape = (n, factor_dim)
        x0    = torch.tensor(x0_all[start:start + n], device=DEVICE)

        A = [sample_skewed_levy(levy_alpha, shape, DEVICE) for _ in range(t_star + 1)]
        Sigmas = [sigmas[0] ** 2 * A[0]]
        for t in range(1, t_star + 1):
            Sigmas.append(sigmas[t] ** 2 * A[t] + gammas[t] ** 2 * Sigmas[-1])

        z_init = torch.randn(n, factor_dim, device=DEVICE)
        x = bargammas[t_star] * x0 + Sigmas[t_star].sqrt() * z_init

        for t in range(t_star, 0, -1):
            t_b      = torch.full((n,), t, dtype=torch.long, device=DEVICE)
            eps_pred = model(x, t_b)

            Sigma_t  = Sigmas[t]
            Sigma_t1 = Sigmas[t - 1]

            Gamma_t = 1 - (gammas[t] ** 2 * Sigma_t1) / (Sigma_t + 1e-8)
            mean = (x - barsigmas[t] * Gamma_t * eps_pred) / gammas[t]
            var = (Gamma_t * Sigma_t1).clamp(min=0.0)

            if start == 0:
                var_history.append((t, float(var.mean()), float(var.max())))

            if t > 1:
                snr_t = bargammas[t] ** 2 / (barsigmas[t] ** 2 + 1e-8)
                w_t   = (snr_t / (snr_t + 1.0)) ** guidance_decay_pow
                with torch.enable_grad():
                    x_g        = x.detach().requires_grad_(True)
                    eps_pred_g = model(x_g, t_b)
                    x0_hat     = (x_g - barsigmas[t] * eps_pred_g) / bargammas[t]
                    loss       = cond_fn(x0_hat).sum()
                    grad       = torch.autograd.grad(loss, x_g)[0]
                if start == 0:
                    grad_history.append((t, float(grad.mean()), float(grad.max()), float(w_t)))
                mean = mean - guidance_scale * w_t * var * grad.detach()

                x = mean + var.sqrt() * torch.randn_like(x)
            else:
                x = mean

        batches.append(x.cpu())

    return scaler.inverse_transform(torch.cat(batches).numpy()), var_history, grad_history


def generate_rejection(model, scaler, cfg, *, cond_fn, num_samples=None,
                       guidance_scale=1.0, max_batches=50000, L=None):
    """
    Exact conditional sampling via rejection on top of generate().

    Each iteration draws an unconditional batch via generate(); each sample is then
    accepted with probability exp(-guidance_scale * cond_fn(x)). Returns np.ndarray
    in original space, shape (num_samples, factor_dim).
    """
    if num_samples is None:
        num_samples = cfg["num_generate"]
    batch_size = cfg["batch_size"]

    accepted = []
    n_tried  = 0

    for _ in range(max_batches):
        if len(accepted) >= num_samples:
            break

        batch_np, _, _ = generate(model, scaler, cfg, num_samples=batch_size, L=L)
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
        print(f"[rejection] accepted {n_got}/{n_tried}  (rate={rate:.4f})")

    return np.stack(accepted[:num_samples])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", help="experiment name; loads model/<exp_name>/checkpoints/*.pt")
    parser.add_argument("--ckpt", default=None,
                        help="checkpoint filename (without .pt); defaults to ckpt['cfg']['ckpt_name']")
    args = parser.parse_args()

    prefix = f"model/{args.exp_name}"
    ckpt_name = args.ckpt
    if ckpt_name is None:
        # peek the only ckpt or fall back to the directory listing
        ckpts = [f for f in os.listdir(f"{prefix}/checkpoints") if f.endswith(".pt")]
        assert len(ckpts) == 1, f"specify --ckpt; found {ckpts}"
        ckpt_name = ckpts[0][:-3]
    ckpt_path = f"{prefix}/checkpoints/{ckpt_name}.pt"

    ckpt   = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg    = ckpt["cfg"]
    model  = FactorDenoiser(**ckpt["model_kwargs"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    scaler = ckpt["scaler"]
    L      = ckpt.get("L_noise")

    out_path = f"{prefix}/samples/factor_{cfg['ckpt_name']}_{cfg['num_generate']}.npy"
    os.makedirs(f"{prefix}/samples", exist_ok=True)

    samples, _, _ = generate(model, scaler, cfg, L=L)
    np.save(out_path, samples)
    print(f"Saved {samples.shape} samples → {out_path}")
