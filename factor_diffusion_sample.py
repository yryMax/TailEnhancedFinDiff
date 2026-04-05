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

FACTOR_NAMES  = _tcfg["factor_names"]
EPOCHS        = _tcfg["epochs"]
NUM_TIMESTEPS = _tcfg["num_timesteps"]
LEVY_ALPHA    = _tcfg["levy_alpha"]
PREFIX        = _tcfg["prefix"]
BATCH_SIZE    = _tcfg["batch_size"]

NUM_GENERATE  = _scfg["num_generate"]
FACTOR_DIM    = len(FACTOR_NAMES)
CHECKPOINT    = f"{PREFIX}/checkpoints/001.pt"
OUT_PATH      = f"{PREFIX}/samples/factor_{NUM_GENERATE}.npy"




@torch.no_grad()
def generate(model, gammas, sigmas, barsigmas, levy_alpha, scaler):
    """
    DLPM reverse process. alpha=2 automatically degenerates to DDPM.
    For each batch:
    1. Pre-sample A_{1:T} ~ S(alpha/2, beta=1) — the full Lévy path.
    2. Compute Sigma_t chain:
           Sigma_0 = sigma_0^2 * A_0
           Sigma_t = sigma_t^2 * A_t + gamma_t^2 * Sigma_{t-1}
    3. Start from x_T ~ barsigmas[-1] * SαS  (marginal forward distribution).
    4. Reverse denoising T-1 → 1:
           Gamma_t  = 1 - gamma_t^2 * Sigma_{t-1} / Sigma_t
           mean     = (x_t - barsigmas[t] * Gamma_t * eps_pred) / gammas[t]
           variance = Gamma_t * Sigma_{t-1}
           x_{t-1}  = mean + sqrt(variance) * N(0,I)
    """
    T = len(gammas)
    gammas    = gammas.to(DEVICE)
    sigmas    = sigmas.to(DEVICE)
    barsigmas = barsigmas.to(DEVICE)

    model.eval()
    batches = []

    for start in range(0, NUM_GENERATE, BATCH_SIZE):
        n     = min(BATCH_SIZE, NUM_GENERATE - start)
        shape = (n, FACTOR_DIM)

        A = [sample_skewed_levy(levy_alpha, shape, DEVICE) for _ in range(T)]

        Sigmas = [sigmas[0] ** 2 * A[0]]
        for t in range(1, T):
            Sigmas.append(sigmas[t] ** 2 * A[t] + gammas[t] ** 2 * Sigmas[-1])

        a_init = sample_skewed_levy(levy_alpha, shape, DEVICE)
        x = barsigmas[-1] * sample_sas(a_init)

        for t in range(T - 1, 0, -1):
            t_b      = torch.full((n,), t, dtype=torch.long, device=DEVICE)
            eps_pred = model(x, t_b)

            Sigma_t  = Sigmas[t]
            Sigma_t1 = Sigmas[t - 1]

            # posterior contraction factor
            Gamma_t = 1 - (gammas[t] ** 2 * Sigma_t1) / (Sigma_t + 1e-8)
            Gamma_t = Gamma_t.clamp(0.0, 1.0)

            # posterior mean
            mean = (x - barsigmas[t] * Gamma_t * eps_pred) / gammas[t]

            # posterior variance (Gaussian, conditioned on A)
            var = (Gamma_t * Sigma_t1).clamp(min=0.0)

            if t > 1:
                x = mean + var.sqrt() * torch.randn_like(x)
            else:
                x = mean   # no noise at last step

        batches.append(x.cpu())
    return scaler.inverse_transform(torch.cat(batches).numpy())


def generate_conditional(model, scheduler, scaler, factor, threshold, guidance_scale=200.0):
    model.eval()
    scheduler.set_timesteps(scheduler.config.num_train_timesteps)

    factor_idx     = FACTOR_NAMES.index(factor)
    dummy          = np.zeros((1, FACTOR_DIM))
    dummy[0, factor_idx] = threshold
    threshold_norm = float(scaler.transform(dummy)[0, factor_idx])

    batches = []
    for start in range(0, NUM_GENERATE, BATCH_SIZE):
        n = min(BATCH_SIZE, NUM_GENERATE - start)
        # start from pure noise
        x = torch.randn(n, FACTOR_DIM, device=DEVICE)
        # for each timestep
        for t_val in scheduler.timesteps:
            # enable gradient tracking for x
            x = x.detach().requires_grad_(True)

            # broadcast timestep to batch
            t_b = torch.full((n,), t_val, dtype=torch.long, device=DEVICE)

            # predict noise (detach so gradient only flows through direct x → x0_hat path)
            with torch.no_grad():
                noise_pred = model(x, t_b)

            alpha_bar   = scheduler.alphas_cumprod[t_val].to(DEVICE)

            # estimate x0 (gradient flows through x only, not through the model)
            x0_hat    = (x - (1 - alpha_bar).sqrt() * noise_pred) / alpha_bar.sqrt()

            # panalty if the volatility is smaller than the threshold
            cond_loss = torch.relu(threshold_norm - x0_hat[:, factor_idx]).pow(2).mean()

            # where the sample should go for larger panalty.
            grad      = torch.autograd.grad(cond_loss, x)[0]

            with torch.no_grad():
                # predict noise and compute previous sample
                x = scheduler.step(noise_pred, t_val, x.detach()).prev_sample
                grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                # shift the sample toward the direction that decrease the panalty.
                x = x - guidance_scale * grad / grad_norm

        batches.append(x.detach().cpu())

    return scaler.inverse_transform(torch.cat(batches).numpy())


if __name__ == "__main__":
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    print(f"Loaded checkpoint from {CHECKPOINT}, generate {NUM_GENERATE} samples...")
    os.makedirs(f"{PREFIX}/samples", exist_ok=True)
    model = FactorDenoiser(**ckpt["model_kwargs"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])

    scaler = ckpt["scaler"]
    gammas, _, sigmas, barsigmas = levy_noise_schedule(LEVY_ALPHA, NUM_TIMESTEPS)

    print(f"LEVY_ALPHA={LEVY_ALPHA}, T={NUM_TIMESTEPS}")
    samples = generate(model, gammas, sigmas, barsigmas, LEVY_ALPHA, scaler)
    np.save(OUT_PATH, samples)
    print(f"Saved {samples.shape} samples → {OUT_PATH}")
