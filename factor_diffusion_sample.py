import numpy as np
import torch
from diffusers import DDPMScheduler
import os
from factor_diffusion_train import FactorDenoiser, FACTOR_NAMES, IdentityScaler, BATCH_SIZE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Config ─────────────────────────────────────────────────────────────────────
CHECKPOINT   = "checkpoints/factor_ddpm_ep0200.pt"
NUM_GENERATE = 2048
FACTOR_DIM = 6
OUT_PATH     = f"samples/factor_ddpm_{NUM_GENERATE}.npy"


# ── Sample ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate(model, scheduler, scaler):

    model.eval()
    scheduler.set_timesteps(scheduler.config.num_train_timesteps)
    batches = []
    for start in range(0, NUM_GENERATE, BATCH_SIZE):
        n = min(BATCH_SIZE, NUM_GENERATE - start)
        # start from pure noise
        x = torch.randn(n, FACTOR_DIM, device=DEVICE)
        # for each timestep
        for t_val in scheduler.timesteps:
            # broadcast timestep to batch
            t_b = torch.full((n,), t_val, dtype=torch.long, device=DEVICE)

            # predict noise and compute previous sample
            x   = scheduler.step(model(x, t_b), t_val, x).prev_sample
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
    os.makedirs("samples", exist_ok=True)
    ckpt   = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model  = FactorDenoiser(**ckpt["model_kwargs"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    scaler        = ckpt["scaler"]
    num_timesteps = ckpt["num_timesteps"]

    scheduler = DDPMScheduler(num_train_timesteps=num_timesteps,
                              beta_schedule="squaredcos_cap_v2", prediction_type="epsilon")

    # unconditional
    samples = generate(model, scheduler, scaler)
    np.save(OUT_PATH, samples)


    print("Unconditional  volatility: "
          f"mean={samples[:, FACTOR_NAMES.index('volatility')].mean():+.6f}  "
          f"std={samples[:, FACTOR_NAMES.index('volatility')].std():.6f}")


    
    # volatility > 0.001
    cond_samples = generate_conditional(model, scheduler, scaler,
                                        factor="volatility", threshold=0.001, guidance_scale=0.3)
    np.save(f"samples/factor_ddpm_cond_volatility_{NUM_GENERATE}.npy", cond_samples)
    print(f"\nConditional (volatility > 0.001)  volatility: "
          f"mean={cond_samples[:, FACTOR_NAMES.index('volatility')].mean():+.6f}  "
          f"std={cond_samples[:, FACTOR_NAMES.index('volatility')].std():.6f}")

