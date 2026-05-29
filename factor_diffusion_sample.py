import argparse
import os
import numpy as np
import torch
from factor_diffusion_train import FactorDenoiser
from factor_diffusion_levy import levy_noise_schedule

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def _reverse(model, cfg, cond, n, cond_fn=None, guidance_scale=5.0,
             guidance_decay_pow=1.0):
    """
    1. x_T ~ N(0, barsigmas[T-1]^2 I)
    2. For t = T-1 ... 1, with clean condition `cond` fed at every step:
           eps_pred = model(x_t, t, cond)
           Gamma_t  = 1 - gammas[t]^2 * barsigmas[t-1]^2 / barsigmas[t]^2
           mean     = (x_t - barsigmas[t] * Gamma_t * eps_pred) / gammas[t]
           var      = Gamma_t * barsigmas[t-1]^2
           x_{t-1}  = mean + sqrt(var) * N(0,I)

    :param cond: (n, F) clean condition tensor on DEVICE, normalized space. None →
                 unconditional draw from the learned null token
    :param cond_fn: optional cond_fn(x0_hat) -> per-sample energy; enables gradient guidance
    :param guidance_scale: guidance strength `s`; mean -= s * w_t * var * grad
    :param guidance_decay_pow: exponent p in w_t = (SNR_t / (SNR_t + 1))**p (Karras/EDM SNR weight)
    :return: (x, var_history, grad_history); x is (n, F) normalized tensor on DEVICE
    """
    num_timesteps = cfg["num_timesteps"]

    gammas, bargammas, _, barsigmas = levy_noise_schedule(cfg['levy_alpha'], num_timesteps)
    T = len(gammas)
    gammas    = gammas.to(DEVICE)
    bargammas = bargammas.to(DEVICE)
    barsigmas = barsigmas.to(DEVICE)

    Sigma2    = barsigmas ** 2
    factor_dim = len(cfg["factors"])
    x = barsigmas[-1] * torch.randn(n, factor_dim, device=DEVICE)


    if cond is not None:
        assert cond.shape == (n, factor_dim)

    var_history, grad_history = [], []
    for t in range(T - 1, 0, -1):
        t_b      = torch.full((n,), t, dtype=torch.long, device=DEVICE)
        eps_pred = model(x, t_b, cond)

        # posterior contraction factor: Gamma_t = beta_t / (1 - bar_alpha_t)
        Gamma_t = 1 - (gammas[t] ** 2 * Sigma2[t - 1]) / (Sigma2[t] + 1e-8)

        # posterior mean
        mean = (x - barsigmas[t] * Gamma_t * eps_pred) / gammas[t]

        # posterior variance (scalar, deterministic)
        var = (Gamma_t * Sigma2[t - 1]).clamp(min=0.0)

        # DLPM-correct guidance: shift posterior mean by -s * w_t * var * ∂loss/∂x_t.
        # eps recomputed under enable_grad so ∂eps/∂x_t (dense Jacobian carrying
        # cross-factor coupling) flows into the guidance gradient. The condition is
        # held fixed (it is the given day-t state, not something we steer).
        if cond_fn is not None and t > 1:
            snr_t = bargammas[t] ** 2 / (barsigmas[t] ** 2 + 1e-8)
            w_t   = (snr_t / (snr_t + 1.0)) ** guidance_decay_pow
            with torch.enable_grad():
                x_g        = x.detach().requires_grad_(True)
                eps_pred_g = model(x_g, t_b, cond)
                x0_hat     = (x_g - barsigmas[t] * eps_pred_g) / bargammas[t]
                loss       = cond_fn(x0_hat).sum()
                grad       = torch.autograd.grad(loss, x_g)[0]
            mean = mean - guidance_scale * w_t * var * grad.detach()

        if t > 1:
            x = mean + var.sqrt() * torch.randn_like(x)
        else:
            x = mean   # no noise at last step

    return x, var_history, grad_history


def generate(model, scaler, cfg, cond, num_repeat=1, cond_fn=None,
             guidance_scale=5.0, guidance_decay_pow=1.0):
    """
    Single conditional step: given day-t conditions, sample day t+1.

    :param cond: (n, F) or (F,) condition rows in ORIGINAL (unscaled) space
    :param num_repeat: independent next-day draws per condition row (tiled along batch)
    :return: (samples, var_history, grad_history); samples is (n*num_repeat, F) original space
    """
    model.eval()
    cond = np.asarray(cond, dtype=np.float32)
    if cond.ndim == 1:
        cond = cond[None, :]
    cond_norm = torch.tensor(scaler.transform(cond), dtype=torch.float32, device=DEVICE)
    if num_repeat > 1:
        cond_norm = cond_norm.repeat(num_repeat, 1)

    x, vh, gh = _reverse(model, cfg, cond_norm, cond_fn=cond_fn,
                         guidance_scale=guidance_scale,
                         guidance_decay_pow=guidance_decay_pow)
    return scaler.inverse_transform(x.cpu().numpy()), vh, gh


def generate_uncond(model, scaler, cfg, num_samples):
    """
    Unconditional draw from the learned null token: sample the marginal p(F_0) with
    no previous-day condition. Requires a model trained with condition dropout
    (cond_drop_prob > 0). Used to self-start a rollout — no external seed needed.
    :return: (num_samples, F) array in ORIGINAL space.
    """
    model.eval()
    x, _, _ = _reverse(model, cfg, None, num_samples)
    return scaler.inverse_transform(x.cpu().numpy())


def generate_path(model, scaler, cfg, horizon, num_paths, seed_cond=None,
                  cond_fn=None, guidance_scale=5.0, guidance_decay_pow=1.0):
    """
    :param horizon:   number of days in each path
    :param num_paths: number of independent trajectories
    :param seed_cond: optional (F,) or (num_paths, F) seed in ORIGINAL space. None →
                      self-start from the null token (no seed required).
    :return: (num_paths, horizon, F) array in ORIGINAL space. Transpose to
             (num_paths, F, horizon) for the [M, N, D] layout in metrics/statistic_temporal.
    """
    model.eval()

    path = []

    if seed_cond is None:
        # Day 0: unconditional draw from the null token, then roll forward horizon-1 days.
        cur, _, _ = _reverse(model, cfg, None, num_paths)
        path.append(cur)
        n_cond_steps = horizon - 1
    else:
        seed = np.asarray(seed_cond, dtype=np.float32)
        if seed.ndim == 1:
            seed = np.broadcast_to(seed, (num_paths, seed.shape[0])).copy()
        assert seed.shape[0] == num_paths, f"seed rows {seed.shape[0]} != num_paths {num_paths}"
        cur = torch.tensor(scaler.transform(seed), dtype=torch.float32, device=DEVICE)
        n_cond_steps = horizon

    for _ in range(n_cond_steps):
        cur, _, _ = _reverse(model, cfg, cur, num_paths, cond_fn=cond_fn,
                             guidance_scale=guidance_scale,
                             guidance_decay_pow=guidance_decay_pow)
        path.append(cur)

    path = torch.stack(path, dim=1).cpu().numpy()      # (num_paths, horizon, F) normalized
    flat = scaler.inverse_transform(path.reshape(-1, path.shape[-1]))
    return flat.reshape(path.shape)


def generate_rejection(model, scaler, cfg, cond, *, cond_fn, num_samples=None,
                       guidance_scale=1.0, max_batches=50000):
    """
    Exact energy-reweighted sampling of day t+1 given a fixed day-t condition, via
    rejection on top of generate().

    Each iteration draws a batch of next-day candidates from the same `cond` (a single
    day-t row); each candidate is accepted with probability exp(-guidance_scale *
    cond_fn(x)). Returns np.ndarray in original space, shape (num_samples, factor_dim).

    :param cond: (F,) or (1, F) day-t condition row in ORIGINAL space
    """
    if num_samples is None:
        num_samples = cfg["num_generate"]
    batch_size = cfg["batch_size"]

    accepted = []
    n_tried  = 0

    for _ in range(max_batches):
        if len(accepted) >= num_samples:
            break

        batch_np, _, _ = generate(model, scaler, cfg, cond, num_repeat=batch_size)
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
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", help="experiment name; loads model/<exp_name>/checkpoints/*.pt")

    args = parser.parse_args()

    prefix = f"model/{args.exp_name}"
    with open(f"model/{args.exp_name}/cfg.yaml") as f:
        cfg = yaml.safe_load(f)

    ckpt_path = f"{prefix}/checkpoints/{cfg['ckpt_name']}.pt"

    ckpt   = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model  = FactorDenoiser(**ckpt["model_kwargs"]).to(DEVICE)
    model.load_state_dict(ckpt["ema_state"])
    scaler = ckpt["scaler"]

    horizon    = cfg["seq_len"]
    num_paths  = cfg["num_generate"]

    paths = generate_path(model, scaler, cfg, horizon, num_paths)

    out_path = f"{prefix}/samples/path_{cfg['ckpt_name']}_{num_paths}x{horizon}.npy"
    os.makedirs(f"{prefix}/samples", exist_ok=True)
    np.save(out_path, paths.transpose(0, 2, 1))
    print(f"Saved {paths.shape} paths (num_paths, horizon, F) → {out_path}")
