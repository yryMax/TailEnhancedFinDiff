import argparse
import os
import numpy as np
import torch
from tqdm.auto import tqdm
from factor_diffusion_train import FactorDenoiser, FactorDenoiserXS, build_denoiser, is_conditional_ckpt
from factor_diffusion_levy import levy_noise_schedule, sample_skewed_levy, sample_sas

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
#  Conditional (Markov-1 temporal) reverse process — FactorDenoiser            #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def _reverse(model, cfg, cond, n=None, vix=None, cond_fn=None, guidance_scale=5.0,
             guidance_decay_pow=1.0):
    """
    1. x_T ~ N(0, barsigmas[T-1]^2 I)
    2. For t = T-1 ... 1, with clean condition `cond` fed at every step:
           eps_pred = model(x_t, t, cond, vix)
           Gamma_t  = 1 - gammas[t]^2 * barsigmas[t-1]^2 / barsigmas[t]^2
           mean     = (x_t - barsigmas[t] * Gamma_t * eps_pred) / gammas[t]
           var      = Gamma_t * barsigmas[t-1]^2
           x_{t-1}  = mean + sqrt(var) * N(0,I)

    :param cond: (n, F) clean condition tensor on DEVICE, normalized space. None →
                 unconditional draw from the learned null token
    :param n:    batch size; inferred from cond.shape[0] when None (cond must be given then)
    :param vix:  optional (n, 1) normalized VIX condition for this step. None → the model
                 uses its learned null-VIX token (i.e. not conditioned on VIX)
    :param cond_fn: optional cond_fn(x0_hat) -> per-sample energy; enables gradient guidance
    :param guidance_scale: guidance strength `s`; mean -= s * w_t * var * grad
    :param guidance_decay_pow: exponent p in w_t = (SNR_t / (SNR_t + 1))**p (Karras/EDM SNR weight)
    :return: (x, var_history, grad_history); x is (n, F) normalized tensor on DEVICE
    """
    num_timesteps = cfg["num_timesteps"]
    if n is None:
        n = cond.shape[0]

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
        eps_pred = model(x, t_b, cond, vix=vix)

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
                eps_pred_g = model(x_g, t_b, cond, vix=vix)
                x0_hat     = (x_g - barsigmas[t] * eps_pred_g) / bargammas[t]
                loss       = cond_fn(x0_hat).sum()
                grad       = torch.autograd.grad(loss, x_g)[0]
            mean = mean - guidance_scale * w_t * var * grad.detach()

        if t > 1:
            x = mean + var.sqrt() * torch.randn_like(x)
        else:
            x = mean   # no noise at last step

    return x, var_history, grad_history


# --------------------------------------------------------------------------- #
#  Cross-sectional (unconditional) reverse process — FactorDenoiserXS          #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def _generate_xs(model, scaler, cfg, num_samples=None, cond_fn=None,
                 guidance_scale=5.0, guidance_decay_pow=1.0, L=None):
    """
    DLPM reverse process for cross-sectional checkpoints. alpha=2 automatically
    degenerates to DDPM.

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


# --------------------------------------------------------------------------- #
#  Unified entry points                                                        #
# --------------------------------------------------------------------------- #

def generate(model, scaler, cfg, cond=None, num_samples=None, num_repeat=1,
             cond_fn=None, guidance_scale=5.0, guidance_decay_pow=1.0, L=None):
    """
    Unified sampling entry point; dispatches on `cond` and the model class.

    - cond given  (conditional model): single Markov step — given day-t condition rows,
      sample day t+1. `num_repeat` tiles each condition row for independent draws.
    - cond None + conditional model: unconditional draw from the learned null token.
    - cond None + cross-sectional model: DLPM/DDPM reverse process with stochastic
      A-chain (and optional L-coloured noise).

    :param cond: (n, F) or (F,) condition rows in ORIGINAL (unscaled) space; None → unconditional
    :param num_samples: unconditional sample count; defaults to cfg["num_generate"]
    :param num_repeat: independent next-day draws per condition row (conditional path only)
    :param L: optional Cholesky factor (cross-sectional path only)
    :return: (samples, var_history, grad_history); samples in ORIGINAL space
    """
    model.eval()

    if cond is not None:
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

    if hasattr(model, "null_cond"):
        # conditional model, unconditional draw via the learned null token
        if num_samples is None:
            num_samples = cfg["num_generate"]
        batch_size = cfg["batch_size"]
        batches, vh, gh = [], [], []
        for start in range(0, num_samples, batch_size):
            n = min(batch_size, num_samples - start)
            x, vh_i, gh_i = _reverse(model, cfg, None, n=n, cond_fn=cond_fn,
                                     guidance_scale=guidance_scale,
                                     guidance_decay_pow=guidance_decay_pow)
            batches.append(x.cpu())
            if start == 0:
                vh, gh = vh_i, gh_i
        return scaler.inverse_transform(torch.cat(batches).numpy()), vh, gh

    return _generate_xs(model, scaler, cfg, num_samples=num_samples, cond_fn=cond_fn,
                        guidance_scale=guidance_scale,
                        guidance_decay_pow=guidance_decay_pow, L=L)


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


@torch.no_grad()
def generate_style_transfer(model, scaler, cfg, x0_obs, cond_fn, *,
                    strength=0.6, guidance_scale=5.0, guidance_decay_pow=1.0):
    """
    SDEdit-style per-sample re-synthesis under an energy condition (cross-sectional
    checkpoints only).

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


# --------------------------------------------------------------------------- #
#  Temporal rollout (conditional checkpoints)                                  #
# --------------------------------------------------------------------------- #

def vix_cond_gen(vix_cond, length, num_paths, vix_scaler=None):
    """
    Expand a VIX spec into a length-`length` list of per-step conditions for the rollout,
    so the reverse loop is branch-free. Minimal version: constant or None only.

      vix_cond is None  -> [None] * length            (not conditioned on VIX; uses null_vix)
      vix_cond is scalar -> the same VIX value on every day, shared across all paths

    Each non-None element is a (num_paths, 1) tensor on DEVICE. When `vix_scaler` is given the
    raw VIX level is mapped to training space by the scaler itself (it encapsulates log +
    standardize); without a scaler the value is used as-is (assumed already normalized).
    """
    if vix_cond is None:
        return [None] * length
    v = float(vix_cond)
    if vix_scaler is not None:
        v = float(vix_scaler.transform([[v]])[0, 0])           # scaler encapsulates log + standardize
    step = torch.full((num_paths, 1), v, dtype=torch.float32, device=DEVICE)
    return [step] * length


def generate_path(model, scaler, cfg, horizon, num_paths, seed_cond=None,
                  vix_cond=None, vix_scaler=None, cond_fn=None,
                  guidance_scale=5.0, guidance_decay_pow=1.0):
    """
    :param horizon:   number of days in each path
    :param num_paths: number of independent trajectories
    :param seed_cond: optional (F,) or (num_paths, F) seed in ORIGINAL space. None →
                      self-start from the null token (no seed required).
    :param vix_cond:  optional constant VIX (shared by all paths/days) or None. Expanded by
                      vix_cond_gen into a length-horizon per-step sequence; None → no VIX.
    :param vix_scaler: fitted VIX scaler (from the checkpoint); used to map a raw VIX level
                      into training space. None → vix_cond is taken as already normalized.
    :return: (num_paths, horizon, F) array in ORIGINAL space. Transpose to
             (num_paths, F, horizon) for the [M, N, D] layout in metrics/statistic_temporal.
    """
    model.eval()

    vix_seq = vix_cond_gen(vix_cond, horizon, num_paths, vix_scaler)   # per-step VIX (None or (num_paths,1))

    path = []

    if seed_cond is None:
        # Day 0: unconditional draw from the null token, then roll forward horizon-1 days.
        cur, _, _ = _reverse(model, cfg, None, num_paths, vix=vix_seq[0])
        path.append(cur)
    else:
        seed = np.asarray(seed_cond, dtype=np.float32)
        if seed.ndim == 1:
            seed = np.broadcast_to(seed, (num_paths, seed.shape[0])).copy()
        assert seed.shape[0] == num_paths, f"seed rows {seed.shape[0]} != num_paths {num_paths}"
        cur = torch.tensor(scaler.transform(seed), dtype=torch.float32, device=DEVICE)

    # produce the remaining days; vix_seq[d] aligns with the day d being generated
    for d in tqdm(range(len(path), horizon), desc=f"rollout (horizon={horizon})", leave=False):
        cur, _, _ = _reverse(model, cfg, cur, num_paths, vix=vix_seq[d], cond_fn=cond_fn,
                             guidance_scale=guidance_scale,
                             guidance_decay_pow=guidance_decay_pow)
        path.append(cur)

    path = torch.stack(path, dim=1).cpu().numpy()      # (num_paths, horizon, F) normalized
    flat = scaler.inverse_transform(path.reshape(-1, path.shape[-1]))
    return flat.reshape(path.shape)


def generate_rejection(model, scaler, cfg, cond=None, *, cond_fn, num_samples=None,
                       guidance_scale=1.0, max_batches=50000, L=None):
    """
    Exact energy-reweighted sampling via rejection on top of generate().

    Each iteration draws a candidate batch — unconditional when `cond` is None, or
    next-day draws from the fixed day-t condition row otherwise; each candidate is
    accepted with probability exp(-guidance_scale * cond_fn(x)). Returns np.ndarray
    in original space, shape (num_samples, factor_dim).

    :param cond: optional (F,) or (1, F) day-t condition row in ORIGINAL space
    """
    if num_samples is None:
        num_samples = cfg["num_generate"]
    batch_size = cfg["batch_size"]

    accepted = []
    n_tried  = 0

    for _ in range(max_batches):
        if len(accepted) >= num_samples:
            break

        if cond is not None:
            batch_np, _, _ = generate(model, scaler, cfg, cond, num_repeat=batch_size)
        else:
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
    import yaml
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", help="experiment name; loads model/<exp_name>/checkpoints/*.pt")
    parser.add_argument("--ckpt", default=None,
                        help="checkpoint filename (without .pt); defaults to cfg['ckpt_name']")
    parser.add_argument("--seed-oos", action="store_true",
                        help="[temporal ckpt] seed the rollout from a random OOS day")
    parser.add_argument("--seed-date", default=None,
                        help="[temporal ckpt] specific OOS date (YYYY-MM-DD) to seed from")
    parser.add_argument("--vix", type=float, default=None,
                        help="[temporal ckpt] constant VIX level to condition on (e.g. 40); omit → no VIX")
    args = parser.parse_args()

    prefix = f"model/{args.exp_name}"
    with open(f"{prefix}/cfg.yaml") as f:
        cfg = yaml.safe_load(f)

    ckpt_name = args.ckpt or cfg["ckpt_name"]
    ckpt_path = f"{prefix}/checkpoints/{ckpt_name}.pt"

    ckpt   = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model  = build_denoiser(ckpt, DEVICE)
    scaler = ckpt["scaler"]
    cfg    = ckpt.get("cfg", cfg)

    os.makedirs(f"{prefix}/samples", exist_ok=True)

    if is_conditional_ckpt(ckpt):
        # ---- temporal rollout ------------------------------------------------ #
        from factor_model import load_model, dump_to_achievement

        vix_scaler = ckpt.get("vix_scaler")
        horizon    = cfg["seq_len"]
        num_paths  = cfg["num_generate"]
        factors    = cfg["factors"]

        cond_seed, dates = None, None
        if args.seed_oos or args.seed_date is not None:
            oos = pd.read_csv(f"{prefix}/test/factors.csv", index_col=0, parse_dates=True)[factors]
            if args.seed_date is not None:
                seed_ts = pd.Timestamp(args.seed_date)
                if seed_ts not in oos.index:
                    raise SystemExit(f"--seed-date {args.seed_date} not in OOS index "
                                     f"({oos.index.min().date()} .. {oos.index.max().date()})")
                i = oos.index.get_loc(seed_ts)
            else:
                i = int(np.random.randint(len(oos) - 1))   # leave at least one following day
            cond_seed = oos.iloc[i].to_numpy(np.float32)    # (F,) day-t condition, shared by all paths
            # real OOS trading days for the generated horizon (the path starts at i+1, the day
            # after the seed). Use the actual calendar so parquet dates align exactly with the
            # OOS ground truth; only if the horizon runs past the end of OOS do we extend with
            # synthetic business days.
            dates = oos.index[i + 1: i + 1 + horizon]
            if len(dates) < horizon:
                anchor = (dates[-1] if len(dates) else oos.index[i]) + pd.offsets.BDay(1)
                dates  = dates.append(pd.bdate_range(anchor, periods=horizon - len(dates)))
            print(f"[seed] OOS day {oos.index[i].date()} (idx {i}) → "
                  f"path {dates[0].date()} .. {dates[-1].date()}")

        if args.vix is not None and vix_scaler is None:
            raise SystemExit("--vix given but checkpoint has no vix_scaler (model trained without VIX)")
        paths = generate_path(model, scaler, cfg, horizon, num_paths, seed_cond=cond_seed,
                              vix_cond=args.vix, vix_scaler=vix_scaler)
        paths = paths.transpose(0, 2, 1)                     # (num_paths, F, horizon)
        if args.vix is not None:
            print(f"[vix] conditioned on constant VIX = {args.vix}")

        stem = f"{prefix}/samples/path_{ckpt_name}_{num_paths}x{horizon}"
        np.save(f"{stem}.npy", paths)
        print(f"Saved {paths.shape} paths (num_paths, F, horizon) → {stem}.npy")

        fmodel = load_model(prefix)
        if args.seed_oos or args.seed_date is not None:
            dump_to_achievement(fmodel, f"{stem}.npy",
                                out_dir="achievement/with_date", dates=dates)
        else:
            dump_to_achievement(fmodel, f"{stem}.npy", out_dir="achievement")
    else:
        # ---- cross-sectional samples ---------------------------------------- #
        L = ckpt.get("L_noise")
        out_path = f"{prefix}/samples/factor_{ckpt_name}_{cfg['num_generate']}.npy"
        samples, _, _ = generate(model, scaler, cfg, L=L)
        np.save(out_path, samples)
        print(f"Saved {samples.shape} samples → {out_path}")
