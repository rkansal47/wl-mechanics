"""
Train WLF-UOT+ on EB 5D and extract couplings between consecutive marginals.

The trained WLF model learns a potential s(t,x) whose gradient gives the velocity
field. We use the unbalanced ODE generator to flow samples from one training
time point to the next, then match the flowed samples to actual samples at
the target time using the transport plan to build the coupling.

Usage:
    python train_and_extract_couplings.py --test-id 1 --seed 0

Outputs:
    results/wlf_couplings_fold{test_id}_seed{seed}.npz
"""

import argparse
import functools
import os
import time
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["WANDB_MODE"] = "disabled"

import jax
import jax.numpy as jnp
import flax
import flax.jax_utils as flax_utils
import numpy as np
import ot
from jax import random

import losses
import datasets
import train_utils as tutils
import eval_utils as eutils
from models import utils as mutils
from models import mlp  # noqa: F401 – registers models

LOSS_TO_CONFIG = {
    "ubotp": "configs/embrio/ubotp.py",
}


def load_config(loss_key, test_id, seed):
    import importlib.util
    cfg_path = LOSS_TO_CONFIG[loss_key]
    spec = importlib.util.spec_from_file_location("cfg", cfg_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    config = mod.get_config()
    config.seed = seed
    config.data.test_id = test_id
    config.model_q.n_marginals = 4
    if not hasattr(config, "metric"):
        config.metric = "w1"
    if not hasattr(config, "lambd"):
        config.lambd = 0.1
    return config


def train_and_get_state(config, quiet=False):
    """Train WLF-UOT+ and return the trained state plus data iterators."""
    key = random.PRNGKey(config.seed)
    key, *init_key = random.split(key, 3)

    model_s, initial_params_s = mutils.init_model_s(init_key[0], config.model_s)
    optimizer_s = tutils.get_optimizer(config.optimizer_s)
    opt_state_s = optimizer_s.init(initial_params_s)
    time_sampler, init_sampler_state = tutils.get_time_sampler(config)

    state_s = mutils.State(
        step=1, opt_state=opt_state_s, model_params=initial_params_s,
        ema_rate=config.model_s.ema_rate, params_ema=initial_params_s,
        sampler_state=init_sampler_state, key=key, wandbid=0,
    )

    model_q, initial_params_q = mutils.init_model_q(init_key[1], config.model_q)
    optimizer_q = tutils.get_optimizer(config.optimizer_q)
    opt_state_q = optimizer_q.init(initial_params_q)

    state_q = mutils.State(
        step=1, opt_state=opt_state_q, model_params=initial_params_q,
        ema_rate=config.model_q.ema_rate, params_ema=initial_params_q,
        sampler_state=init_sampler_state, key=key, wandbid=0,
    )

    loss_fn = losses.get_loss(config, model_s, model_q, time_sampler, train=True)
    step_fn = tutils.get_step_fn(config, optimizer_s, optimizer_q, loss_fn)
    step_fn = jax.pmap(functools.partial(jax.lax.scan, step_fn), axis_name="batch")

    key, *init_key = random.split(key, 3)
    batch_iterator, inv_scaler = datasets.get_batch_iterator(config, init_key[0])
    test_iterator, _ = datasets.get_batch_iterator(config, init_key[1], eval=True)
    test_data = test_iterator()

    pairwise_dist = jax.jit(
        lambda _x, _y: jnp.linalg.norm(_x[:, None, :] - _y[None, :, :], axis=-1)
    )
    ode_generator, _ = eutils.get_generator(model_s, config)
    ode_generator = jax.jit(ode_generator)

    state_s = flax_utils.replicate(state_s)
    state_q = flax_utils.replicate(state_q)
    key = jax.random.fold_in(key, jax.process_index())

    n_iters = config.train.n_iters
    log_every = config.train.log_every
    eval_every = config.train.eval_every

    best_w1 = float("inf")
    wall_start = time.time()

    for step in range(1, n_iters + 1, config.train.n_jitted_steps):
        key, batch_key = random.split(key)
        batch = batch_iterator(batch_key)
        key, *next_key = random.split(key, num=jax.local_device_count() + 1)
        next_key = jnp.asarray(next_key)
        (_, state_s, state_q), (total_loss, metrics) = step_fn(
            (next_key, state_s, state_q), batch
        )

        if step % log_every == 0 and not quiet:
            loss_val = flax.jax_utils.unreplicate(total_loss).mean().item()
            elapsed = time.time() - wall_start
            print(f"  step {step:6d}/{n_iters} | loss {loss_val:.4f} | {elapsed:.0f}s", flush=True)

        if step % eval_every == 0:
            X_init, t_init, X_end, t_end = test_data
            for i in range(len(X_init)):
                key, eval_key = random.split(key)
                (ode_solution, weights), _ = ode_generator(
                    eval_key, flax_utils.unreplicate(state_s),
                    (X_init[i], t_init[i], X_end[i], t_end[i]),
                )
                w1 = eutils.get_w1(
                    pairwise_dist(inv_scaler(ode_solution), inv_scaler(X_end[i])), weights,
                )
                if w1 < best_w1:
                    best_w1 = float(w1)
                if not quiet:
                    print(f"    eval step {step}: W1={w1:.4f} (best={best_w1:.4f})", flush=True)

    final_state_s = flax_utils.unreplicate(state_s)
    print(f"Training complete. Best W1={best_w1:.4f}")
    return model_s, final_state_s, config, inv_scaler, key


def extract_wlf_couplings(model_s, state_s, config, inv_scaler, key):
    """
    Extract couplings between consecutive training marginals using the trained
    WLF model. Flows samples from t_i to t_{i+1} via the ODE, then computes
    an OT plan between the flowed samples and the actual samples at t_{i+1}
    to get the coupling indices.
    """
    ode_generator, _ = eutils.get_generator(model_s, config)
    ode_generator = jax.jit(ode_generator)

    # Reload data to get the raw marginals (before the held-out removal)
    key_data = random.PRNGKey(0)
    X_train_all, _, _, _, times = datasets.get_data(config, key_data)

    # X_train_all is a list of arrays per timepoint (in normalized space)
    # times are the normalized timepoints
    all_day_indices = list(range(len(X_train_all)))
    test_id = config.data.test_id

    # Get training indices (exclude the held-out timepoint)
    if test_id is not None:
        train_indices = [i for i in all_day_indices if i != test_id]
    else:
        train_indices = all_day_indices

    print(f"Training marginals: {train_indices} (held out: {test_id})")

    # Normalized times for training marginals
    t_normalized = np.linspace(0.0, 1.0, len(X_train_all))

    pairwise_dist = jax.jit(
        lambda _x, _y: jnp.linalg.norm(_x[:, None, :] - _y[None, :, :], axis=-1)
    )

    couplings = {}
    for idx in range(len(train_indices) - 1):
        src_day = train_indices[idx]
        tgt_day = train_indices[idx + 1]

        X_src = jnp.array(X_train_all[src_day])
        X_tgt = jnp.array(X_train_all[tgt_day])
        t_src = t_normalized[src_day]
        t_tgt = t_normalized[tgt_day]

        print(f"Flowing day {src_day} (t={t_src:.3f}) -> day {tgt_day} (t={t_tgt:.3f})")
        print(f"  Source: {X_src.shape}, Target: {X_tgt.shape}")

        key, eval_key = random.split(key)
        (flowed, weights), n_steps = ode_generator(
            eval_key, state_s, (X_src, t_src, X_tgt, t_tgt)
        )
        print(f"  ODE integration used {n_steps} steps")

        # Resample if weights are present (unbalanced case)
        if weights is not None:
            w = np.array(weights).ravel()
            w = np.maximum(w, 0)
            w /= w.sum()
            flowed_np = np.array(flowed)
        else:
            w = np.ones(len(flowed)) / len(flowed)
            flowed_np = np.array(flowed)

        tgt_np = np.array(X_tgt)

        # Compute OT plan between flowed samples and target samples
        a = w.astype(np.float64)
        b = np.ones(len(tgt_np), dtype=np.float64) / len(tgt_np)
        M = np.array(pairwise_dist(jnp.array(flowed_np), jnp.array(tgt_np))).astype(np.float64)
        plan = ot.emd(a, b, M, numItermax=int(1e7))
        mapping = np.argmax(plan, axis=1)

        couplings[(src_day, tgt_day)] = mapping
        print(f"  Coupling computed: {len(mapping)} mappings")

    return couplings, X_train_all, t_normalized, train_indices


def main():
    parser = argparse.ArgumentParser(description="Train WLF-UOT+ and extract couplings")
    parser.add_argument("--test-id", type=int, default=1, help="Leave-one-out fold (1, 2, or 3)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"WLF-UOT+ | seed={args.seed} | test_id={args.test_id}")
    print(f"{'='*60}")

    config = load_config("ubotp", args.test_id, args.seed)

    # Train
    model_s, state_s, config, inv_scaler, key = train_and_get_state(config, quiet=args.quiet)

    # Extract couplings
    print(f"\n{'='*60}")
    print("Extracting WLF couplings...")
    print(f"{'='*60}")

    couplings, X_all, t_normalized, train_indices = extract_wlf_couplings(
        model_s, state_s, config, inv_scaler, key
    )

    # Save couplings
    save_data = {
        "train_indices": np.array(train_indices),
        "t_normalized": t_normalized,
        "test_id": args.test_id,
        "seed": args.seed,
        "dim": config.data.dim,
    }
    for (src, tgt), mapping in couplings.items():
        save_data[f"coupling_{src}_{tgt}"] = mapping
    for i, X in enumerate(X_all):
        save_data[f"X_{i}"] = np.array(X)

    output_path = results_dir / f"wlf_couplings_fold{args.test_id}_seed{args.seed}.npz"
    np.savez(output_path, **save_data)
    print(f"\nCouplings saved to {output_path}")

    # Print summary
    print(f"\nSummary:")
    print(f"  Test fold: {args.test_id}")
    print(f"  Training marginals: {train_indices}")
    print(f"  Couplings: {list(couplings.keys())}")
    for (src, tgt), mapping in couplings.items():
        print(f"    {src} -> {tgt}: {len(mapping)} mappings")


if __name__ == "__main__":
    main()
