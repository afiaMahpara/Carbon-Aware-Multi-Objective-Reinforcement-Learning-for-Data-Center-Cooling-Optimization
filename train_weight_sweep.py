"""
train_weight_sweep.py
CA-MORL: Weight-Sweep Training Script
======================================
Trains one PPO agent per weight vector (w_E, w_W, w_C) across
multiple random seeds and saves per-seed results to CSV.

Usage
-----
    python train_weight_sweep.py [--timesteps 1000000] [--seeds 3]
                                 [--ci_csv path/to/ci.csv]
                                 [--sinergym_env <env_id>]
                                 [--output_dir logs]

Output
------
    logs/weight_sweep_results_corrected.csv   (all seeds × configs)
    logs/reward_curves/<config>_seed<s>.npy   (per-seed return arrays)
    checkpoints/<config>_seed<s>.pt           (saved model weights)

Reviewer notes addressed
------------------------
- Reports all 3 seeds individually so mean ± std can be computed
- Saves reward curves for convergence plotting
- Logs the actual CI series statistics so r=1.000 can be diagnosed
- Writes per-config per-seed checkpoint to detect loading bugs
"""

import argparse
import os
import csv
import time
import numpy as np

from environment import make_env
from ppo_agent import PPOAgent

# ─────────────────────────────────────────────────────────────────────────────
# Weight configurations
# ─────────────────────────────────────────────────────────────────────────────
WEIGHT_CONFIGS = [
    # name,                w_E,  w_W,  w_C
    ('Energy Only',        1.00, 0.00, 0.00),
    ('Water Only',         0.00, 1.00, 0.00),
    ('Carbon Only',        0.00, 0.00, 1.00),
    ('Energy+Water',       0.50, 0.50, 0.00),
    ('Energy+Carbon',      0.50, 0.00, 0.50),
    ('Water+Carbon',       0.00, 0.50, 0.50),
    ('Equal (Balanced)',   0.34, 0.33, 0.33),
    ('Carbon Focused',     0.20, 0.20, 0.60),
    ('Carbon Heavy',       0.10, 0.10, 0.80),
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_one_config(name: str, weights: tuple, seed: int, args) -> dict:
    """Train one PPO agent for one (config, seed) pair."""
    np.random.seed(seed)

    env, reward_fn = make_env(
        weights=weights,
        ci_csv=args.ci_csv,
        seed=seed,
        sinergym_env_id=args.sinergym_env,
    )

    agent = PPOAgent(
        obs_dim=11, act_dim=2,
        lr=3e-4, clip_ratio=0.2, gamma=0.99, gae_lambda=0.95,
        update_epochs=10, batch_size=64, buffer_size=2048,
        vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, hidden=256,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    obs, _ = env.reset()
    total_timesteps = args.timesteps

    episode_rewards, episode_lengths = [], []
    ep_reward, ep_len = 0.0, 0
    ep_energy = ep_water = ep_carbon = 0.0
    ep_violations = 0

    total_energy = total_water = total_carbon = 0.0
    all_violations = 0
    all_rewards = []
    return_curve = []   # episodic returns — for convergence plots

    t0 = time.time()
    for step in range(total_timesteps):
        action, log_prob, value = agent.select_action(obs)

        # Clip to env bounds
        action_clipped = np.clip(
            action,
            np.array([15.0, 0.1], dtype=np.float32),
            np.array([22.0, 1.0], dtype=np.float32),
        )

        next_obs, (energy, water, carbon), terminated, truncated, info = \
            env.step(action_clipped)

        zone_temps = next_obs[:2]
        reward, _ = reward_fn(energy, water, carbon, zone_temps)

        done = terminated or truncated
        agent.store_transition(obs, action, reward, float(done), log_prob, value)

        ep_reward   += reward
        ep_energy   += energy
        ep_water    += water
        ep_carbon   += carbon
        ep_len      += 1
        if np.any(zone_temps > 32.0):
            ep_violations += 1

        if agent.buffer.full():
            agent.update(next_obs)

        if done:
            total_energy   += ep_energy
            total_water    += ep_water
            total_carbon   += ep_carbon
            all_violations += ep_violations
            all_rewards.append(ep_reward)
            return_curve.append(ep_reward)

            ep_reward = ep_len = ep_energy = ep_water = ep_carbon = ep_violations = 0.0
            obs, _ = env.reset()
        else:
            obs = next_obs

    elapsed = time.time() - t0

    # ── Save reward curve ──────────────────────────────────────────────────────
    curve_dir = os.path.join(args.output_dir, 'reward_curves')
    os.makedirs(curve_dir, exist_ok=True)
    safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'p')
    np.save(os.path.join(curve_dir, f'{safe_name}_seed{seed}.npy'),
            np.array(return_curve))

    # ── Save checkpoint ────────────────────────────────────────────────────────
    ck_dir = os.path.join(args.output_dir, '..', 'checkpoints')
    os.makedirs(ck_dir, exist_ok=True)
    agent.save(os.path.join(ck_dir, f'{safe_name}_seed{seed}.pt'))

    # ── CI statistics (for r=1.000 diagnosis) ─────────────────────────────────
    ci_vals = env.ci.ci_series if hasattr(env, 'ci') else \
              env._env.ci.ci_series if hasattr(env, '_env') else np.array([0.5])
    ci_mean = float(ci_vals.mean())
    ci_std  = float(ci_vals.std())
    ci_cv   = ci_std / ci_mean

    mean_reward = float(np.mean(all_rewards)) if all_rewards else float('nan')

    result = {
        'config_name':          name,
        'weight_energy':        weights[0],
        'weight_water':         weights[1],
        'weight_carbon':        weights[2],
        'seed':                 seed,
        'total_energy_kWh':     total_energy,
        'total_water_L':        total_water,
        'total_carbon_kgCO2':   total_carbon,
        'mean_reward':          mean_reward,
        'violations':           all_violations,
        'carbon_per_energy_ratio': total_carbon / max(total_energy, 1e-9),
        'ci_mean':              ci_mean,
        'ci_std':               ci_std,
        'ci_cv':                ci_cv,
        'training_time_s':      elapsed,
    }

    print(f"  [{name}|seed={seed}]  E={total_energy:.0f} kWh  "
          f"W={total_water:.0f} L  C={total_carbon:.0f} kgCO₂  "
          f"CI_cv={ci_cv:.3f}  t={elapsed:.0f}s")

    # ── Reviewer warning if CI was effectively constant ────────────────────────
    if ci_cv < 0.10:
        print(f"  ⚠  WARNING: CI CV={ci_cv:.3f} < 0.10  "
              "→ carbon-aware and energy-only policies may be indistinguishable. "
              "Use a real CI CSV or enable improved synthetic CI.")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='CA-MORL weight-sweep training')
    parser.add_argument('--timesteps',    type=int,   default=1_000_000)
    parser.add_argument('--seeds',        type=int,   default=3)
    parser.add_argument('--ci_csv',       type=str,   default=None,
                        help='Path to real CI CSV (ElectricityMap/WattTime export)')
    parser.add_argument('--sinergym_env', type=str,   default=None,
                        help='Sinergym env ID if installed')
    parser.add_argument('--output_dir',   type=str,   default='logs')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    out_csv = os.path.join(args.output_dir, 'weight_sweep_results_corrected.csv')
    fieldnames = ['config_name','weight_energy','weight_water','weight_carbon',
                  'seed','total_energy_kWh','total_water_L','total_carbon_kgCO2',
                  'mean_reward','violations','carbon_per_energy_ratio',
                  'ci_mean','ci_std','ci_cv','training_time_s']

    total_runs = len(WEIGHT_CONFIGS) * args.seeds
    print("=" * 70)
    print(f"CA-MORL Weight-Sweep Training")
    print(f"  Configs : {len(WEIGHT_CONFIGS)}")
    print(f"  Seeds   : {args.seeds}")
    print(f"  Steps   : {args.timesteps:,} per run")
    print(f"  Total   : {total_runs} runs")
    print(f"  Output  : {out_csv}")
    print("=" * 70)

    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        run_idx = 0
        for name, w_E, w_W, w_C in WEIGHT_CONFIGS:
            weights = (w_E, w_W, w_C)
            print(f"\n[{run_idx+1}/{len(WEIGHT_CONFIGS)}] Config: {name}  "
                  f"weights=({w_E:.2f},{w_W:.2f},{w_C:.2f})")
            for seed in range(args.seeds):
                result = run_one_config(name, weights, seed, args)
                writer.writerow({k: result[k] for k in fieldnames})
                f.flush()
            run_idx += 1

    print("\n" + "=" * 70)
    print(f"Training complete. Results saved to {out_csv}")
    print("=" * 70)


if __name__ == '__main__':
    main()
