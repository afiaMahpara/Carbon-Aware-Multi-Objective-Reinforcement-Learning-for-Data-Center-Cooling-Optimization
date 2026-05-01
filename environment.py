"""
environment.py
CA-MORL: Carbon-Aware Multi-Objective RL Environment Wrapper
=============================================================
Wraps the Sinergym/EnergyPlus data-centre environment and injects:
  - Grid carbon-intensity (CI) as an observation
  - Cooling-tower water consumption as an observation
  - Vectorised multi-objective reward (energy, water, carbon)
  - ASHRAE TC9.9 hard-safety penalty

This module is designed so that, when Sinergym is NOT installed
(e.g. analysis-only machines), it falls back to a lightweight
synthetic environment with identical API for testing / debugging.
"""

import os
import numpy as np
from typing import Tuple, Optional, Dict, Any

# ── Optional Sinergym import ─────────────────────────────────────────────────
try:
    import sinergym  # noqa: F401
    import gymnasium as gym
    _SINERGYM_AVAILABLE = True
except ImportError:
    _SINERGYM_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Carbon Intensity Module
# ─────────────────────────────────────────────────────────────────────────────

class CarbonIntensityModule:
    """
    Provides hourly grid carbon-intensity values for the simulation.

    Priority
    --------
    1. Load from a CSV file (real ElectricityMap / WattTime export)
    2. Fall back to a synthetic time-series with realistic diurnal +
       seasonal variation (0.2 – 0.8 kgCO₂/kWh).

    Parameters
    ----------
    csv_path : str or None
        Path to a CSV with columns [hour, carbon_intensity_kgco2_per_kwh].
        Rows are indexed 0..8759 matching EnergyPlus hourly timesteps.
    seed : int
        Random seed for synthetic fallback.

    Notes
    -----
    REVIEWER COMMENT addressed here: the original submission used a
    synthetic series that was effectively constant (CV ≈ 4.8 %).
    The improved synthetic generator below produces a series with
    diurnal variance ≈ ±0.15 kgCO₂/kWh so that carbon-aware policies
    genuinely differ from energy-only policies.
    """

    def __init__(self, csv_path: Optional[str] = None, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        if csv_path and os.path.exists(csv_path):
            import pandas as pd
            df = pd.read_csv(csv_path)
            col = [c for c in df.columns if 'carbon' in c.lower() or 'ci' in c.lower()][0]
            self.ci_series = df[col].values.astype(np.float32)
            print(f"[CarbonIntensityModule] Loaded real CI from {csv_path}  "
                  f"(mean={self.ci_series.mean():.3f}, "
                  f"std={self.ci_series.std():.3f} kgCO₂/kWh)")
        else:
            self.ci_series = self._synthetic_ci()
            print(f"[CarbonIntensityModule] Using IMPROVED synthetic CI  "
                  f"(mean={self.ci_series.mean():.3f}, "
                  f"std={self.ci_series.std():.3f} kgCO₂/kWh)")

    def _synthetic_ci(self, n_hours: int = 8760) -> np.ndarray:
        """
        Generate a realistic synthetic CI time-series.
        Components:
          - Seasonal trend:  low in summer (more solar), high in winter
          - Diurnal pattern: high at night (coal dominates), low at noon (solar peak)
          - Random noise
        """
        hours = np.arange(n_hours)
        day_of_year = hours // 24
        hour_of_day = hours % 24

        # Seasonal: cosine with peak at day 0/365 (winter), trough at day 182 (summer)
        seasonal = 0.10 * np.cos(2 * np.pi * day_of_year / 365)

        # Diurnal: high at night (hour 0-6), low midday (hour 12)
        diurnal = -0.15 * np.cos(2 * np.pi * hour_of_day / 24)

        # Base + components
        ci = 0.50 + seasonal + diurnal
        ci += self.rng.normal(0, 0.03, n_hours)   # small noise
        ci = np.clip(ci, 0.20, 0.80).astype(np.float32)
        return ci

    def get(self, timestep: int) -> float:
        """Return CI at given EnergyPlus hourly timestep (0-indexed)."""
        return float(self.ci_series[timestep % len(self.ci_series)])


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Fallback Environment (for CI / analysis machines)
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticDataCenterEnv:
    """
    Lightweight synthetic data-centre environment with identical API
    to the Sinergym-wrapped version. Used when Sinergym/EnergyPlus is
    not installed (e.g. CI runners, analysis notebooks).

    Observation: 11-D  [zone1_T, zone2_T, supply_T, outdoor_T,
                         outdoor_RH, IT_power, HVAC_energy, water_use,
                         grid_CI, sin_tod, cos_tod]
    Action:      2-D   [chilled_water_setpoint ∈ [15,22],
                         supply_air_flow ∈ [0.1,1.0]]
    """

    OBS_DIM = 11
    ACT_DIM = 2

    # Action bounds
    ACT_LOW  = np.array([15.0, 0.1], dtype=np.float32)
    ACT_HIGH = np.array([22.0, 1.0], dtype=np.float32)

    def __init__(self, ci_module: CarbonIntensityModule, seed: int = 42):
        self.ci = ci_module
        self.rng = np.random.default_rng(seed)
        self.timestep = 0
        self._zone_temps = np.array([22.0, 22.0], dtype=np.float32)
        self.episode_length = 8760

    def reset(self):
        self.timestep = 0
        self._zone_temps = np.array([22.0, 22.0], dtype=np.float32)
        return self._obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, self.ACT_LOW, self.ACT_HIGH)
        cwsp, flow = action[0], action[1]

        # Simplified physics
        outdoor_T = 15.0 + 12.0 * np.sin(2*np.pi*(self.timestep % 8760)/8760)
        heat_load  = 20.0 + self.rng.uniform(-2, 2)
        cooling    = (22.0 - cwsp) * flow * 1.5
        dT         = (heat_load - cooling) * 0.05
        self._zone_temps = np.clip(self._zone_temps + dT, 15.0, 35.0)

        HVAC_energy = flow * (22.0 - cwsp) * 0.8 + 0.5        # kWh
        water_use   = flow * max(0.0, outdoor_T - 10) * 0.3   # L
        ci_val      = self.ci.get(self.timestep)
        carbon      = HVAC_energy * ci_val                     # kgCO₂

        self.timestep += 1
        terminated = (self.timestep >= self.episode_length)
        return self._obs(), (HVAC_energy, water_use, carbon), terminated, False, {}

    def _obs(self):
        tod  = self.timestep % 24
        doy  = self.timestep // 24 % 365
        ci   = self.ci.get(self.timestep)
        return np.array([
            self._zone_temps[0],
            self._zone_temps[1],
            20.0,                                          # supply_T (fixed demo)
            15.0 + 12*np.sin(2*np.pi*self.timestep/8760), # outdoor_T
            60.0,                                          # RH
            15.0,                                          # IT power kW
            0.5,                                           # HVAC kWh
            0.3,                                           # water L
            ci,
            np.sin(2*np.pi*tod/24),
            np.cos(2*np.pi*tod/24),
        ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Sinergym Wrapper (production environment)
# ─────────────────────────────────────────────────────────────────────────────

class CAMORLSinergymWrapper:
    """
    Wraps a Sinergym EnergyPlus environment with CA-MORL extensions.

    Parameters
    ----------
    env_id   : Sinergym environment ID, e.g.
               'Eplus-2ZoneDataCenterHVAC_wEconomizer_CW-cool-v1'
    ci_module: CarbonIntensityModule instance
    """

    OBS_DIM = 11
    ACT_DIM = 2

    def __init__(self, env_id: str, ci_module: CarbonIntensityModule):
        import gymnasium as gym
        self._env = gym.make(env_id)
        self.ci = ci_module
        self._timestep = 0

    def reset(self):
        obs, info = self._env.reset()
        self._timestep = 0
        return self._augment_obs(obs), info

    def step(self, action: np.ndarray):
        obs, _, terminated, truncated, info = self._env.step(action)
        ci_val = self.ci.get(self._timestep)

        HVAC_energy = float(info.get('HVAC_electricity_demand_rate', 0.5))
        water_use   = float(info.get('cooling_tower_water_consumption', 0.3))
        carbon      = HVAC_energy * ci_val

        self._timestep += 1
        aug_obs = self._augment_obs(obs)
        return aug_obs, (HVAC_energy, water_use, carbon), terminated, truncated, info

    def _augment_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        ci = self.ci.get(self._timestep)
        tod = self._timestep % 24
        sin_tod = np.sin(2 * np.pi * tod / 24)
        cos_tod = np.cos(2 * np.pi * tod / 24)
        # Trim/pad raw obs to 8 dims, then append CI + time encodings
        base = raw_obs[:8] if len(raw_obs) >= 8 else np.pad(raw_obs, (0, 8-len(raw_obs)))
        return np.concatenate([base, [ci, sin_tod, cos_tod]]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Reward Function
# ─────────────────────────────────────────────────────────────────────────────

class MORewardFunction:
    """
    Scalarises the (energy, water, carbon) objective vector.

    R_t = w_E * r_E + w_W * r_W + w_C * r_C
          - λ * max(0, T_max - 32)

    where r_X = -X_t / X_ref  (normalised, lower-is-better).

    Parameters
    ----------
    weights       : (w_E, w_W, w_C) summing to 1
    ref_energy    : reference energy normalisation (kWh/step)
    ref_water     : reference water normalisation  (L/step)
    ref_carbon    : reference carbon normalisation (kgCO₂/step)
    safety_lambda : penalty coefficient for temperature violations
    """

    def __init__(self,
                 weights: Tuple[float, float, float] = (0.34, 0.33, 0.33),
                 ref_energy: float  = 3.0,
                 ref_water:  float  = 15.0,
                 ref_carbon: float  = 2.5,
                 safety_lambda: float = 100.0):
        self.weights = np.array(weights, dtype=np.float32)
        self.ref_energy = ref_energy
        self.ref_water  = ref_water
        self.ref_carbon = ref_carbon
        self.safety_lambda = safety_lambda

    def __call__(self, energy: float, water: float, carbon: float,
                 zone_temps: np.ndarray) -> Tuple[float, Dict[str, float]]:
        r_E = -energy / self.ref_energy
        r_W = -water  / self.ref_water
        r_C = -carbon / self.ref_carbon

        scalar = float(np.dot(self.weights, [r_E, r_W, r_C]))

        # Safety penalty
        t_max = float(np.max(zone_temps))
        penalty = self.safety_lambda * max(0.0, t_max - 32.0)
        total = scalar - penalty

        return total, {'r_energy': r_E, 'r_water': r_W, 'r_carbon': r_C,
                       'safety_penalty': -penalty, 'total': total}


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def make_env(weights: Tuple[float, float, float],
             ci_csv: Optional[str] = None,
             seed: int = 42,
             sinergym_env_id: Optional[str] = None,
             ref_energy: float = 3.0,
             ref_water:  float = 15.0,
             ref_carbon: float = 2.5):
    """
    Build a (env, reward_fn) pair ready for CA-MORL training.

    Parameters
    ----------
    weights         : (w_E, w_W, w_C) for this run
    ci_csv          : path to real CI CSV, or None for synthetic
    seed            : random seed
    sinergym_env_id : if set and Sinergym is installed, use real env
    ref_*           : normalisation denominators from rule-based baseline
    """
    ci_module = CarbonIntensityModule(csv_path=ci_csv, seed=seed)

    if sinergym_env_id and _SINERGYM_AVAILABLE:
        env = CAMORLSinergymWrapper(sinergym_env_id, ci_module)
        print(f"[make_env] Using Sinergym env: {sinergym_env_id}")
    else:
        env = SyntheticDataCenterEnv(ci_module, seed=seed)
        print(f"[make_env] Using synthetic env (Sinergym not available or not requested)")

    reward_fn = MORewardFunction(
        weights=weights,
        ref_energy=ref_energy,
        ref_water=ref_water,
        ref_carbon=ref_carbon,
    )
    return env, reward_fn
