# CA-MORL: Carbon-Aware Multi-Objective Reinforcement Learning for Sustainable Data Center Cooling Optimization

<p align="center">
  <img src="figures/fig1_mean_std_bars.png" width="800" alt="CA-MORL Results"/>
</p>

<p align="center">
  <a href="https://spicscon.org/2026/"><img src="https://img.shields.io/badge/Conference-IEEE%20SPICSCON%202026-blue" alt="Conference"/></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-green" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License"/>
  <img src="https://img.shields.io/badge/Status-Under%20Review-red" alt="Status"/>
</p>

---

## 📌 Overview

This repository contains the official implementation of **CA-MORL**, submitted to **IEEE SPICSCON 2026**.

> **Paper title:** Carbon-Aware Multi-Objective Reinforcement Learning for Sustainable Data Center Cooling Optimization
>
> **Authors:** Afia Mahpara · Rebeka Sultana Dina 
>
> **Institution:** Department of Computer Science and Engineering, East West University, Dhaka, Bangladesh

Most existing RL approaches for data center cooling optimize **only for energy**, ignoring carbon emissions and water usage. CA-MORL simultaneously minimizes all three using a **multi-objective PPO** agent with dynamic grid carbon-intensity signals.

---

##  Repository Structure

```
CA-MORL/
│
├──  README.md                          ← You are here
├──  requirements.txt                   ← All dependencies
├──  LICENSE                            ← MIT License
│
├──  environment.py                     ← CA-MORL environment wrapper
│                                            (synthetic fallback + Sinergym adapter)
│                                            (CarbonIntensityModule with diurnal/seasonal CI)
│
├──  ppo_agent.py                       ← PPO implementation
│                                            (Actor-Critic MLP, GAE, RolloutBuffer)
│
├──  train_weight_sweep.py              ← Main training script
│                                            (9 configs × 3 seeds = 27 runs)
│
├──  analysis_corrected.ipynb           ← Full analysis notebook
│                                            (mean±std, Pareto, CI sensitivity, figures)
│
├── data/
│   ├── weight_sweep_results_corrected.csv  ← Per-seed results (27 rows)
│   └── carbon_intensity_example.csv        ← Example CI data format
│
├── figures/
│   ├── fig1_mean_std_bars.png            ← Per-objective performance (mean±std)
│   ├── fig2_per_seed_scatter.png         ← Seed reproducibility check
│   ├── fig3_ci_sensitivity.png           ← Post-hoc CI sensitivity analysis
│   ├── fig4_3d_pareto.png               ← 3D Pareto frontier
│   └── fig5_radar.png                   ← Policy radar chart
│
└── paper/
    └── CA_MORL_IEEE_Final.tex            ← IEEE 2-column LaTeX source
```

---

##  Key Results

| Policy | Energy (kWh) | Water (L) | Carbon (kgCO₂) | Violations |
|---|---|---|---|---|
| Energy Only | 2804 ± 13 | 8420 ± 45 | 2020 ± 11 | 0 |
| Water Only | 19442 ± 45 | 136227 ± 151 | 13129 ± 47 | 1949 |
| Carbon Only | 19456 ± 34 | 13957 ± 22 | 14842 ± 30 | 2015 |
| Equal (Balanced) | 19451 ± 32 | 133932 ± 162 | 13126 ± 34 | 2001 |
| Carbon Focused | 2807 ± 13 | 8431 ± 42 | 2022 ± 11 | 0 |
| **Carbon Heavy ** | **2809 ± 13** | **1683 ± 6** | **1979 ± 9** | **0** |

> All values are mean ± std across **3 random seeds**.  = Pareto-optimal best policy.

**Key findings:**
- Carbon Heavy policy achieves **~85.6% energy reduction** and **~98.8% water reduction** vs worst-performing configurations
- Any configuration with `w_E ≥ 0.10` achieves **zero ASHRAE temperature violations**
- Post-hoc CI sensitivity analysis confirms **7× carbon advantage** for energy-efficient policies at any grid intensity level

---

##  Installation

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/CA-MORL.git
cd CA-MORL
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — (Optional) Install Sinergym for real EnergyPlus simulation
```bash
pip install sinergym[extra]
```
> **Note:** If Sinergym is not installed, the code automatically falls back to the built-in synthetic physics-based simulator with an identical API. All reported results use this synthetic fallback.

---

##  Quick Start

### Run a smoke test (2–3 minutes)
```bash
python train_weight_sweep.py --timesteps 5000 --seeds 1
```
Confirm `logs/weight_sweep_results_corrected.csv` is created.

### Run the full weight sweep (48–72 hours on CPU)
```bash
python train_weight_sweep.py --timesteps 1000000 --seeds 3
```

### Run with real carbon intensity data (optional)
```bash
python train_weight_sweep.py --timesteps 1000000 --seeds 3 \
    --ci_csv data/carbon_intensity_example.csv
```

### Run analysis and generate figures
```bash
jupyter notebook analysis_corrected.ipynb
# Run all cells top to bottom
```

---

##  Framework Overview

```
CA-MORL System
├── CarbonIntensityModule
│   ├── Real CI: loads from ElectricityMap/WattTime CSV
│   └── Synthetic CI: diurnal + seasonal variation (CV ≈ 26%)
│
├── Environment
│   ├── SyntheticDataCenterEnv  ← used in this paper (fallback)
│   └── CAMORLSinergymWrapper   ← for EnergyPlus (future work)
│
├── PPOAgent
│   ├── ActorCritic (MLP [256, 256], ReLU)
│   ├── RolloutBuffer (GAE advantage estimation)
│   └── PPO clipped surrogate objective (ε = 0.2)
│
└── MORewardFunction
    └── R_t = w_E·r_E + w_W·r_W + w_C·r_C − λ·max(0, T_t − 32)
```

### Observation space (11-D)
| Variable | Unit |
|---|---|
| Zone 1 & 2 Temperature | °C |
| Supply Air Temperature | °C |
| Outdoor Dry-Bulb Temperature | °C |
| Outdoor Relative Humidity | % |
| IT Power Consumption | kW |
| HVAC Energy (per timestep) | kWh |
| Cooling Tower Water Use | L |
| Grid Carbon Intensity (CIₜ) | kgCO₂/kWh |
| Time-of-Day | sin/cos encoding |

### Action space (2-D continuous)
| Action | Range |
|---|---|
| Chilled Water Setpoint | [15, 22] °C |
| Supply Air Flow Rate | [0.1, 1.0] normalised |

---

##  Weight Configurations

| Policy | w_E | w_W | w_C |
|---|---|---|---|
| Energy Only | 1.00 | 0.00 | 0.00 |
| Water Only | 0.00 | 1.00 | 0.00 |
| Carbon Only | 0.00 | 0.00 | 1.00 |
| Energy+Water | 0.50 | 0.50 | 0.00 |
| Energy+Carbon | 0.50 | 0.00 | 0.50 |
| Water+Carbon | 0.00 | 0.50 | 0.50 |
| Equal (Balanced) | 0.34 | 0.33 | 0.33 |
| Carbon Focused | 0.20 | 0.20 | 0.60 |
| **Carbon Heavy** | 0.10 | 0.10 | **0.80** |

---

##  Results and Figures

<p align="center">
  <img src="figures/fig2_per_seed_scatter.png" width="700" alt="Per-seed scatter"/>
  <br><em>Fig 2: Per-seed reproducibility check — tight clusters confirm low variance, clear separation confirms no checkpoint bug</em>
</p>

<p align="center">
  <img src="figures/fig3_ci_sensitivity.png" width="700" alt="CI Sensitivity"/>
  <br><em>Fig 3: Post-hoc CI sensitivity — energy-efficient policies maintain 7× carbon advantage at any grid intensity</em>
</p>

<p align="center">
  <img src="figures/fig4_3d_pareto.png" width="500" alt="3D Pareto Front"/>
  <br><em>Fig 4: 3D Pareto frontier — stars indicate the two Pareto-optimal solutions</em>
</p>

<p align="center">
  <img src="figures/fig5_radar.png" width="450" alt="Radar Chart"/>
  <br><em>Fig 5: Policy radar chart — Carbon Heavy dominates across all three objectives</em>
</p>

---

##  Known Limitations

1. **Synthetic CI signal** — Training used a synthetic carbon-intensity time-series (CV ≈ 26%). Real ElectricityMap/WattTime data is needed to validate the full temporal adaptation benefit of carbon-aware control.
2. **Synthetic simulator** — Results use a synthetic physics-based fallback. Full Sinergym/EnergyPlus deployment is left for future work due to compute constraints.
3. **Bimodal energy distribution** — The weight-sweep resolution produces two clusters (~2,800 kWh and ~19,450 kWh). A finer weight grid would produce a richer Pareto frontier.
4. **Simulation only** — No real-world hardware deployment.

---

##  Future Work

- [ ] Integrate real ElectricityMap/WattTime CI data
- [ ] Full Sinergym/EnergyPlus deployment
- [ ] Finer-grained weight sweep (step 0.05 instead of 0.10)
- [ ] Multi-data-center scaling
- [ ] Uncertainty quantification in CI forecasting
- [ ] Real-time deployment with live grid signals

---


##  License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Department of Computer Science and Engineering<br>
  East West University, Dhaka, Bangladesh
</p>
