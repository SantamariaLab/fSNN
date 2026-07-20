# fSNN
Fractional order spiking neural networks applied to cart-pole problem.


# Fractional-Order Spiking Neural Networks for Robust Cartpole Control

Code accompanying the manuscript:

> **"Fractional-order spiking neural networks for robust control "** — *Neurocomputing* (under review)

This repository contains the full pipeline for training, evaluating, and analysing fractional-order Leaky Integrate-and-Fire (fLIF) Spiking Neural Networks (SNNs) as controllers for the Cart-Pole balancing task. The central claim is that replacing the integer-order membrane dynamics with fractional-order dynamics produces controllers that are more robust to external perturbations and that exhibit neuronal avalanche statistics consistent with criticality.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dependencies and Requirements](#dependencies-and-requirements)
3. [Building the Cython Extensions](#building-the-cython-extensions)
4. [Folder Structure](#folder-structure)
5. [How to Run the Code](#how-to-run-the-code)
6. [Key Parameters and Configuration](#key-parameters-and-configuration)

---

## Project Overview

### Biological motivation
Standard LIF neurons integrate current linearly over time (first-order ODE). Fractional-order neurons replace the integer derivative with a fractional one of order α ∈ (0, 1], introducing a power-law memory kernel. The closer α is to 0, the stronger the long-range temporal memory.

### Control task
The CartPole-v1 environment (Gymnasium) is solved using **only cart position and pole angle**. A two-layer SNN receives these two observations and outputs a binary push-left / push-right decision every 10 SNN time steps.

### Training
Network weights are optimised by a parallelised **Genetic Algorithm (GA)** using MPI. Each MPI rank evaluates one individual in the population. The GA evolves neuron bias and synaptic weights. The network architecture and fractional order are fixed per run.

### Analyses
- **Figures 1–2**: Fractional order capacitor V–Q curves and fLIF firing-rate hysteresis as a function of α.
- **Figure 3**: Network diagrams and evolutionary fitness curves for fSNN vs SNN.
- **Figure 4**: 1000-trial performance distributions across 40 evolutionary runs for different values of α.
- **Figure 5**: Robustness to external perturbations.
- **Figure 6**: Hybrid networks — a non fractional SNN initialises the episode and an fSNN takes over at a fixed switch point.
- **Figure 7**: Neuronal avalanche size distributions (CCDF), power-law fits and firing rates for different fractional orders.
- **Figure 8**: Computational cost analysis — MFLOPS/second and spikes/second across fractional orders.

---

## Dependencies and Requirements

| Package | Tested version | Purpose |
|---|---|---|
| Python | ≥ 3.10 | Runtime |
| NumPy | ≥ 1.24 | Numerical core |
| SciPy | ≥ 1.10 | Signal processing, curve fitting, stats |
| Matplotlib | ≥ 3.7 | Plotting |
| Seaborn | ≥ 0.12 | Statistical plots |
| Pandas | ≥ 2.0 | DataFrames for Seaborn |
| Gymnasium | ≥ 0.26 | CartPole-v1 environment |
| mpi4py | ≥ 3.1 | MPI parallelism for GA and evaluation |
| Cython | ≥ 3.0 | Fast fLIF/LIF update kernels |
| statsmodels | ≥ 0.14 | OLS regression for avalanche power-law fits |

Install with:

```bash
pip install numpy scipy matplotlib seaborn pandas gymnasium mpi4py cython statsmodels
```

An MPI implementation (OpenMPI or MPICH) must be installed system-wide before `mpi4py`.

---

## Building the Cython Extensions

Two Cython extensions provide fast single-step updates for the fLIF and LIF networks. They **must be compiled before running any training or evaluation**.

```bash
# Compile both extensions (run from Figures_3_4_5_6_7_8/utils/)
cd Figures_3_4_5_6_7_8/utils
python setup_flif.py build_ext --inplace
python setup_lif.py build_ext --inplace
```
---

## Folder Structure

```
Fractional_order_for_robust_control/
│
├── Figures_1_2/                        # Figures 1–2: fractional capacitor & fLIF hysteresis
│   ├── figure_1_fractional_capacitor.py
│   ├── figure_2_FLIF_memelement.py
│   ├── hysteresis/
│   │   ├── flifnetwork.py              # Single-neuron fLIF class (standalone)
│   │   └── run_I_slope.py              # MPI script — computes firing-rate hysteresis
│   └── figures/  (fig_1.png, fig_2.png, *.pdf)
│
├── Figures_3_4_5_6_7_8/               # Figures 3–8: main results
│   │
│   ├── utils/                          # Shared network classes & build scripts
│   │   ├── flifnetwork.py              # FractionalLIFNetwork (2-layer fSNN)
│   │   ├── lifnetwork.py               # LIFNetwork (2-layer standard SNN)
│   │   ├── flifnetwork_step.pyx        # Cython inner loop for fLIF
│   │   ├── lifnetwork_step.pyx         # Cython inner loop for LIF
│   │   ├── setup_flif.py               # Build script for flifnetwork_step
│   │   ├── setup_lif.py                # Build script for lifnetwork_step
│   │   ├── fSNN.py                     # Network diagram drawing utilities
│   │   └── cartpole_depiction.py       # CartPole schematic drawing
│   │
│   ├── fractional/                     # fSNN training pipeline (α < 1)
│   │   ├── run_evolution.py            # MPI-parallel GA for 6 values of α
│   │   ├── selection.py                # Select best individual per evolution run
│   │   ├── evaluate_selected.py        # 1000-trial post-evolution evaluation
│   │   ├── evaluate_perturbation.py    # Evaluation under external force noise
│   │   ├── *.job                       # SLURM batch scripts (HPC cluster)
│   │   └── selected/                   # Pre-computed pkl results
│   │
│   ├── non_fractional/                 # Standard SNN training pipeline (α = 1)
│   │   ├── run_evolution.py            # MPI-parallel GA (600 epochs)
│   │   ├── selection.py                # Select best individual
│   │   ├── evaluate_selected.py        # 1000-trial evaluation
│   │   ├── evaluate_perturbation.py    # Noisy evaluation
│   │   ├── *.job                       # SLURM batch scripts
│   │   └── selected/                   # Pre-computed pkl results
│   │
│   ├── evaluate_hybrids/               # Hybrid SNN→fSNN controller evaluation
│   │   ├── run_evaluation_hybrids.py   # MPI evaluation of hybrid networks
│   │   ├── get_indv_to_evaluate.py     # Collect selected individuals
│   │   ├── to_hybrid_evaluation.pkl    # Individual lookup table
│   │   └── data/                       # Per-individual evaluation results
│   │
│   ├── dynamic_avalanches/             # Neuronal dynamics analysis
│   │   ├── run_dynamic_selected.py     # Record spike trains for selected networks
│   │   ├── dynamic_analysis_avalanches.py  # Detect and quantify avalanches
│   │   ├── get_rates.py                # Compute firing rates and CVs
│   │   ├── flifnetwork.py              # Local fLIF copy (standalone for avalanche run)
│   │   ├── lifnetwork.py               # Local LIF copy
│   │   ├── *.pyx / setup_*.py          # Cython sources and build scripts
│   │   └── *.pkl                       # Pre-computed avalanche data
│   │
│   ├── data_figures/                   # Aggregated data for figure scripts
│   │   ├── evolutions.pkl              # Fitness-over-epoch history (all α, all runs)
│   │   └── evaluation_1000.pkl         # Mean 1000-trial performance per individual
│   │
│   ├── figure3_networks_and_evolutions.py
│   ├── figure4_performance_bests.py
│   ├── figure5_noisy_environment.py
│   ├── figure6_hybrid_networks.py
│   ├── figure7_plot_avalanches.py
│   ├── figure8_energy.py               # MFLOPS/s and spikes/s across fractional orders
│   └── figures/                        # Generated paper figures (PDF + PNG)
│
├── Figures/                            # Final paper-ready PDF copies of all figures
```

---

## How to Run the Code

All scripts in `Figures_3_4_5_6_7_8/fractional/`, `non_fractional/`, `evaluate_hybrids/`, and `dynamic_avalanches/` are designed to run on an HPC cluster via SLURM and MPI. The SLURM `.job` files document the exact job parameters used in the paper.

### Step 0 — Compile Cython extensions

```bash
cd Figures_3_4_5_6_7_8/utils
python setup_flif.py build_ext --inplace
python setup_lif.py build_ext --inplace
```

### Step 1 — Run evolutionary training

**Fractional SNN (6 values of α, 40 independent runs each):**

```bash
cd Figures_3_4_5_6_7_8/fractional
mpirun -n 150 python run_evolution.py
```

**Standard SNN (α = 1, 40 runs):**

```bash
cd Figures_3_4_5_6_7_8/non_fractional
mpirun -n 150 python run_evolution.py
```

Each run saves one `.pkl` file per evolution: `data/evolution_{alpha}_{run}.pkl`.

### Step 2 — Select best individuals

```bash
# From each folder:
mpirun -n 4 python selection.py
```

Outputs: `selected/selected_{alpha}.pkl` (fractional) or `selected/selected.pkl` (non-fractional).

### Step 3 — Post-evolution evaluation (1000 trials, noiseless)

```bash
mpirun -n 100 python evaluate_selected.py
```

Outputs: `selected/evaluation_1000.pkl`

### Step 4 — Noisy environment evaluation

```bash
mpirun -n 100 python evaluate_perturbation.py
```

Sweeps noise amplitudes σ ∈ [0.1, 15] N and saves per-amplitude performance arrays.

### Step 5 — Hybrid network evaluation

```bash
cd Figures_3_4_5_6_7_8/evaluate_hybrids
mpirun -n 100 python run_evaluation_hybrids.py
```

### Step 6 — Avalanche analysis

```bash
cd Figures_3_4_5_6_7_8/dynamic_avalanches
python run_dynamic_selected.py      # record spike trains
python dynamic_analysis_avalanches.py
python get_rates.py
```

### Step 7 — Generate figures

All figure scripts can be run locally (no MPI needed) once the `.pkl` result files exist. Pre-computed results are already included.

```bash
cd Figures_3_4_5_6_7_8

python figure3_networks_and_evolutions.py
python figure4_performance_bests.py
python figure5_noisy_environment.py
python figure6_hybrid_networks.py
python figure7_plot_avalanches.py
python figure8_energy.py
```

For Figures 1–2:

```bash
cd Figures_1_2
python figure_1_fractional_capacitor.py

# figure 2 requires pre-computed hysteresis data
cd hysteresis
mpirun -n 4 python run_I_slope.py   # generates data.pkl
cd ..
python figure_2_FLIF_memelement.py
```

---

## Key Parameters and Configuration

### Network architecture

| Parameter | fSNN | SNN |
|---|---|---|
| Inputs | 2 (cart position *x*, pole angle *θ*) | 2 |
| Hidden neurons | 6 fLIF | 12 LIF |
| Output neurons | 1 | 1 |
| Observation normalisation | [*x*/2.4, *θ*/0.05] | same |

### Neuron parameters

| Parameter | Symbol | Value |
|---|---|---|
| Membrane time constant | τ | 3.5 ms (fSNN) / 10 ms (SNN) |
| Firing threshold | V_th | 10 mV |
| Reset voltage | V_reset | 0 mV |
| Grünwald–Letnikov memory length | L | 120 steps |
| Simulation time step | dt | 1 ms |
| SNN steps per CartPole step | — | 10 |

### Fractional orders studied

```
α ∈ {0.20, 0.35, 0.50, 0.65, 0.80, 0.95}   (fSNN)
α = 1.0                                        (standard SNN baseline)
```

### Genetic Algorithm hyperparameters

| Parameter | fSNN | SNN |
|---|---|---|
| Population size | 150 | 150 |
| Generations | 150 | 600 |
| Independent runs | 40 (× 6 α values) | 40 |
| Elite fraction | 10% | 10% |
| Random injection | 30% → 0% (linear) | same |
| Crossover rate | 12% | 12% |
| Mutation rate | 15% per parameter | 15% |
| Mutation strength | ±2 (bias), ±2 (weights) | ±1 (bias), ±2 (weights) |
| Weight range | [−22, 22] | [−20, 20] |
| Bias range | [9, 16] (fSNN) | [10, 12] (SNN) |
| Fitness function | mean episode length over 10 random seeds | same |
| Maximum fitness (stopping) | 15 000 steps (300 s) | same |
| Early stopping | 10 consecutive max-fitness generations | same |
