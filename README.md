# Identifying Potential Escapers with Graph Neural Networks: Recovering Cluster Dynamics from Partial Kinematics

**Author:** Adrian Cozma ([@adricozma](https://github.com/adricozma))  
**Affiliation:** Laboratory of Astrophysics (LASTRO), École Polytechnique Fédérale de Lausanne (EPFL)  
**Supervisors:** Prof. Yves Revaz, Dr. Srikanth Nagesh  

---

## Overview

As globular clusters orbit the Milky Way, gravitational tidal forces strip stars from their outer envelopes, forming elongated **stellar streams**. Determining whether an observed star remains gravitationally bound to the progenitor core, is a **potential escaper** trapped inside the Jacobi surface with excess energy, or has already escaped into the unbound tidal tails formally requires full 6D phase-space coordinates $(x, y, z, v_x, v_y, v_z)$ and explicit knowledge of the host galactic potential.

In observational surveys such as *Gaia*, however, radial velocities $(v_{\mathrm{los}})$ are missing for the vast majority of faint stars $(G \gt 16)$, leaving only 5D astrometric measurements: sky positions $(\alpha, \delta)$, parallax $(\varpi)$, and proper motions $(\mu_{\alpha^*}, \mu_\delta)$.

This repository provides a complete numerical and deep-learning pipeline that:
1. Reconstructs realistic progenitor initial conditions via backward orbit integration of observed Milky Way stellar streams from the *Gaia* DR3 atlas (Ibata et al. 2024) inside a realistic three-component Milky Way potential (`MWPotential2014`, Bovy 2015).
2. Evolves 81 self-consistent *N*-body Plummer globular clusters forward in time for $2.0\text{ Gyr}$ using the cosmological/astrophysical code `SWIFT` (Schaller et al. 2024) on the EPFL HPC cluster **Jed** (and 75 clusters in Study 1).
3. Computes exact physical boundness and potential escaper labels from 3D Jacobi energies and tidal radii following Küpper et al. (2010a,b), using a 3D *k*-nearest-neighbor local density estimator (Casertano & Hut 1985) and iterative shrinking spheres to track the surviving core accurately even in heavily stripped streams.
4. Projects the simulations into synthetic 5D *Gaia*-like astrometric observables with data augmentation over random observer orientations and physical-scale standardization.
5. Trains **Residual Graph Convolutional Networks (GCN)** and **Multi-Head Residual Graph Attention Networks (GAT)** with local phase-space relational encoding (`3d + 3` invariants) on *k*-NN graphs $(k = 20)$, using **Two-Stage Curriculum Learning** (noiseless pre-training $\to$ *Gaia* DR3 magnitude-dependent noisy fine-tuning), **subspace-aligned $5\mathrm{D} \to 4\mathrm{D}$ weight transfer** for systematic observable ablations, and multi-node distributed training with micro-batch gradient accumulation and automatic disk activation offloading.

> **Live Training Progress:**  
> Real-time training metrics and learning curve figures (`latest_live_training.png` & `live_plots_GNN_*.png`) are tracked in [**`LIVE_STATUS.md`**](LIVE_STATUS.md), and pre-trained model checkpoints (`.pth`) are indexed in [**`GNN-jed/trained_models/study2/weights/MODEL_WEIGHTS_REGISTRY.md`**](GNN-jed/trained_models/study2/weights/MODEL_WEIGHTS_REGISTRY.md).

---

## Repository Structure

The repository is organized to separate core physical/feature-engineering modules (`src/`), executable production pipelines (`scripts/`), SLURM cluster templates (`slurm/`), initial condition generation (`make_ICs/`), visualization utilities (`visualization/`), and interactive analysis notebooks (`notebooks/`):

```text
.
├── assets/
│   └── figures/                         # Figures from simulations, orbital dynamics, and GNN evaluation
├── configs/
│   ├── params.yml                       # SWIFT runtime configuration (MilkyWayPotential2014, leapfrog/gravity)
│   └── dependency_graph_0.csv           # SWIFT task-level dependency graph
├── make_ICs/
│   ├── Ibata_catalog.txt                # Gaia DR3 stellar stream catalog (Ibata et al. 2024)
│   ├── stars_6D_ibata2024.csv           # Filtered 6D stream stars used to define progenitor orbits
│   ├── select_unbiased_orbits.py        # Unbiased progenitor orbit selector (R_p >= 0.0 kpc, 81 streams)
│   ├── generate_unbiased_ics.py         # Batch virialized Plummer sphere HDF5 generator for SWIFT
│   ├── IC_plummer_swift.py              # Isotropic/anisotropic Plummer sphere HDF5 generator
│   ├── OrbitIntegration_Ibata.ipynb     # Backward orbit integration (-2 Gyr) in galpy MWPotential2014
│   ├── WorkflowIbata.ipynb              # Catalog processing and progenitor seed extraction workflow
│   ├── SWIFT_seeds.npy                  # Study 1 6D phase-space seeds at t = -2 Gyr (75 streams)
│   ├── SWIFT_seeds_unbiased.npy         # Study 2 unbiased 6D phase-space seeds at t = -2 Gyr (81 streams)
│   ├── unbiased_orbits_summary.csv      # Orbital parameters (R_p, R_a, eccentricity) across streams
│   ├── orbits/                          # Integrated 2 Gyr orbital trajectories (Study 1)
│   └── orbits_unbiased/                 # Integrated 2 Gyr unbiased orbital trajectories (Study 2)
├── src/
│   ├── bound_labels.py                  # 3D k-NN density + shrinking-sphere centering & exact Jacobi/Roche solver
│   ├── observables_matrix.py            # Haar-uniform SO(3) rotations, ICRS projection, Gaia DR3 noise & scaling
│   ├── gnn_models.py                    # Local phase-space relational encoder, Residual GCN & Multi-Head GAT
│   ├── oom_Pin_and_offload.py           # Distributed DDP initializer & cgroup-aware disk activation offloader
│   └── Stellar_objects/                 # Utilities for King/Plummer cluster profiles and LOS dispersion
├── scripts/
│   ├── extract_observables.py           # Parallel extraction of bound labels + Clean & Gaia DR3 Noisy observables
│   ├── build_graph_dataset.py           # Parallel cKDTree k-NN graph builder (5D full & 4D ablation subsets)
│   ├── distributed_trainer.py           # Unified DDP, micro-batch gradient accumulation & checkpoint engine
│   ├── train_gcn.py                     # 3-layer Residual GCN classifier training CLI (clean_5obs baseline)
│   ├── train_gat.py                     # 3-layer Multi-Head Residual GAT training & curriculum fine-tuning CLI
│   ├── generate_study2_figures.py       # Automated evaluation & publication figure generator (.png & .html)
│   ├── live_plot_utils.py               # Real-time learning curve plotter from training logs
│   └── live_monitor.py                  # Automated training telemetry and status updater
├── slurm/
│   ├── step1_orbits_and_ics.sh          # Step 1: Orbit integration & Plummer HDF5 initial conditions
│   ├── step2_run_swift_array.sh         # Step 2: SWIFT N-body simulation job array
│   ├── step3_extract_observables.sh     # Step 3: Bound label & Clean/Noisy observable extraction
│   ├── step4_build_graphs.sh            # Step 4: 5D cKDTree k-NN PyTorch Geometric graph construction
│   ├── step5_train_curriculum_and_plot.sh # Step 5: 5-Observable Clean -> Gaia DR3 Noisy curriculum training
│   ├── step6_ablation_4obs.sh           # Step 6: 4-Observable ablation study (5D -> 4D weight transfer)
│   ├── step7_retrain_gcn_stochastic.sh  # Step 7: Stochastic SO(3) GCN Clean baseline retraining
│   └── launch_full_study2_pipeline.sh   # Master SLURM pipeline launcher chaining Steps 5, 6, and 7
├── visualization/
│   ├── plot_swift_output.py             # HDF5 snapshot loader (DMParticles coordinates, velocities, masses, time)
│   ├── visu_stream_outputs.py           # 3D Galactocentric stream rendering and projection animations
│   ├── plot_metric_live.py              # Training vs. validation loss curve plotter from training logs
│   ├── final_sims_vizu.ipynb            # All-sky orthographic (Cartopy) and depth-sorted Galactocentric plots
│   └── fraction_bound_stars.ipynb       # Bound fraction evolution vs. orbital eccentricity and perigalacticon
├── notebooks/
│   ├── labels_data_matrices.ipynb       # Interactive generation and verification of labels and observable matrices
│   ├── graph_data.ipynb                 # PyTorch Geometric k-NN graph construction from 5D observables
│   └── test-model-GCN.ipynb             # Model inference, perigalacticon error analysis, and RMT SVD diagnostics
└── GNN-jed/
    └── trained_models/
        ├── GCN-01/                      # Study 1 baseline GCN weights (best_gnn_model_01.pth) & logs
        ├── GCN-02/                      # Study 1 multi-head GAT weights (best_gnn_model_02.pth) & logs
        └── study2/                      # Study 2 curriculum & ablation logs, figures, and checkpoints
            └── weights/                 # Central archive of best_*.pth checkpoints & MODEL_WEIGHTS_REGISTRY.md
```

*Note:* Bulk `SWIFT` simulation snapshots, intermediate `.npy` observable/label arrays, and serialized `.pt` graph datasets are excluded from version control via `.gitignore` and can be generated in any user-specified local or scratch directory (`$DATA_DIR`).

---

## Physical & Methodological Pipeline

### 1. Progenitor Orbit Reconstruction & *N*-Body Simulations (`make_ICs/`, `configs/`)

Starting from the *Gaia* DR3 stellar stream catalog of Ibata et al. (2024), stars with complete 6D phase-space measurements are grouped by stream identifier (`Track`). Representative present-day stream anchors are integrated backward in time for $\Delta t = -2.0\text{ Gyr}$ using `galpy` (Bovy 2015) inside `MWPotential2014` — a three-component potential comprising a Miyamoto-Nagai disk, a power-law bulge with exponential cut-off, and an NFW dark matter halo, calibrated to $R_0 = 8.0\text{ kpc}$ and $V_c = 220\text{ km s}^{-1}$.

| Present-Day Stream Anchors $(t = 0\text{ Gyr})$ | Backward-Integrated Progenitor Seeds $(t = -2.0\text{ Gyr})$ |
| :---: | :---: |
| ![Present-day positions](assets/figures/positions_today.png) | ![Backward-integrated seeds](assets/figures/seeds_positions.png) |

At each $t = -2.0\text{ Gyr}$ seed position, an isotropic $(q = 0)$ Plummer sphere (Plummer 1911) is initialized via `make_ICs/IC_plummer_swift.py` (Study 1 baseline: $M = 10^4\,M_\odot$, $a = 4\text{ pc}$, $N \in [2500, 3500]$; Study 2 unbiased benchmark via `make_ICs/generate_unbiased_ics.py`: 75 symlinked Study 1 streams plus 6 massive outer-halo progenitors spanning $N \in [20\,000, 50\,000]$ and $a \in [1, 50]\text{ pc}$):

$$
\rho(r) = \frac{3M}{4\pi a^3}\left(1 + \frac{r^2}{a^2}\right)^{-5/2}, \qquad \Phi(r) = -\frac{GM}{\sqrt{r^2 + a^2}}.
$$

Each cluster is evolved forward for $2.0\text{ Gyr}$ in `SWIFT` (Schaller et al. 2024) using `configs/params.yml` with 101 HDF5 snapshots per simulation $(\Delta t_{\mathrm{snap}} = 20\text{ Myr})$ and a comoving Plummer-equivalent softening length $\epsilon = 1\text{ pc}$ (Study 1) or $\epsilon = 5\text{ pc}$ (Study 2 `configs/params.yml`).

| Tidal Disruption Sequence (`IC_53`, Low Perigalacticon) | Multi-Epoch Stream Formation (`IC_70`) |
| :---: | :---: |
| ![Tidal disruption sequence IC_53](assets/figures/sequence_IC_53.png) | ![Stream series IC_70](assets/figures/series_IC_70.png) |

---

### 2. Dynamical Boundness, Potential Escapers & Orbital Evolution (`src/bound_labels.py`)

Following Binney & Tremaine (2008) and Küpper et al. (2010a,b), a star in a cluster orbiting at galactocentric distance $R_{\mathrm{gal}}$ with instantaneous angular velocity $\mathbf{\Omega} = (\mathbf{R}_{\mathrm{gal}} \times \mathbf{V}_{\mathrm{gal}}) / R_{\mathrm{gal}}^2$ experiences an effective tidal boundary given by the King (1962) / Jacobi radius:

$$
R_J = \left(\frac{G M(\lt R_J)}{\Omega^2 - \frac{\partial^2 \Phi_{\mathrm{gal}}}{\partial R_{\mathrm{gal}}^2}}\right)^{1/3}, \qquad E_{\mathrm{crit}} = -\frac{3}{2}\frac{G M(\lt R_J)}{R_J}.
$$

Each star's specific Jacobi energy in the co-rotating frame relative to the surviving cluster core is:

$$
E_J = \frac{1}{2}\lVert\mathbf{v}_{\mathrm{rel}}\rVert^2 + \Phi_{\mathrm{cluster}}(r_{3D}) - \frac{1}{2}\left[\Omega^2 r_{3D}^2 - (\mathbf{\Omega}\cdot\mathbf{r}_{\mathrm{rel}})^2\right].
$$

To prevent extended, asymmetric tidal tails from biasing the cluster center-of-mass toward empty space between the tails, `src/bound_labels.py` locates the surviving core via a 3D *k*-nearest-neighbor local density estimator $\rho_i \propto d_{k,i}^{-3}$ (Casertano & Hut 1985) followed by an iterative shrinking-sphere refinement from $0.5\text{ kpc}$ down to $0.05\text{ kpc}$. Stars are classified into:
- **Bound Core Stars (`label = 1`):** $E_J \lt E_{\mathrm{crit}}$ and $r_{3D} \lt R_J$ (solved iteratively to convergence).
- **Potential Escapers:** $E_J \ge E_{\mathrm{crit}}$ and $r_{3D} \lt R_J$ (stars energetically unbound that remain spatially trapped inside the Jacobi surface near the $L_1 / L_2$ Lagrange points).
- **Unbound Stream Stars (`label = 0`):** Stars outside the Jacobi radius or exceeding the critical Jacobi energy.

![Bound Star Fraction vs Galactocentric Radius](assets/figures/R_vs_t_6panels_rp1.png)
*Evolution of the Galactocentric radius $R(t)$ over $2.0\text{ Gyr}$ across simulated clusters (75 in Study 1, expanded to 81 unbiased streams in Study 2), grouped into six panels by their first perigalacticon distance $R_{p1}$ and color-coded by the instantaneous fraction of bound stars.*

---

### 3. Synthetic 5D *Gaia* Observables & *k*-NN Graph Construction (`src/observables_matrix.py`)

For each snapshot, the 3D Galactocentric particle coordinates are rotated by $n_{\mathrm{rot}} = 5$ Haar-uniform 3D Euler matrices $\mathbf{R} \in \mathrm{SO}(3)$ to augment viewing angles, transformed into the heliocentric International Celestial Reference System (ICRS) via `astropy.coordinates.SkyCoord`, optionally perturbed with Lindegren et al. (2021) *Gaia* DR3 astrometric uncertainties at reference absolute magnitude $M_G = 4.5$ (`inject_gaia_dr3_noise`), and restricted to the 5D observable vector:

$$
\mathbf{x}_i = \left(\Delta\alpha_i\cos\delta_c,\; \Delta\delta_i,\; D_i,\; \mu_{\alpha^*,i},\; \mu_{\delta,i}\right) \in \mathbb{R}^5, \qquad D_i = \frac{1}{\operatorname{clip}(\varpi_{\mathrm{obs},i},\, 0.005,\, 10.0)}\text{ kpc}.
$$

In `src/observables_matrix.py`, Right Ascension is centered via the circular median on $\mathbb{S}^1$ and projected onto the local tangent plane. `standardise_matrix` supports both `mode="physical"` (fixed galactic scale factors) and `mode="zscore"` ($6\sigma_{\mathrm{MAD}}$ winsorization followed by per-snapshot Z-score standardization); the live Study 2 benchmark extraction (`slurm/step3_extract_observables.sh`) ran with `--norm_mode zscore --n_rotations 5 --abs_mag_g 4.5`. Directed *k*-nearest-neighbor graphs $(k = 20)$ are built in standardized 5D (or 4D ablation) feature space using `scipy.spatial.cKDTree` (`scripts/build_graph_dataset.py`).

---

### 4. Graph Neural Network Architectures, Curriculum Learning & 5D $\to$ 4D Ablation Transfer (`src/`, `scripts/`, `slurm/`)

To overcome the scale blindness of constant-degree $k$-NN graphs ($k = 20$), [`src/gnn_models.py`](src/gnn_models.py) augments each star's standardized astrometric vector $\mathbf{x}_i \in \mathbb{R}^d$ ($d \in \{4, 5\}$) on the fly with local $k$-NN relational invariants:

$$
\mathbf{h}_i^{(0)} = \left[\mathbf{x}_i,\; \mathbf{x}_i - \langle\mathbf{x}\rangle_{\mathcal{N}(i)},\; \boldsymbol{\sigma}_{\mathcal{N}(i)},\; \bar{d}_i,\; \ln\hat{\rho}_{\mathrm{phase},i},\; \lVert\mathbf{x}_i\rVert_2\right] \in \mathbb{R}^{3d + 3}.
$$

We evaluate two residual architectures across **81 unbiased *Gaia* DR3 progenitor streams** (75 Study 1 streams with $N \in [2500, 3500]$ plus 6 outer-halo progenitors spanning $N \in [20\,000, 50\,000]$, partitioned deterministically at the simulation level with `seed=42` into 65 Train / 8 Validation / 8 Test streams):
1. **3-Layer Residual GCN (`scripts/train_gcn.py`):** `(3d+3) -> 64 -> 128 -> 1` with `GCNConv`, linear residual skip projections, `LayerNorm`, `GELU` activations, and an ego-relational readout head `MLP([h2, h0])` (`--bs 256`, 25 epochs on the 5D noiseless baseline).
2. **3-Layer Multi-Head Residual GAT (`scripts/train_gat.py`):** `GATConv(3d+3, 32, heads=4)` $\to$ `GATConv(128, 64, heads=4)` $\to$ `GATConv(256, 1, heads=1)` with `LayerNorm`, `ELU` activations, dropout ($p = 0.4$ on both multi-head `GATConv` attention coefficients and hidden layer / ego-head activations), L2 regularization (`weight_decay = 1e-3` in `AdamW`), adaptive learning rate scheduling (`OneCycleLR` / cosine annealing on resume), validation BCE loss early stopping (`patience = 10` for Stage 1 Clean; `patience = 8` for Stage 2 Noisy and 4D ablations), and an ego-relational readout head `MLP([h2, h0])` (`--bs 64`). Training continuously tracks targeted surviving-core metrics **`CoreFNR` / `CoreRec`** (False Negative Rate and Recall on bound core stars, alongside the diagnostic figure `surviving_core_fnr_by_perigalacticon_GAT_{tag}.{png,html}`).

#### Two-Stage Curriculum, Stochastic $\mathrm{SO}(3)$ Training & $5\mathrm{D} \to 4\mathrm{D}$ Subspace Weight Transfer
- **Step 5 — 5-Observable Curriculum (`slurm/step5_train_curriculum_and_plot.sh`):** Both GCN (25 epochs, architectural baseline) and GAT (50 epochs, `--stochastic_rotation` sampling 1 random rotation per snapshot per epoch across $6\,565$ training graphs/epoch) are trained on noiseless 5D astrometric graphs (`clean_5obs`, $\mathrm{lr} = 10^{-3}$). The converged 5D Clean GAT checkpoint (`best_gat_clean.pth`) is then fine-tuned for up to 30 epochs (`patience = 8`) on *Gaia* DR3 noisy graphs (`noisy_5obs`, $\mathrm{lr} = 5\times 10^{-4}$, Lindegren et al. 2021).
- **Step 6 — 4-Observable Ablation Study (`slurm/step6_ablation_4obs.sh`):** To isolate the physical contribution of each astrometric coordinate (`no_RA`, `no_Dec`, `no_Parallax`, `no_pmRA`, `no_pmDec`), each 4D GAT model ($3d+3 = 15$ input channels) is initialized directly from the converged 5D Clean GAT weights ($18$ input channels) via `load_pretrained_with_channel_adaptation` (`src/gnn_models.py`). All internal message-passing and attention layers are copied identically ($100\%$ preservation), while first-layer projections (`conv1.lin.weight`, `skip1.weight`, and the `h0` slice of `ego_head.0.weight`) are sliced along the retained coordinate indices $\pi(\mathcal{I}) = \mathcal{I} \cup (5+\mathcal{I}) \cup (10+\mathcal{I}) \cup \{15,16,17\}$ and scaled by $\sqrt{18/15}$ prior to 4D Clean pre-training (up to 20 epochs, `patience = 8`) and 4D *Gaia* DR3 noisy fine-tuning (up to 20 epochs, `patience = 8`).
- **Step 7 — Stochastic GCN Retraining (`slurm/step7_retrain_gcn_stochastic.sh`):** Archives the completed static-5-rotation Clean GCN baseline (`Test Acc = 92.17%`, `BalAcc = 91.59%`, `F1 = 0.8886`) into `GNN-jed/trained_models/study2/legacy/clean_5obs_static_5rot/` and retrains Clean GCN for 25 epochs using `--stochastic_rotation --bf16 --bs 256` for direct regime parity with Stochastic GAT (noting that `--bs 256` performs $\sim 650$ optimizer updates over 25 epochs vs. GAT's $\sim 5\,150$ updates over 50 epochs at `--bs 64`).

#### Multi-Node Distributed Execution & Memory Optimization (`scripts/distributed_trainer.py`, `src/oom_Pin_and_offload.py`)
- **Multi-Node DDP & Micro-Batch Gradient Accumulation:** Supports single-machine execution (CPU or GPU) as well as multi-node SLURM `DistributedDataParallel` (`srun` / `torchrun`). Exact node-weighted gradient accumulation over micro-batches (`--micro_bs`) preserves the global effective batch size (`256` for GCN, `64` for GAT) while allowing SLURM wrappers (`run_ddp_with_oom_fallback`) to halve `--micro_bs` automatically and resume from the latest epoch checkpoint if memory limits are approached.
- **Disk Activation Offloading (`DiskActivationOffloader`):** Cgroup-aware saved-tensor hooks (`torch.autograd.graph.saved_tensors_hooks`) monitor node memory against `/sys/fs/cgroup/memory.current` vs. `memory.max` and stream large ($> 8\text{ MB}$) activation tensors to `/scratch` when utilization exceeds $70\%$ (triggered dynamically by node memory pressure; `force_enable` is currently `False` in `distributed_trainer.py`).

| Training & Validation Loss Convergence (GCN vs. GAT) | Weight Matrix Singular Value Spectrum (RMT) |
| :---: | :---: |
| ![Training and validation loss curves](assets/figures/live_plots_GNN.png) | ![Spectral analysis of GNN weights](assets/figures/spectral_analysis_GNN.png) |

In Study 1 ($75$ clusters, $t = 2.0\text{ Gyr}$, snapshot 100), the multi-head GAT achieved **96.26% mean node accuracy**, separating surviving bound cores from stripped tidal tails while concentrating remaining errors on marginal systems near $R_{p1} \in [8, 18]\text{ kpc}$. In the expanded 81-stream Study 2 pipeline, the completed static Clean GCN baseline achieved **92.17% Test Accuracy** (`F1 = 0.8886`), while Stochastic Clean GAT, Gaia DR3 Noisy GAT, and the 4-observable ablations are currently running on EPFL Jed (see [**`LIVE_STATUS.md`**](LIVE_STATUS.md)).

| Dissolved Stream (`IC_1`, $R_{p1} = 1.9\text{ kpc}$, 100% Accuracy) | Surviving Bound Core (`IC_81`, $R_{p1} = 16.5\text{ kpc}$, 98.11% Accuracy) |
| :---: | :---: |
| ![GAT predictions on IC_1 at snapshot 100](assets/figures/prediction_GAT_ID1_output100.png) | ![GAT predictions on IC_81 at snapshot 100](assets/figures/prediction_GAT_ID81_output100.png) |

![GAT Error Percentage by First Perigalacticon](assets/figures/error_percentage_by_perigalacticon.png)
*Classification error percentage at $t = 2.0\text{ Gyr}$ (snapshot 100) across simulated streams ordered by first perigalacticon distance $R_{p1}$. Streams with small $R_{p1}$ are completely disrupted early on, whereas clusters with intermediate $R_{p1}$ retain marginally bound cores rich in potential escapers.*

---

## Installation & Reproducibility Guide

Follow the steps below to reproduce the entire pipeline — from progenitor orbit integration and *N*-body simulations to graph construction, curriculum training, 4-observable ablations, and figure generation.

### 1. Environment Setup

Clone the repository and create a Python 3.10+ virtual environment with the required astrophysics and PyTorch Geometric packages:

```bash
git clone https://github.com/adricozma/GnnClusterClassifier.git
cd GnnClusterClassifier

python3 -m venv env_astro
source env_astro/bin/activate

pip install --upgrade pip
pip install numpy scipy pandas matplotlib h5py astropy galpy swiftsimio unyt cartopy plotly
pip install torch torch_geometric torch_cluster
```

Define a working directory `DATA_DIR` with sufficient storage for simulation snapshots, extracted observables, and graph datasets:

```bash
export DATA_DIR="./data_study2"
mkdir -p "$DATA_DIR"
```

---

### 2. Step-by-Step Data Generation (Steps 1–4)

#### Step 1: Progenitor Orbits & Virialized Plummer Initial Conditions
Select the 81 unbiased Milky Way stream progenitors ($R_p \ge 0.0\text{ kpc}$ in `MWPotential2014`, retaining both close-pericenter dissolving systems and outer-halo clusters inside $\max(|x|,|y|,|z|) \le 480\text{ kpc}$) and generate virialized Plummer `.hdf5` initial conditions at $t = -2.0\text{ Gyr}$:

```bash
python3 make_ICs/select_unbiased_orbits.py
python3 make_ICs/generate_unbiased_ics.py
```

#### Step 2: *N*-Body Evolution in `SWIFT`
Evolve each initial condition `make_ICs/initial_conditions_unbiased/IC_<ID>.hdf5` forward for $2.0\text{ Gyr}$ (101 snapshots, $\Delta t = 20\text{ Myr}$) using `SWIFT` compiled with `--with-ext-potential=MWPotential2014` and `configs/params.yml`:

```bash
# Example for a single cluster (or submit slurm/step2_run_swift_array.sh on a SLURM cluster):
swift --external-gravity --self-gravity --threads=8 configs/params.yml
```

#### Step 3: Exact Dynamical Bound Labels & 5D Astrometric Observables
Extract the exact ground-truth Jacobi bound labels (`B_L_snapshot_*.npy`), noiseless 5D ICRS observables (`--obs_dir`), and *Gaia* DR3 magnitude-dependent noisy observables (`--obs_noisy_dir`) simultaneously across 5 Haar-uniform $\mathrm{SO}(3)$ observer rotations per snapshot:

```bash
python3 scripts/extract_observables.py \
    --sim_dir "$DATA_DIR/outputs_unbiased" \
    --labels_dir "$DATA_DIR/labels_clean" \
    --obs_dir "$DATA_DIR/obs_clean" \
    --obs_noisy_dir "$DATA_DIR/obs_gaia_noisy" \
    --n_rotations 5 \
    --norm_mode zscore \
    --abs_mag_g 4.5 \
    --num_workers 8
```

In Python scripts or notebooks, `compute_bound_stars` (`src/bound_labels.py`) can also return the boolean mask of **potential escapers** ($r_{3D} \lt R_J$ and $E_J \ge E_{\mathrm{crit}}$):

```python
from src.bound_labels import compute_bound_stars

is_bound, is_potential_escaper, x_rel, y_rel, z_rel, vx_rel, vy_rel, vz_rel, rj, m_bound, e_j, e_crit = (
    compute_bound_stars(x, y, z, vx, vy, vz, masses, return_potential_escapers=True)
)
```

#### Step 4: Constructing 5D *k*-NN Phase-Space Graphs ($k = 20$)
Convert the standardized observable matrices and labels into PyTorch Geometric `.pt` graphs:

```bash
# 5D Noiseless graphs:
python3 scripts/build_graph_dataset.py \
    --obs_dir "$DATA_DIR/obs_clean" \
    --labels_dir "$DATA_DIR/labels_clean" \
    --output_dir "$DATA_DIR/graph_data_clean" \
    --rotations 5 \
    --k_neighbors 20 \
    --num_workers 8

# 5D Gaia DR3 Noisy graphs:
python3 scripts/build_graph_dataset.py \
    --obs_dir "$DATA_DIR/obs_gaia_noisy" \
    --labels_dir "$DATA_DIR/labels_clean" \
    --output_dir "$DATA_DIR/graph_data_noisy" \
    --rotations 5 \
    --k_neighbors 20 \
    --num_workers 8
```

---

### 3. Reproducing the GNN Training, Curriculum Fine-Tuning & Ablations (Steps 5–7)

All training scripts use a fixed random generator seed (`42`) with `--split_mode ic` to partition the 81 streams into the exact same **65 Train / 8 Validation / 8 Test** simulations, ensuring reproducibility and zero data leakage across snapshots or rotations. Passing `--stochastic_rotation` samples one rotation per snapshot per epoch ($6\,565$ training graphs/epoch).

#### Step 5a: Stage 1 — 5-Observable Noiseless Pre-Training (GCN Baseline & GAT)

```bash
# 1. Train the 3-Layer Residual GCN Baseline (static 5 rotations, 25 epochs, effective batch size 256):
python3 scripts/train_gcn.py \
    --data_folder "$DATA_DIR/graph_data_clean" \
    --n 5 \
    --lr 0.001 \
    --bs 256 \
    --micro_bs 16 \
    --epochs 25 \
    --split_mode ic \
    --save_model "GNN-jed/trained_models/study2/clean_5obs/best_gcn_clean.pth" \
    --log_file "GNN-jed/trained_models/study2/clean_5obs/training_metrics_gcn_clean.log"

# 2. Train the 3-Layer Multi-Head Residual GAT (stochastic SO(3) rotations, 50 epochs, effective batch size 64):
python3 scripts/train_gat.py \
    --data_folder "$DATA_DIR/graph_data_clean" \
    --n 5 \
    --stochastic_rotation \
    --bf16 \
    --lr 0.001 \
    --bs 64 \
    --micro_bs 16 \
    --epochs 50 \
    --dropout 0.4 \
    --weight_decay 1e-3 \
    --scheduler onecycle \
    --patience 10 \
    --split_mode ic \
    --save_model "GNN-jed/trained_models/study2/clean_5obs/best_gat_clean.pth" \
    --log_file "GNN-jed/trained_models/study2/clean_5obs/training_metrics_gat_clean.log"
```

#### Step 5b: Stage 2 — Curriculum Fine-Tuning on *Gaia* DR3 Noisy Graphs (GAT)

Initialize the GAT with the converged Stage 1 noiseless weights (`--pretrained_weights`) and fine-tune on *Gaia* DR3 noisy graphs for up to 30 epochs (`patience = 8`) at $\mathrm{lr} = 5\times 10^{-4}$:

```bash
python3 scripts/train_gat.py \
    --data_folder "$DATA_DIR/graph_data_noisy" \
    --n 5 \
    --stochastic_rotation \
    --bf16 \
    --lr 0.0005 \
    --bs 64 \
    --micro_bs 16 \
    --epochs 30 \
    --dropout 0.4 \
    --weight_decay 1e-3 \
    --scheduler onecycle \
    --patience 8 \
    --split_mode ic \
    --pretrained_weights "GNN-jed/trained_models/study2/clean_5obs/best_gat_clean.pth" \
    --save_model "GNN-jed/trained_models/study2/noisy_5obs/best_gat_noisy.pth" \
    --log_file "GNN-jed/trained_models/study2/noisy_5obs/training_metrics_gat_noisy.log"
```

#### Step 6: 4-Observable Ablation Study ($5\mathrm{D} \to 4\mathrm{D}$ Subspace Weight Transfer)

To train an ablation model dropping one astrometric observable (columns: `0=RA`, `1=Dec`, `2=D_kpc (from Parallax)`, `3=pmRA`, `4=pmDec` — e.g. `1,2,3,4` for `4obs_no_RA`), build the 4D graphs and pass the 5D Clean GAT checkpoint to `--pretrained_weights` (`scripts/train_gat.py` automatically performs variance-preserving $5\mathrm{D} \to 4\mathrm{D}$ channel adaptation):

```bash
export FEAT="1,2,3,4"
export TAG="4obs_no_RA"

# 1. Build 4D Clean & Noisy k-NN graphs:
python3 scripts/build_graph_dataset.py \
    --obs_dir "$DATA_DIR/obs_clean" --labels_dir "$DATA_DIR/labels_clean" \
    --output_dir "$DATA_DIR/graph_data_${TAG}_clean" --feature_indices "$FEAT" --rotations 5 --k_neighbors 20

python3 scripts/build_graph_dataset.py \
    --obs_dir "$DATA_DIR/obs_gaia_noisy" --labels_dir "$DATA_DIR/labels_clean" \
    --output_dir "$DATA_DIR/graph_data_${TAG}_noisy" --feature_indices "$FEAT" --rotations 5 --k_neighbors 20

# 2. Stage 1: 4D Clean GAT pre-training (initialized from 5D Clean GAT weights):
python3 scripts/train_gat.py \
    --data_folder "$DATA_DIR/graph_data_${TAG}_clean" \
    --n 5 --stochastic_rotation --bf16 --lr 0.001 --bs 64 --micro_bs 32 --epochs 20 --split_mode ic \
    --dropout 0.4 --weight_decay 1e-3 --scheduler onecycle --patience 8 \
    --pretrained_weights "GNN-jed/trained_models/study2/clean_5obs/best_gat_clean.pth" \
    --save_model "GNN-jed/trained_models/study2/${TAG}_clean/best_gat_${TAG}_clean.pth" \
    --log_file "GNN-jed/trained_models/study2/${TAG}_clean/training_metrics_gat_${TAG}_clean.log"

# 3. Stage 2: 4D Gaia DR3 Noisy GAT fine-tuning:
python3 scripts/train_gat.py \
    --data_folder "$DATA_DIR/graph_data_${TAG}_noisy" \
    --n 5 --stochastic_rotation --bf16 --lr 0.0005 --bs 64 --micro_bs 32 --epochs 20 --split_mode ic \
    --dropout 0.4 --weight_decay 1e-3 --scheduler onecycle --patience 8 \
    --pretrained_weights "GNN-jed/trained_models/study2/${TAG}_clean/best_gat_${TAG}_clean.pth" \
    --save_model "GNN-jed/trained_models/study2/${TAG}_noisy/best_gat_${TAG}_noisy.pth" \
    --log_file "GNN-jed/trained_models/study2/${TAG}_noisy/training_metrics_gat_${TAG}_noisy.log"
```

#### Step 7: Stochastic $\mathrm{SO}(3)$ GCN Retraining (`slurm/step7_retrain_gcn_stochastic.sh`)

To compare GCN and GAT under identical stochastic $\mathrm{SO}(3)$ rotation sampling, Step 7 archives the static-rotation GCN baseline to `GNN-jed/trained_models/study2/legacy/clean_5obs_static_5rot/` and retrains Clean GCN with `--stochastic_rotation`:

```bash
python3 scripts/train_gcn.py \
    --data_folder "$DATA_DIR/graph_data_clean" \
    --n 5 --stochastic_rotation --bf16 --lr 0.001 --bs 256 --micro_bs 32 \
    --epochs 25 --patience 10 --split_mode ic --allow_non_baseline_gcn \
    --save_model "GNN-jed/trained_models/study2/clean_5obs/best_gcn_clean.pth" \
    --log_file "GNN-jed/trained_models/study2/clean_5obs/training_metrics_gcn_clean.log"
```

#### Multi-Node SLURM Execution & Figure Generation

On a SLURM HPC cluster, Steps 5, 6, and 7 are submitted sequentially with `afterok` dependencies via:

```bash
bash slurm/launch_full_study2_pipeline.sh
```

To generate all diagnostic figures (loss/accuracy curves, confusion matrices, ROC/PR curves, perigalacticon error profiles, RMT weight spectra, and interactive 3D HTML stream visualizations) from any trained checkpoint in [`GNN-jed/trained_models/study2/weights/`](GNN-jed/trained_models/study2/weights/):

```bash
python3 scripts/generate_study2_figures.py \
    --output_dir "GNN-jed/trained_models/study2/clean_5obs" \
    --tag "clean_5obs" \
    --gcn_log "GNN-jed/trained_models/study2/clean_5obs/training_metrics_gcn_clean.log" \
    --gat_log "GNN-jed/trained_models/study2/clean_5obs/training_metrics_gat_clean.log" \
    --gcn_weights "GNN-jed/trained_models/study2/weights/best_gcn_clean.pth" \
    --gat_weights "GNN-jed/trained_models/study2/weights/best_gat_clean.pth" \
    --graph_dir "$DATA_DIR/graph_data_clean" \
    --sim_dir "$DATA_DIR/outputs_unbiased" \
    --labels_dir "$DATA_DIR/labels_clean"
```

---

## Roadmap to Real *Gaia* DR3 Observational Inference (Study 2 V2) & Compute Estimates

While the active Study 2 (V1) pipeline cleanly isolates 5D vs. 4D astrometric information content on synthetic simulations, **Study 2 V2** adapts the pipeline for direct out-of-distribution inference on real *Gaia* DR3 clusters and stellar streams — **without re-running any of the 81 `SWIFT` *N*-body integrations** (see `THEORETICAL_CORRECTIONS_LOG.md`, Section 8, for complete mathematical derivations):

### 1. Planned Observational Domain Fixes (V1 $\to$ V2)
1. **Stellar Luminosity Function, Flux Limit ($G \le 20.7$) & Per-Star Error Inputs:** Replace the fixed $M_G = 4.5$ assumption with realistic IMF magnitude sampling (Kroupa 2001), drop faint stars below *Gaia* completeness ($G > G_{\max} \approx 20.7$), compute star-by-star Lindegren et al. (2021) uncertainties, and pass normalized per-star measurement errors $(\sigma_\varpi, \sigma_{\mu_{\alpha^*}}, \sigma_{\mu_\delta})$ and parallax SNR as explicit node channels.
2. **Linear Parallax Representation:** Pass signed noisy parallax $\varpi_{\mathrm{obs}}$ [mas] directly instead of $D = 1/\operatorname{clip}(\varpi_{\mathrm{obs}}, 0.005, 10)$, eliminating non-linear Lutz–Kelker distance bias and $200\text{ kpc}$ boundary pile-ups (Bailer-Jones 2015; Luri et al. 2018).
3. **2D On-Sky Tangential $k$-NN Topology:** Build `cKDTree` edges exclusively in the high-SNR 2D sky plane $(\Delta\alpha^*, \Delta\delta)$ rather than in noisy 4D/5D feature space.
4. **Physical-Scale Normalization & $N$-Invariant Density:** Replace per-snapshot unit-variance Z-scores (`--norm_mode zscore`) with fixed galactic scale factors (`--norm_mode physical`) and surface-area/population-normalized local density invariants so compact cores and extended streams are distinguishable regardless of observed star count $N_{\mathrm{obs}}$.
5. **Milky Way Field-Star Contamination:** Inject foreground/background Milky Way contaminants (`label = unbound`) into simulated cones so the classifier separates stream/cluster stars from unrelated field stars.
6. **Explicit Potential Escaper Labeling:** Export `is_potential_escaper` ($r_{3D} < R_J, E_J \ge E_{\mathrm{crit}}$) alongside `is_bound` to support 3-class or multi-task classification rather than pooling Lagrange-point escapers into distant tails.

### 2. Automated Selection of the Optimal 4D Model & End-to-End V2 Job Orchestration
Once the active Step 6 array (`41146835_[0-4%1]`) finishes all five noisy 4D experiments (`no_RA`, `no_Dec`, `no_Parallax`, `no_pmRA`, `no_pmDec`), the winning 4D observable subset $\mathcal{I}_{\text{best4D}}$ will be selected automatically on the held-out 8-stream **Test set** (`Gaia DR3 Noisy`) by jointly maximizing **Test F1-score** and **Balanced Accuracy** subject to minimal **Core FNR** (`Core Rec` preservation) and lowest noisy Test BCE loss. Rather than repeating all five 4D runs from scratch in V2, the V2 launcher can then execute a **100% automated SLURM dependency chain** (`afterok`) inside an isolated `study2_v2/` namespace: parsing the Step 6 winning 4D subset $\to$ extracting V2 observables/labels (`Step 3 V2`) $\to$ building 2D sky $k$-NN graphs (`Step 4 V2`) $\to$ training V2 5D Curriculum GAT (`Step 5 V2`) $\to$ transferring $5\mathrm{D} \to 4\mathrm{D}$ weights onto $\mathcal{I}_{\text{best4D}}$ (`Step 6 V2`) $\to$ evaluating on real *Gaia* DR3 catalogs.

### 3. Empirical Compute Budget & Projected Wall-Clock Runtimes (EPFL Jed: 2 Nodes $\times$ 36 CPUs = 72 CPUs, 246 GB/Node)

Observed wall-clock timings from completed Jed runs (`40994872`, `40994873`, `40994875`, `40994877`, active `41146834` at `~45.8 min/epoch` with `--stochastic_rotation`) provide the baseline for remaining V1 and planned V2 jobs:

| Pipeline Phase | Allocation | Status / Observed Speed | Estimated Remaining / Projected Wall-Clock |
| :--- | :--- | :--- | :--- |
| **V1 Steps 1–4 (`SWIFT` + Obs + 5D Graphs)** | `1` node (`36` CPUs) | **Complete** (`SWIFT`: `0.6–2.3 h/IC`; Obs: `11.5 min`; Graphs: `~16–22 min`) | **0 h (Complete & Reusable in V2)** |
| **V1 Step 5 (`41146834`, Stage 1 Clean GAT)** | `2` nodes (`72` CPUs) | Running Ep `28 / 50` (`~45.8 min/ep`, `21h01m` elapsed) | **`~7.6 – 17.6 h`** (`10–23` ep left) |
| **V1 Step 5 (`41146834`, Stage 2 Noisy GAT)** | `2` nodes (`72` CPUs) | Queued inside `41146834` (`epochs=30`, `patience=8`) | **`~7.5 – 23 h`** (realistic `~10–15 h`) |
| **V1 Step 6 (`41146835_[0-4%1]`, 5 Sequential 4D Runs)** | `2` nodes (`72` CPUs) | Pending (`5` tasks $\times$ `[graphs + 20 Clean + 20 Noisy ep]`) | **`~80 – 110 h`** realistic (`~70–150 h` min/max) |
| **V1 Step 7 (`41146836`, Stochastic Clean GCN)** | `2` nodes (`72` CPUs) | Pending (`25` epochs max, `--bs 256`, `patience=10`) | **`~2 – 4 h`** |
| **Total Remaining Active V1 Pipeline** | `2` nodes (`72` CPUs) | Steps 5 (rem.) + 6 + 7 | **`~97 – 155 h` (~4.0 – 6.5 days)** |
| **V2 Step 3–4 (IMF Cut + Contaminants + 2D Graphs)** | `1` node array / `36` CPUs | Reuses all 81 `SWIFT` simulations | **`~25 – 50 min`** total |
| **V2 Step 5 (V2 5D Curriculum GAT)** | `2` nodes (`72` CPUs) | Fewer edges per cluster after $G \le 20.7$ cut (`~20–30 min/ep`) | **`~20 – 28 h`** |
| **V2 Step 6-Best4D (Winning 4D V2 + Gaia Eval)** | `2` nodes (`72` CPUs) | Targeted retraining on winning $\mathcal{I}_{\text{best4D}}$ + real *Gaia* inference | **`~10 – 14 h`** |
| **Total V2 Targeted Pipeline (5D + Best-4D)** | `2` nodes (`72` CPUs) | **End-to-end reusing 81 `SWIFT` runs** | **`~30 – 43 h` (~1.3 – 1.8 days)** *(or `~16–22 h` on 4 nodes)* |

*(Running all five V2 4D ablations instead of only the winning $\mathcal{I}_{\text{best4D}}$ would add `~40–55 h`, bringing an exhaustive V2 sweep to `~70–100 h` / ~3–4 days on 2 nodes.)*

---

## Theoretical References

1. **Ibata, R., Malhan, K., Tenorio, L., et al. (2024).** *Charting the Galactic Acceleration Field. II. A Global Mass Model of the Milky Way from the STREAMFINDER Atlas of Stellar Streams Detected in Gaia DR3.* The Astrophysical Journal, 967(1), 89. [doi:10.3847/1538-4357/ad382d](https://doi.org/10.3847/1538-4357/ad382d)
2. **Küpper, A. H. W., Kroupa, P., Baumgardt, H., & Heggie, D. C. (2010a).** *Tidal tails of star clusters.* Monthly Notices of the Royal Astronomical Society, 401(1), 105–120. [doi:10.1111/j.1365-2966.2009.15690.x](https://doi.org/10.1111/j.1365-2966.2009.15690.x)
3. **Küpper, A. H. W., Kroupa, P., Baumgardt, H., & Heggie, D. C. (2010b).** *Palaeontology of star clusters: faint haloes and eccentric orbits.* Monthly Notices of the Royal Astronomical Society, 407(3), 1580–1600. [doi:10.1111/j.1365-2966.2010.17084.x](https://doi.org/10.1111/j.1365-2966.2010.17084.x)
4. **Binney, J., & Tremaine, S. (2008).** *Galactic Dynamics* (2nd ed.). Princeton University Press.
5. **King, I. (1962).** *The structure of star clusters. I. An empirical density law.* The Astronomical Journal, 67, 471–485. [doi:10.1086/108756](https://doi.org/10.1086/108756)
6. **Plummer, H. C. (1911).** *On the problem of distribution in globular star clusters.* Monthly Notices of the Royal Astronomical Society, 71(5), 460–470. [doi:10.1093/mnras/71.5.460](https://doi.org/10.1093/mnras/71.5.460)
7. **Casertano, S., & Hut, P. (1985).** *Core radius and density measurements in N-body experiments: Connections with theoretical and observational definitions.* The Astrophysical Journal, 298, 80–94. [doi:10.1086/163589](https://doi.org/10.1086/163589)
8. **Bovy, J. (2015).** *galpy: A python Library for Galactic Dynamics.* The Astrophysical Journal Supplement Series, 216(2), 29. [doi:10.1088/0067-0049/216/2/29](https://doi.org/10.1088/0067-0049/216/2/29)
9. **Schaller, M., Borrow, J., Draper, P. W., et al. (2024).** *SWIFT: A modern highly-parallel gravity and smoothed particle hydrodynamics solver for cosmological and planetary simulations.* Monthly Notices of the Royal Astronomical Society, 530(2), 2378–2419. [doi:10.1093/mnras/stae922](https://doi.org/10.1093/mnras/stae922)
10. **Gaia Collaboration, Prusti, T., de Bruijne, J. H. J., et al. (2016).** *The Gaia mission.* Astronomy & Astrophysics, 595, A1. [doi:10.1051/0004-6361/201629272](https://doi.org/10.1051/0004-6361/201629272)
11. **Lindegren, L., Klioner, S. A., Hernández, J., et al. (2021).** *Gaia Early Data Release 3: The astrometric solution.* Astronomy & Astrophysics, 649, A2. [doi:10.1051/0004-6361/202039709](https://doi.org/10.1051/0004-6361/202039709)
12. **Luri, X., Brown, A. G. A., Sarro, L. M., et al. (2018).** *Gaia Data Release 2: Using Gaia parallaxes.* Astronomy & Astrophysics, 616, A9. [doi:10.1051/0004-6361/201832964](https://doi.org/10.1051/0004-6361/201832964)
13. **Bailer-Jones, C. A. L. (2015).** *Estimating Distances from Parallaxes.* Publications of the Astronomical Society of the Pacific, 127(956), 994–1009. [doi:10.1086/683116](https://doi.org/10.1086/683116)
14. **Kroupa, P. (2001).** *On the variation of the initial mass function.* Monthly Notices of the Royal Astronomical Society, 322(2), 231–246. [doi:10.1046/j.1365-8711.2001.04022.x](https://doi.org/10.1046/j.1365-8711.2001.04022.x)
15. **Kipf, T. N., & Welling, M. (2017).** *Semi-Supervised Classification with Graph Convolutional Networks.* International Conference on Learning Representations (ICLR 2017). [arXiv:1609.02907](https://arxiv.org/abs/1609.02907)
16. **Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018).** *Graph Attention Networks.* International Conference on Learning Representations (ICLR 2018). [arXiv:1710.10903](https://arxiv.org/abs/1710.10903)
17. **Alon, U., & Yahav, E. (2021).** *On the Bottleneck of Graph Neural Networks and its Practical Implications.* International Conference on Learning Representations (ICLR 2021). [arXiv:2006.05205](https://arxiv.org/abs/2006.05205)
18. **Martin, C. H., & Mahoney, M. W. (2021).** *Implicit Self-Regularization in Deep Neural Networks: Evidence from Random Matrix Theory and Implications for Learning.* Journal of Machine Learning Research, 22(165), 1–73. [arXiv:1810.01075](https://arxiv.org/abs/1810.01075)
