# Repository Architecture and Pipeline Map

**Project:** *Identifying Potential Escapers with Graph Neural Networks: Recovering Cluster Dynamics from Partial Kinematics*
**Author:** Adrian Cozma (EPFL / LASTRO)
**Supervisors:** Prof. Yves Revaz & Dr. Srikanth Nagesh

This document maps the directory layout, modules, data directories, and SLURM execution scripts across **Study 1** (MA2 Semester Project baseline, preserved untouched) and **Study 2** (Peer-Reviewed Journal Article pipeline).

---

## 1. Directory Structure

```text
.
│
├── README.md                                  # Main scientific overview and end-to-end reproducibility guide
├── REPO_MAP.md                                # Repository architecture and Study 1 vs. Study 2 file map
├── LIVE_STATUS.md                             # Live training progress and metrics summary
│
├── make_ICs/                                  # STEP 1: Orbit backward integration & Plummer IC generation
│   ├── Ibata_catalog.txt                      # Raw Gaia DR3 stellar stream catalog (Ibata et al. 2024)
│   ├── stars_6D_ibata2024.csv                 # Filtered 6D phase-space catalog (1,703 stars across 84 streams)
│   ├── IC_plummer_swift.py                    # Virialized Plummer sphere generator for SWIFT (.hdf5)
│   ├── select_unbiased_orbits.py              # [Study 2] Unbiased orbit selector (R_p >= 0.0 kpc, 81 streams inside SWIFT volume)
│   ├── generate_unbiased_ics.py               # [Study 2] Batch Plummer HDF5 IC generator for unbiased seeds
│   ├── SWIFT_seeds.npy                        # [Study 1 - Untouched] Original 75 progenitor seeds
│   ├── orbits/                                # [Study 1 - Untouched] Original 75 integrated trajectories
│   ├── SWIFT_seeds_unbiased.npy               # [Study 2] Unbiased progenitor seeds at t = -2 Gyr (81 streams)
│   ├── unbiased_orbits_summary.csv            # [Study 2] Orbital diagnostics (R_p, R_a, e, r_init) for all 84 streams
│   ├── orbits_unbiased/                       # [Study 2] Integrated trajectories orbit_<ID>.txt for Study 2
│   └── initial_conditions_unbiased/           # [Study 2 - Git-ignored] Generated IC_<ID>.hdf5 files for SWIFT
│
├── configs/                                   # STEP 2: SWIFT N-body simulation configuration
│   ├── params.yml                             # SWIFT runtime configuration (MWPotential2014, adaptive FMM gravity)
│   └── dependency_graph_0.csv                 # SWIFT task dependency diagnostics
│
├── src/                                       # STEP 3: Core physical, astrometric & GNN library
│   ├── bound_labels.py                        # [Study 2] 3D k-NN density + Shrinking Spheres + Gaussian Mean-Shift
│   │                                          #   core centering, exact monotonic Roche/King R_J solver, Jacobi energy
│   ├── observables_matrix.py                  # [Study 2] Haar-uniform SO(3) rotations, circular RA centering,
│   │                                          #   Gaia DR3 magnitude-dependent noise model & MAD-winsorized scaling
│   ├── gnn_models.py                          # [Study 2] Local k-NN phase-space relational encoder, Residual GCN &
│   │                                          #   Multi-Head Residual GAT (4 heads, LayerNorm, ego-relational readout)
│   ├── oom_Pin_and_offload.py                 # [Study 2] Distributed DDP initializer & cgroup-aware disk activation offloader
│   ├── gaia_v2.py                             # [Study 2 V2] Kroupa IMF, G<=20.7 cut, Lindegren sigmas, linear parallax & MW field mixing
│   ├── gnn_models_v2.py                       # [Study 2 V2] N-invariant sky-kNN GAT_V2, edge_attr, err_x, 3-class & 5D->4D transfer
│   └── Stellar_objects/                       # 3D cluster density, line-of-sight dispersion, and sequence utilities
│       ├── cluster_v4_properties.py
│       ├── dispersion_vel_LOS.py
│       ├── changeLocationVelocity.py
│       └── visu.py
│
├── scripts/                                   # STEPS 3-6: Production CLI scripts
│   ├── extract_observables.py                 # Parallel extraction of bound labels + Clean & Gaia DR3 Noisy observables
│   ├── build_graph_dataset.py                 # Parallel O(N log N) cKDTree k-NN graph builder (5D or 4D ablation subsets)
│   ├── distributed_trainer.py                 # Unified DDP, micro-batch gradient accumulation & checkpoint engine
│   ├── train_gcn.py                           # 3-layer Residual GCN trainer (--split_mode ic, clean_5obs baseline)
│   ├── train_gat.py                           # 3-layer Multi-Head Residual GAT trainer (p=0.4 dropout, AdamW wd=1e-3, OneCycleLR, CoreFNR/CoreRec)
│   ├── live_plot_utils.py                     # Real-time epoch-by-epoch live learning curve generator (.png)
│   ├── live_monitor.py                        # Automated training telemetry and status updater
│   ├── generate_study2_figures.py             # Automated generator for all report/README plots (.png + interactive .html)
│   ├── v2_select_best_4d.py                   # [Study 2 V2] Automated Step 6 best-4D configuration parser & selector
│   ├── v2_extract_observables.py              # [Study 2 V2] Isolated 3-class Gaia DR3 + MW contaminant observable extractor
│   ├── v2_build_graph_dataset.py              # [Study 2 V2] Pure 2D-sky cKDTree graph builder with [dRA*, dDec, theta] edge attributes
│   ├── v2_train_gat.py                        # [Study 2 V2] DDP GAT_V2 trainer (Clean 5D -> Noisy 5D & Clean Best-4D -> Noisy Best-4D)
│   └── v2_infer_gaia_dr3.py                   # [Study 2 V2] Real Gaia DR3 / Ibata (2024) catalog inference (auto 5D/4D fallback)
│
├── tests/                                     # Unit & end-to-end verification tests
│   └── test_v2_pipeline.py                    # [Study 2 V2] IMF/parallax/N-invariance/GAT_V2/Best-4D selector/smoke test suite
│
├── slurm/                                     # SLURM batch scripts for HPC execution
│   ├── step1_orbits_and_ics.sh                # Generates unbiased orbits and Plummer HDF5 initial conditions
│   ├── step2_run_swift_array.sh               # Runs SWIFT N-body simulations as a SLURM job array
│   ├── step3_extract_observables.sh           # Extracts exact bound labels + Clean & Gaia DR3 Noisy observables
│   ├── step4_build_graphs.sh                  # Builds 5D (and optional 4D) k-NN graphs in $DATA_DIR
│   ├── step5_train_curriculum_and_plot.sh     # Runs Stage 1 Clean (GCN + GAT) -> Stage 2 Noisy (GAT only) + figures
│   ├── step6_ablation_4obs.sh                 # Runs 4-observable GAT-only ablation study (Clean -> Noisy)
│   ├── launch_full_study2_pipeline.sh         # Master script chaining Steps 5 and 6 via SLURM afterok dependencies
│   ├── v2_step3_extract_obs.sh                # [Study 2 V2] Parallel SLURM array extracting Gaia DR3 V2 observables
│   ├── v2_step4_build_graphs.sh               # [Study 2 V2] Best-4D selector + Clean/Noisy sky-kNN graph build
│   ├── v2_step5_train_5d.sh                   # [Study 2 V2] 2-node DDP Clean 5D -> Noisy 5D GAT_V2 training
│   ├── v2_step6_train_best4d.sh               # [Study 2 V2] 2-node DDP Best-4D transfer training + Ibata (2024) inference
│   └── launch_study2_v2.sh                    # [Study 2 V2] Automated SLURM launcher chained after active V1 pipeline jobs
│
├── GNN-jed/                                   # Trained models & intermediate observable/label matrices
│   ├── trained_models/
│   │   ├── GCN-01/                            # [Study 1 - Untouched] Baseline GCN weights and logs
│   │   ├── GCN-02/                            # [Study 1 - Untouched] Baseline GAT weights and logs
│   │   └── study2/                            # [Study 2] Clean & Gaia DR3 Noisy checkpoints, logs, PNG & HTML figures
│   │       └── weights/                       # [Study 2] Central archive of all best_*.pth checkpoints & registry
│   ├── labels_bound_stars/                    # [Study 1 - Untouched] Original bound labels
│   ├── labels_clean/                          # [Study 2 - Git-ignored] Exact bound labels B_L_snapshot_*.npy
│   ├── obs_clean/                             # [Study 2 - Git-ignored] Noiseless standardized observable matrices
│   └── obs_gaia_noisy/                        # [Study 2 - Git-ignored] Gaia DR3 noisy standardized observable matrices
│
├── visualization/                             # Plotting utilities and analysis notebooks
│   ├── plot_swift_output.py                   # HDF5 snapshot reader (load_snapshot)
│   ├── plot_metric_live.py                    # Training curve comparison plotter
│   ├── visu_stream_outputs.py                 # 3D stream and orbit visualization
│   ├── fraction_bound_stars.ipynb             # Bound fraction evolution vs. orbital pericenter R_p1
│   └── final_sims_vizu.ipynb                  # Sky projections and galactocentric stream views
│
└── notebooks/                                 # Interactive verification notebooks
    ├── labels_data_matrices.ipynb
    ├── graph_data.ipynb
    └── test-model-GCN.ipynb
```

---

## 2. Generated Data Directories (`$DATA_DIR`)

Bulk simulation outputs, extracted observable matrices, and PyTorch Geometric `.pt` graph datasets are stored outside Git version control in a user-defined `$DATA_DIR` (e.g., `./data_study2` or `$SCRATCH/study2` on an HPC cluster):

| Dataset / Artifact | Default Relative / Cluster Path | Description |
| :--- | :--- | :--- |
| **SWIFT Binary** | `swift` (`--with-ext-potential=MWPotential2014`) | Compiled `SWIFT` $N$-body solver with analytical Milky Way potential |
| **Study 2 SWIFT ICs** | `make_ICs/initial_conditions_unbiased/` | 81 virialized Plummer HDF5 initial conditions ($R_p \ge 0.0\text{ kpc}$) |
| **Study 2 SWIFT Outputs** | `$DATA_DIR/outputs_unbiased/` | 81 directories $\times$ 101 HDF5 snapshots ($2.0\text{ Gyr}$, $\Delta t = 20\text{ Myr}$) |
| **Study 2 Bound Labels** | `$DATA_DIR/labels_clean/` | Exact bound & potential-escaper labels `B_L_snapshot_*.npy` |
| **Study 2 Clean Observables** | `$DATA_DIR/obs_clean/` | Standardized noiseless 5D ICRS observable matrices |
| **Study 2 Gaia Noisy Observables** | `$DATA_DIR/obs_gaia_noisy/` | Standardized *Gaia* DR3 noisy 5D ICRS observable matrices |
| **Study 2 5D Clean Graphs** | `$DATA_DIR/graph_data_clean/` | PyTorch Geometric `.pt` 5D $k$-NN graphs ($k = 20$, noiseless) |
| **Study 2 5D Noisy Graphs** | `$DATA_DIR/graph_data_noisy/` | PyTorch Geometric `.pt` 5D $k$-NN graphs ($k = 20$, *Gaia* DR3 noise) |
| **Study 2 4D Ablation Graphs** | `$DATA_DIR/graph_data_4obs_<tag>_<clean\|noisy>/` | 4-observable $k$-NN graphs for feature-ablation experiments |
| **Trained Model Weights** | `GNN-jed/trained_models/study2/weights/` | Version-controlled `.pth` checkpoints & `MODEL_WEIGHTS_REGISTRY.md` |
| **Study 2 V2 Observables** | `/scratch/cozma/study2_v2/obs_unbiased/` | Gaia DR3 *G* <= 20.7 linear-parallax & MW contaminant `.npz` archives |
| **Study 2 V2 Sky Graphs** | `/scratch/cozma/study2_v2/graphs_<clean\|noisy>/` | Pure 2D-sky *k*-NN PyG graphs (`edge_attr`, `err_x`, 3-class `y`) |
| **Study 2 V2 Models & Logs** | `GNN-jed/trained_models/study2_v2/` | Best-4D config, V2 5D/4D checkpoints, logs & Gaia DR3 inference outputs |

---

## 3. Hardware Constraints & Execution Architecture (Jed HPC Cluster)

- **Allocation Ceiling:** Maximum **2 nodes $\times$ 36 physical cores/node = 72 CPUs total**, with **246 GB RAM/node (492 GB RAM total)** under `--qos=academic --partition=academic` (Dual-socket Intel Xeon Platinum 8360Y Ice Lake-SP, 4 NUMA domains of 18 cores per node, no hardware `avx512_bf16`).
- **Scratch Storage Warning:** Heavy intermediates live on `/scratch/cozma/study2` and `/scratch/cozma/study2_v2`. All converged model checkpoints (`.pth`), manifests (`weights_manifest.json`), and training logs (`.log`) are mirrored inside the Git repository under `GNN-jed/trained_models/`.

---

## 4. Completed Runs & Preserved Assets (DO NOT RE-RUN OR OVERWRITE)

1. **Step 1 & Step 2 (`SWIFT` N-Body Simulations):** `81 / 81` unbiased streams (`101` HDF5 snapshots each, $2.0\text{ Gyr}$) complete in `/scratch/cozma/study2/outputs_unbiased/`.
2. **Step 3 & Step 4 (Study 2 V1 Observables, Labels & $k=20$ Graphs):**
   - `81 / 81` exact Küpper bound labels (`labels_clean/`), clean observables (`obs_clean/`), and noisy observables (`obs_gaia_noisy/`).
   - `40,905` 5D `.pt` graphs each in `/scratch/cozma/graph_data_clean` and `/scratch/cozma/graph_data_noisy`.
   - All 10 pre-built 4D ablation graph directories (`graph_data_4obs_no_{RA,Dec,Parallax,pmRA,pmDec}_{clean,noisy}`, `40,905` `.pt` graphs each) are already built on `/scratch/cozma/`.
3. **Completed Study 2 V1 Model Checkpoints (`GNN-jed/trained_models/study2/weights/`):**
   - `best_gcn_clean.pth` (Static 5-rot, `dropout=0.2`, Ep 24/25): `Val Loss = 0.1817 (92.32%)`, **`Test Acc = 92.17%`, `BalAcc = 91.59%`, `Test F1 = 0.8886`**.
   - `best_gat_clean_ep18_backup.pth` (Static 5-rot, `dropout=0.2`, Ep 18/22): `Val Loss = 0.1470 (94.20%)`, **`Test Acc = 93.04%`, `BalAcc = 93.15%`, `Test F1 = 0.9035`, `CoreFNR = 6.49%`**.
   - `best_gat_clean.pth` (Stochastic 5-rot, `dropout=0.4`, Ep 35/45): `Val Loss = 0.2071 (91.89%)`, **`Test Acc = 91.43%`, `BalAcc = 91.28%`, `Test F1 = 0.8807`, `CoreFNR = 9.22%`**.
   - `best_gat_noisy.pth` (Stochastic 5-rot Curriculum on V1 Gaia DR3 noise, `dropout=0.4`, Best Ep 12/20): `Val Loss = 0.4014 (81.76%)`, **`Test Loss = 0.3181`, `Test Acc = 84.91%`, `BalAcc = 83.88%`, `Test F1 = 0.7879`, `CoreFNR = 19.52%` (`CoreRec = 80.48%`)** — **COMPLETED** (all Stage 2 PNG/HTML figures generated in `GNN-jed/trained_models/study2/noisy_5obs/`).

---

## 5. V1 vs. V2 Pipeline Map & Implemented Upgrades for Agentic Sessions

- **Study 2 V1 (`src/observables_matrix.py`, `src/gnn_models.py`, `scripts/distributed_trainer.py`):**
  - Uses monochromatic $M_G = 4.5$, clipped inverse parallax $D = 1/\operatorname{clip}(\varpi_{\mathrm{obs}}, 0.005, 10.0)\text{ kpc}$, `zscore` ($6\sigma_{\mathrm{MAD}}$ winsorized) standardization, and 5D Euclidean `cKDTree` $k$-NN graphs ($k=20$) with binary `is_bound` labels.
  - **Step 5 (5D Clean + 5D Noisy) is 100% COMPLETED.**
  - **Step 6 (4D Ablations, `41146835_[0-4%1]`) is currently RUNNING** and **Step 7 (`41146836`, Stochastic Clean GCN)** is queued immediately after.
- **Study 2 V2 (`src/gaia_v2.py`, `src/gnn_models_v2.py`, `scripts/v2_*.py`, `slurm/v2_*.sh`):**
  - Introduces Kroupa IMF + RGB luminosity function, $G \le 20.7$ cut, linear signed parallax $\varpi_{\mathrm{obs}}$ (mas), 4 uncertainty/SNR features (`err_x`), 2D sky tangent-plane $k$-NN topology (`edge_attr = [dRA*, dDec, theta_sky]`), $N$-invariant surface density normalization, Milky Way field contaminants, and 3-class labels (`0: Unbound/Field`, `1: Potential Escaper`, `2: Bound Core`).
  - **Upgrades Applied (Active for Step 6, Step 7, and V2 Jobs `41185009–41185012`):**
    1. **Extended RGB Tip ($M_G \in [-2.5, 4.2]$) & Relaxed-Cut Sigma Guard (`src/gaia_v2.py:114-121, 293`):** Ensures evolved red-giant branch stars remain visible out to $D \sim 300\text{ kpc}$ and caps `g_app_vis` noise evaluation at `g_max = 20.7` if `cut_relaxed` triggers.
    2. **Stream-Referenced Centering & Heliocentric-Invariant Parallax (`src/gaia_v2.py:434-485`, `scripts/v2_extract_observables.py:189-195`):** Centers parallax and proper motions on the stream member reference distribution (`reference_mask=~is_field`), preventing high field-star contamination from shifting the coordinate origin.
    3. **Decoupled Feature vs. Attention Dropout & Zero Final-Layer Edge Dropout (`src/gnn_models.py:117-135`, `src/gnn_models_v2.py:119-160`):** Caps internal `GATConv` attention `edge_dropout` at $\le 0.15$, sets `dropout=0.0` on the final output `conv3` layer, and sets `--dropout 0.25 --weight_decay 5e-4` across Step 6 and V2 scripts.
    4. **In-RAM Graph + `h0` Relational Feature Caching & Node-Weighted Accumulation (`scripts/distributed_trainer.py:132-186, 406-438`, `scripts/v2_train_gat.py:128-165, 509-549`):** Caches loaded `.pt` graphs and precomputed `h0` relational features in RAM (up to 55% of the 246 GB/node ceiling) with single-threaded persistent DataLoader workers (`torch.set_num_threads(1)`), eliminating repeated `/scratch` GPFS I/O and $O(E)$ scatter recomputations after initial epochs.

