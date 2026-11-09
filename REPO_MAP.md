# Repository Architecture and Pipeline Map

**Project:** *Identifying Potential Escapers with Graph Neural Networks: Recovering Cluster Dynamics from Partial Kinematics*
**Author:** Adrian Cozma (EPFL / LASTRO)
**Supervisors:** Prof. Yves Revaz & Dr. Sruthi Nagesh

This document maps the directory layout, modules, data directories, and SLURM execution scripts across **Study 1** (MA2 Semester Project baseline, preserved untouched) and **Study 2** (Peer-Reviewed Journal Article pipeline).

---

## 1. Directory Structure

```text
/home/cozma/NeuralNetwork/
│
├── README.md                                  # Main scientific overview and Study 1 baseline results
├── REPO_MAP.md                                # Repository architecture and Study 1 vs. Study 2 file map
├── THEORETICAL_CORRECTIONS_LOG.md             # [Git-ignored] Mathematical & physical corrections log
├── .agents/rules/scientific_tracking.md       # [Git-ignored] Permanent engineering and scientific rules
│
├── make_ICs/                                  # STEP 1: Orbit backward integration & Plummer IC generation
│   ├── Ibata_catalog.txt                      # Raw Gaia DR3 stellar stream catalog (Ibata et al. 2024)
│   ├── stars_6D_ibata2024.csv                 # Filtered 6D phase-space catalog (1,703 stars across 84 streams)
│   ├── IC_plummer_swift.py                    # Virialized Plummer sphere generator for SWIFT (.hdf5)
│   ├── select_unbiased_orbits.py              # [Study 2] Unbiased orbit selector (R_p >= 5 kpc, no 100 kpc cutoff)
│   ├── generate_unbiased_ics.py               # [Study 2] Batch Plummer HDF5 IC generator for unbiased seeds
│   ├── SWIFT_seeds.npy                        # [Study 1 - Untouched] Original 75 progenitor seeds
│   ├── orbits/                                # [Study 1 - Untouched] Original 75 integrated trajectories
│   ├── SWIFT_seeds_unbiased.npy               # [Study 2] Unbiased progenitor seeds at t = -2 Gyr
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
│   └── Stellar_objects/                       # 3D cluster density, line-of-sight dispersion, and sequence utilities
│       ├── cluster_v4_properties.py
│       ├── dispersion_vel_LOS.py
│       ├── changeLocationVelocity.py
│       └── visu.py
│
├── scripts/                                   # STEPS 3-6: Production CLI scripts (72-core parallelized)
│   ├── extract_observables.py                 # Parallel extraction of bound labels + Clean & Gaia DR3 Noisy observables
│   ├── build_graph_dataset.py                 # Parallel O(N log N) cKDTree k-NN graph builder (5D or 4D ablation subsets)
│   ├── train_gcn.py                           # 3-layer Residual GCN trainer (--split_mode ic, Stage 1 Clean & Stage 2 Noisy)
│   ├── train_gat.py                           # 3-layer Multi-Head Residual GAT trainer (--split_mode ic, Clean & Noisy)
│   ├── live_plot_utils.py                     # Real-time epoch-by-epoch live learning curve generator (.png)
│   ├── live_monitor.py                        # Live SLURM self-healing daemon & ntfy.sh / GitHub telemetry publisher
│   └── generate_study2_figures.py             # Automated generator for all report/README plots (.png + interactive .html)
│
├── slurm/                                     # SLURM batch scripts for EPFL Jed HPC (72 CPUs/node, academic QOS)
│   ├── step1_orbits_and_ics.sh                # Generates unbiased orbits and Plummer HDF5 initial conditions
│   ├── step2_run_swift_array.sh               # Runs SWIFT N-body simulations as a SLURM job array
│   ├── step3_extract_observables.sh           # Extracts exact bound labels + Clean & Gaia DR3 Noisy observables
│   ├── step4_build_graphs.sh                  # Builds 5D (and optional 4D) k-NN graphs in /scratch/cozma/
│   ├── step5_train_curriculum_and_plot.sh     # Runs Stage 1 Clean (GCN + GAT) -> Stage 2 Noisy (GAT only) + figures
│   ├── step6_ablation_4obs.sh                 # Runs 4-observable GAT-only ablation study (Clean -> Noisy)
│   └── launch_full_study2_pipeline.sh         # Master script chaining Steps 1 -> 5 via SLURM afterok dependencies
│
├── GNN-jed/                                   # Trained models & intermediate observable/label matrices
│   ├── trained_models/
│   │   ├── GCN-01/                            # [Study 1 - Untouched] Baseline GCN weights and logs
│   │   ├── GCN-02/                            # [Study 1 - Untouched] Baseline GAT weights and logs
│   │   └── study2/                            # [Study 2] Clean & Gaia DR3 Noisy checkpoints, logs, PNG & HTML figures
│   │       └── weights/                       # [Study 2] Central archive of all best_*.pth checkpoints & registry
│   ├── obs_all_stars -> /scratch/cozma/study1/obs_all_stars   # [Study 1 - Content untouched] 58 GB original observable matrices (symlink)
│   ├── labels_bound_stars/                    # [Study 1 - Untouched] 586 MB original bound labels
│   ├── labels_clean -> /scratch/cozma/study2/labels_clean     # [Study 2 - Git-ignored] Exact bound labels B_L_snapshot_*.npy (symlink)
│   ├── obs_clean -> /scratch/cozma/study2/obs_clean           # [Study 2 - Git-ignored] Noiseless standardized observable matrices (symlink)
│   └── obs_gaia_noisy -> /scratch/cozma/study2/obs_gaia_noisy # [Study 2 - Git-ignored] Gaia DR3 noisy standardized observable matrices (symlink)
│
├── visualization/                             # Plotting utilities and analysis notebooks
│   ├── plot_swift_output.py                   # HDF5 snapshot reader (load_snapshot)
│   ├── plot_metric_live.py                    # Training curve comparison plotter
│   ├── visu_stream_outputs.py                 # 3D stream and orbit visualization
│   ├── fraction_bound_stars.ipynb             # Bound fraction evolution vs. orbital pericenter R_p1
│   └── final_sims_vizu.ipynb                  # Sky projections and galactocentric stream views
│
├── notebooks/                                 # Interactive verification notebooks
│   ├── labels_data_matrices.ipynb
│   ├── graph_data.ipynb
│   └── test-model-GCN.ipynb
│
└── archive_study1/                            # [Git-ignored] Archived loose duplicate scripts from May 2026
    ├── root_legacy/
    └── GNN_jed_legacy/
```

---

## 2. External Data Locations on Jed

| Resource | Path on Jed | Description |
| :--- | :--- | :--- |
| **SWIFT Binary** | `/home/cozma/swift_sims/swift` | Compiled `SWIFT v2026.04` with `--with-ext-potential=MWPotential2014` |
| **Study 1 SWIFT ICs** | `/home/cozma/swift_sims/ICs/` | 75 HDF5 initial conditions from Study 1 (untouched) |
| **Study 1 SWIFT Outputs** | `/home/cozma/swift_sims/outputs/` | 75 directories $\times$ 101 HDF5 snapshots from Study 1 (untouched) |
| **Study 1 Observables** | `/scratch/cozma/study1/obs_all_stars/` | 75 $\times$ 101 original observable matrices (58 GB, content untouched), symlinked from `GNN-jed/obs_all_stars` |
| **Study 2 SWIFT Outputs** | `/scratch/cozma/study2/outputs_unbiased/` | Unbiased $N$-body simulation outputs ($R_p \ge 5\text{ kpc}$), symlinked from `/home/cozma/swift_sims/outputs_unbiased` |
| **Study 2 Bound Labels** | `/scratch/cozma/study2/labels_clean/` | Exact bound labels `B_L_snapshot_*.npy`, symlinked from `GNN-jed/labels_clean` |
| **Study 2 Clean Observables** | `/scratch/cozma/study2/obs_clean/` | Standardized noiseless observable matrices, symlinked from `GNN-jed/obs_clean` |
| **Study 2 Gaia Noisy Observables** | `/scratch/cozma/study2/obs_gaia_noisy/` | Standardized Gaia DR3 noisy observable matrices, symlinked from `GNN-jed/obs_gaia_noisy` |
| **Study 2 Clean Graphs** | `/scratch/cozma/graph_data_clean/` | PyTorch Geometric `.pt` 5D $k$-NN graphs (noiseless) |
| **Study 2 Noisy Graphs** | `/scratch/cozma/graph_data_noisy/` | PyTorch Geometric `.pt` 5D $k$-NN graphs (with Gaia DR3 noise) |
| **Study 2 4D Ablation Graphs** | `/scratch/cozma/graph_data_4obs_<tag>/` | 4-observable $k$-NN graphs for feature-ablation experiments |

> **Note:** `/scratch` is not backed up and may be purged. Copy final Study 2 results (checkpoints, figures, key matrices) to `/home` or another persistent location.
