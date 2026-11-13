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
