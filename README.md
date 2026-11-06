# Identifying Potential Escapers with Graph Neural Networks: Recovering Cluster Dynamics from Partial Kinematics

**Author:** Adrian Cozma (adrian.cozma@epfl.ch)  
**Affiliation:** Laboratory of Astrophysics (LASTRO), EPFL, Lausanne, Switzerland

## Overview

Observational studies of globular cluster (GC) stellar streams in the Milky Way are frequently constrained by incomplete 6D phase-space information, specifically the lack of radial line-of-sight velocities ($v_{\mathrm{los}}$) which require expensive spectroscopic follow-up. 

This repository contains the complete numerical and machine-learning pipeline to:
1. Generate observationally motivated initial conditions (ICs) for globular clusters from the Gaia DR3 / STREAMFINDER catalogue of Ibata et al. (2024).
2. Evolve N-body Plummer clusters in a static `MWPotential2014` galactic potential (Bovy 2015) using the **SWIFT** cosmological and astrophysical code on the EPFL HPC cluster (Jed).
3. Classify stars as gravitationally bound, unbound, or potential escapers using the Jacobi radius ($R_J$) and Jacobi energy ($E_J < E_{\mathrm{crit}}$) criteria of Küpper et al. (2010a,b).
4. Train and evaluate Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT) in PyTorch Geometric to recover bound cluster populations using only 5D astrometric observables ($\alpha, \delta, D, \mu_{\alpha^*}, \mu_\delta$), completely blind to $v_{\mathrm{los}}$.

---

## Workflow and Pipeline Structure

The codebase is organized into four main stages.

### Stage 1: Initial Conditions and Backward Orbit Integration (`make_ICs/`)

* **Step 1.1: Observational Data Filtering**
  * **File:** `make_ICs/WorkflowIbata.ipynb`
  * **Input:** `make_ICs/Ibata_catalog.txt` (24,540 stream stars from Ibata et al. 2024)
  * **Action:** Filters the catalogue to retain only stars with complete 6D phase-space measurements.
  * **Output:** `make_ICs/stars_6D_ibata2024.csv` (1,703 stars across 84 streams)

* **Step 1.2: Backward Orbit Integration**
  * **File:** `make_ICs/OrbitIntegration_Ibata.ipynb`
  * **Input:** `make_ICs/stars_6D_ibata2024.csv`
  * **Action:** Computes median 6D coordinates per stream and integrates the orbits backward in time for $2\text{ Gyr}$ in `MWPotential2014` using `galpy`. Filters out unbounded trajectories outside a $100\text{ kpc}$ galactocentric box.
  * **Output:** `make_ICs/SWIFT_seeds.npy` and `make_ICs/orbits/orbit_<ID>.txt` (75 retained orbits)

* **Step 1.3: Plummer Cluster Realization for SWIFT**
  * **Files:** `make_ICs/IC_plummer_swift.py`, `make_ICs/auto_multiple_IC_swift.ipynb`
  * **Input:** `make_ICs/SWIFT_seeds.npy`
  * **Action:** Generates discrete N-body realizations of Plummer spheres ($N \in [20\,000, 50\,000]$, $m = 1.0\,M_\odot$) boosted onto the $t = -2\text{ Gyr}$ orbital positions and velocities.
  * **Output:** `make_ICs/initial_conditions/IC_<ID>.hdf5`

---

### Stage 2: N-Body Simulations with SWIFT

* **Configuration:** `old/params.yml`
* **Action:** Each of the 75 initial conditions is evolved forward for $2\text{ Gyr}$ (101 snapshots per orbit) inside a static `MWPotential2014` potential in SWIFT.
* **Visualization:** `plot_swift_output.py`, `visu_stream_outputs.py`, `final_sims_vizu.ipynb`, and `Stellar_objects/cluster_v4_properties.py` compute 3D $k$-d tree volume densities and project orbital evolution.

---

### Stage 3: Dynamical Labeling and 5D Observable Projection

* **Step 3.1: Jacobi Radius and Energy Criteria**
  * **Files:** `bound_labels.py`, `fraction_bound_stars.ipynb`
  * **Action:** Locates the progenitor cluster core via 3D $k$-NN local density estimation and shrinking spheres, iteratively solves for the Jacobi radius $R_J$ using the projected cylindrical Hessian of `MWPotential2014`, and evaluates the Jacobi energy $E_J$ in the local co-rotating frame against the critical Lagrange energy $E_{\mathrm{crit}} = -3GM / (2R_J)$. Labels bound stars ($r < R_J$ and $E_J < E_{\mathrm{crit}}$) and potential escapers ($r < R_J$ and $E_J \ge E_{\mathrm{crit}}$).
  * **Output:** `GNN-jed/labels_bound_stars/IC_<ID>/B_L_snapshot_<snap>.npy`

* **Step 3.2: Coordinate Transformation, 3D Rotations, and Normalization**
  * **Files:** `observables_matrix.py`, `labels_data_matrices.ipynb`
  * **Action:** Applies 5 random 3D Euler rotations per snapshot for data augmentation, transforms Galactocentric Cartesian vectors into ICRS astrometric observables ($\alpha, \delta, D, \mu_{\alpha^*}, \mu_\delta, v_{\mathrm{los}}$) via `astropy`, centers each stream on the tangent sky plane, and normalizes features before discarding $v_{\mathrm{los}}$.
  * **Output:** `GNN-jed/obs_all_stars/IC_<ID>/obs_matrices_snapshot_<snap>.npy`

---

### Stage 4: Graph Neural Network Training and Evaluation (`GNN-jed/`)

* **Step 4.1: Graph Construction**
  * **File:** `GNN-jed/graph_data.ipynb`
  * **Action:** Builds $k$-nearest-neighbor ($k$-NN) graphs in the reduced 5D astrometric space (`Data(x, edge_index)`).
  * **Output:** `GNN-jed/graph_data/IC_<ID>/graph_data_output_<snap>_rotation_<i>.pt`

* **Step 4.2: GCN and GAT Architectures**
  * **Files:** `GNN-jed/GNN_01.py`, `GNN-jed/trained_models/GCN-02/GNN_02.py`
  * **Action:** Implements 3-layer message-passing networks (`GCNConv` baseline and `GATConv` self-attention architecture) trained with `BCEWithLogitsLoss`, Adam optimizer, and `ReduceLROnPlateau` scheduling on the EPFL Jed cluster. Supports stream-level (`IC_ID`) dataset splitting to prevent spatio-temporal or rotational data leakage between training, validation, and test sets.
  * **Output:** `best_gnn_model_01.pth` (GCN), `best_gnn_model_02.pth` (GAT), and training logs.

* **Step 4.3: Post-Training Evaluation and Spectral Analysis**
  * **Files:** `GNN-jed/trained_models/test-model-GCN.ipynb`, `GNN-jed/trained_models/plot_metric_live.py`
  * **Action:** Evaluates classification metrics across streams ordered by first pericenter passage ($R_{p1}$), generates spatial TP/TN/FP/FN projections, and performs Random Matrix Theory (RMT) Singular Value Decomposition (SVD) spectral analysis on the trained weight matrices.

---

## Requirements

* Python 3.9+
* `numpy`, `scipy`, `pandas`, `matplotlib`, `tqdm`
* `astropy`, `galpy`, `unyt`, `swiftsimio`, `h5py`
* `torch`, `torch_geometric`, `torch_cluster`
