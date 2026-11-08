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
2. Evolves 75 self-consistent *N*-body Plummer globular clusters forward in time for $2.0\text{ Gyr}$ using the cosmological/astrophysical code `SWIFT` (Schaller et al. 2024) on the EPFL HPC cluster **Jed**.
3. Computes exact physical boundness and potential escaper labels from 3D Jacobi energies and tidal radii following Küpper et al. (2010a,b), using a 3D *k*-nearest-neighbor local density estimator (Casertano & Hut 1985) and iterative shrinking spheres to track the surviving core accurately even in heavily stripped streams.
4. Projects the simulations into synthetic 5D *Gaia*-like astrometric observables with data augmentation over random observer orientations and physical-scale standardization.
5. Trains **Graph Convolutional Networks (GCN)** and **Graph Attention Networks (GAT)** on *k*-NN phase-space graphs $(k = 20)$ to recover node-level dynamical boundness directly from incomplete 5D kinematics, using strict stream-level (`IC_ID`) train/validation/test partitioning to evaluate generalization to unseen orbital families.

---

## Repository Structure

The repository is organized to separate core physical/feature-engineering modules (`src/`), executable production pipelines (`scripts/`), initial condition generation (`make_ICs/`), visualization utilities (`visualization/`), and interactive analysis notebooks (`notebooks/`):

```text
.
├── assets/
│   └── figures/                         # Example figures from simulations, dynamics, and GNN evaluation
├── configs/
│   ├── params.yml                       # SWIFT runtime configuration (MilkyWayPotential2014, leapfrog/gravity)
│   └── dependency_graph_0.csv           # SWIFT task-level dependency graph from HPC execution
├── make_ICs/
│   ├── Ibata_catalog.txt                # Gaia DR3 stellar stream catalog (Ibata et al. 2024)
│   ├── stars_6D_ibata2024.csv           # Filtered 6D stream stars used to define progenitor orbits
│   ├── OrbitIntegration_Ibata.ipynb     # Backward orbit integration (-2 Gyr) in galpy MWPotential2014
│   ├── WorkflowIbata.ipynb              # Catalog processing and progenitor seed extraction workflow
│   ├── SWIFT_seeds.npy                  # 6D phase-space seeds at t = -2 Gyr for the 75 stream progenitors
│   ├── IC_plummer_swift.py              # Isotropic/anisotropic Plummer sphere HDF5 generator for SWIFT
│   ├── auto_multiple_IC_swift.ipynb     # Batch generation of the 75 initial condition directories
│   └── orbits/                          # Integrated 2 Gyr orbital trajectories (orbit_1.txt ... orbit_87.txt)
├── src/
│   ├── bound_labels.py                  # Core-density centering + Jacobi energy & tidal radius labeling
│   ├── observables_matrix.py            # 3D rotation augmentation, Astropy ICRS projection, physical scaling
│   └── Stellar_objects/                 # Utilities for King/Plummer cluster profiles and LOS dispersion
├── scripts/
│   ├── extract_observables.py           # Batch extraction of 5D/6D observable matrices from SWIFT HDF5 outputs
│   ├── train_gcn.py                     # Baseline 3-layer Graph Convolutional Network (GCN) training script
│   └── train_gat.py                     # Multi-head Graph Attention Network (GAT) with stream-level IC split
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
        ├── GCN-01/                      # Trained baseline GCN weights (best_gnn_model_01.pth) & logs
        └── GCN-02/                      # Trained multi-head GAT weights (best_gnn_model_02.pth), script & logs
```

*Note:* Bulk simulation snapshots (`outputs_SWIFT_jed/`, ~35 GB), raw pre-computed `.npy` observable/label arrays (`GNN-jed/obs_all_stars/`, `GNN-jed/labels_bound_stars/`), and serialized `.pt` graph datasets (`GNN-jed/graph_data/`, ~23 GB) are excluded from version control via `.gitignore`.

---

## Physical & Methodological Pipeline

### 1. Progenitor Orbit Reconstruction & *N*-Body Simulations (`make_ICs/`, `configs/`)

Starting from the *Gaia* DR3 stellar stream catalog of Ibata et al. (2024), stars with complete 6D phase-space measurements are grouped by stream identifier (`Track`). Representative present-day stream anchors are integrated backward in time for $\Delta t = -2.0\text{ Gyr}$ using `galpy` (Bovy 2015) inside `MWPotential2014` — a three-component potential comprising a Miyamoto-Nagai disk, a power-law bulge with exponential cut-off, and an NFW dark matter halo, calibrated to $R_0 = 8.0\text{ kpc}$ and $V_c = 220\text{ km s}^{-1}$.

| Present-Day Stream Anchors $(t = 0\text{ Gyr})$ | Backward-Integrated Progenitor Seeds $(t = -2.0\text{ Gyr})$ |
| :---: | :---: |
| ![Present-day positions](assets/figures/positions_today.png) | ![Backward-integrated seeds](assets/figures/seeds_positions.png) |

At each $t = -2.0\text{ Gyr}$ seed position, an isotropic $(q = 0)$ Plummer sphere (Plummer 1911) of total mass $M = 10^4\,M_\odot$, scale radius $a = 4\text{ pc}$, and $N \in [2500, 3500]$ equal-mass particles is initialized via `make_ICs/IC_plummer_swift.py`:

$$
\rho(r) = \frac{3M}{4\pi a^3}\left(1 + \frac{r^2}{a^2}\right)^{-5/2}, \qquad \Phi(r) = -\frac{GM}{\sqrt{r^2 + a^2}}.
$$

Each cluster is evolved forward for $2.0\text{ Gyr}$ in `SWIFT` (Schaller et al. 2024) using `configs/params.yml` with 101 HDF5 snapshots per simulation $(\Delta t_{\mathrm{snap}} = 20\text{ Myr})$ and a comoving Plummer-equivalent softening length $\epsilon = 1\text{ pc}$.

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
*Evolution of the Galactocentric radius $R(t)$ over $2.0\text{ Gyr}$ across the 75 simulated clusters, grouped into six panels by their first perigalacticon distance $R_{p1}$ and color-coded by the instantaneous fraction of bound stars.*

---

### 3. Synthetic 5D *Gaia* Observables & *k*-NN Graph Construction (`src/observables_matrix.py`)

For each snapshot, the 3D Galactocentric particle coordinates are rotated by $n_{\mathrm{rot}} = 5$ random 3D Euler matrices $\mathbf{R} \in \mathrm{SO}(3)$ to augment viewing angles, transformed into the heliocentric International Celestial Reference System (ICRS) via `astropy.coordinates.SkyCoord`, and restricted to the 5D astrometric feature vector:

$$
\mathbf{x}_i = \left(\Delta\alpha_i\cos\delta_c,\; \Delta\delta_i,\; \Delta\varpi_i,\; \Delta\mu_{\alpha^*,i},\; \Delta\mu_{\delta,i}\right) \in \mathbb{R}^5.
$$

In `src/observables_matrix.py`, features are centered on the robust stream median (handling the $0^\circ / 360^\circ$ Right Ascension wrap via circular trigonometry) and normalized by fixed reference physical scales (`mode="physical"`) rather than per-snapshot unit-variance Z-scores, preserving the contrast in angular and kinematic compactness between bound cores and dispersed tidal tails. A directed *k*-nearest-neighbor graph $(k = 20)$ is constructed in this 5D feature space using `torch_geometric.nn.knn_graph`.

---

### 4. Graph Neural Network Architectures & Evaluation (`scripts/`, `notebooks/`)

We compare two message-passing architectures implemented in PyTorch Geometric:
1. **Baseline GCN (`scripts/train_gcn.py`):** A 3-layer Graph Convolutional Network (`5 -> 16 -> 32 -> 1`, Kipf & Welling 2017) with isotropic neighborhood aggregation.
2. **Multi-Head GAT (`scripts/train_gat.py`):** A 3-layer Graph Attention Network (Veličković et al. 2018) with multi-head attention (`GATConv(5, 16, heads=4)` to `GATConv(64, 32, heads=2)` to `GATConv(64, 1, heads=1, concat=False)`), Exponential Linear Unit (`ELU`) activations, feature dropout $(p = 0.2)$, and residual linear skip connections (`Linear(5, 64)`, `Linear(64, 64)`) to prevent over-smoothing and preserve each star's local kinematics relative to its neighbors.

| Training & Validation Loss Convergence (GCN vs. GAT) | Weight Matrix Singular Value Spectrum (RMT) |
| :---: | :---: |
| ![Training and validation loss curves](assets/figures/live_plots_GNN.png) | ![Spectral analysis of GNN weights](assets/figures/spectral_analysis_GNN.png) |

At $t = 2.0\text{ Gyr}$ (snapshot 100), the multi-head GAT model achieves **96.26% mean node accuracy** across all 75 streams, accurately separating surviving bound cores from stripped tidal tails while concentrating its remaining errors on marginal systems undergoing final dissolution at intermediate perigalacticon $(R_{p1} \in [8, 18]\text{ kpc})$.

| Dissolved Stream (`IC_1`, $R_{p1} = 1.9\text{ kpc}$, 100% Accuracy) | Surviving Bound Core (`IC_81`, $R_{p1} = 16.5\text{ kpc}$, 98.11% Accuracy) |
| :---: | :---: |
| ![GAT predictions on IC_1 at snapshot 100](assets/figures/prediction_GAT_ID1_output100.png) | ![GAT predictions on IC_81 at snapshot 100](assets/figures/prediction_GAT_ID81_output100.png) |

![GAT Error Percentage by First Perigalacticon](assets/figures/error_percentage_by_perigalacticon.png)
*Classification error percentage at $t = 2.0\text{ Gyr}$ (snapshot 100) across all 75 simulated streams ordered by first perigalacticon distance $R_{p1}$. Streams with small $R_{p1}$ are completely disrupted early on, whereas clusters with intermediate $R_{p1}$ retain marginally bound cores rich in potential escapers.*

---

## Installation & Usage

### Dependencies

Create a Python 3.10+ environment with the required astrophysics and deep-learning packages:

```bash
pip install numpy scipy pandas matplotlib h5py astropy galpy swiftsimio unyt cartopy
pip install torch torch_geometric torch_cluster
```

### 1. Computing Dynamical Labels & 5D Observable Matrices

To extract ground-truth bound labels (`src/bound_labels.py`) and standardized 5D observable matrices (`src/observables_matrix.py`) from raw `SWIFT` HDF5 snapshots:

```bash
python scripts/extract_observables.py
```

Or interactively via `notebooks/labels_data_matrices.ipynb`. In Python, `compute_bound_stars` can also return the boolean mask of **potential escapers** satisfying $r_{3D} \lt R_J$ and $E_J \ge E_{\mathrm{crit}}$:

```python
from src.bound_labels import compute_bound_stars

is_bound, is_potential_escaper, x_rel, y_rel, z_rel, vx_rel, vy_rel, vz_rel, rj, m_bound, e_j, e_crit = (
    compute_bound_stars(x, y, z, vx, vy, vz, masses, return_potential_escapers=True)
)
```

### 2. Constructing *k*-NN Graphs

Use `notebooks/graph_data.ipynb` to convert the standardized 5D observable matrices (`obs_matrices_snapshot_*.npy`) and bound labels (`B_L_snapshot_*.npy`) into PyTorch Geometric `Data` objects with *k*-NN connectivity.

### 3. Training the GAT / GCN Models

Run `scripts/train_gat.py` to train the multi-head GAT model with stream-level (`IC_ID`) partitioning (ensuring no snapshot or rotation of a validation/test stream leaks into the training set) and optional positive-class loss weighting:

```bash
python scripts/train_gat.py \
    --data_folder /path/to/graph_data \
    --split_mode ic \
    --epochs 70 \
    --lr 0.005 \
    --batch_size 64 \
    --pos_weight 2.91 \
    --save_model best_gnn_model_02.pth \
    --log_file training_metrics.log
```

To plot live training and validation loss curves from a log file:

```bash
python visualization/plot_metric_live.py
```

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
11. **Kipf, T. N., & Welling, M. (2017).** *Semi-Supervised Classification with Graph Convolutional Networks.* International Conference on Learning Representations (ICLR 2017). [arXiv:1609.02907](https://arxiv.org/abs/1609.02907)
12. **Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018).** *Graph Attention Networks.* International Conference on Learning Representations (ICLR 2018). [arXiv:1710.10903](https://arxiv.org/abs/1710.10903)
13. **Alon, U., & Yahav, E. (2021).** *On the Bottleneck of Graph Neural Networks and its Practical Implications.* International Conference on Learning Representations (ICLR 2021). [arXiv:2006.05205](https://arxiv.org/abs/2006.05205)
14. **Martin, C. H., & Mahoney, M. W. (2021).** *Implicit Self-Regularization in Deep Neural Networks: Evidence from Random Matrix Theory and Implications for Learning.* Journal of Machine Learning Research, 22(165), 1–73. [arXiv:1810.01075](https://arxiv.org/abs/1810.01075)
